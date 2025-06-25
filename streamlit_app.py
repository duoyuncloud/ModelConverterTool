"""
Streamlit UI for Model Converter Tool

Run with:
    streamlit run streamlit_app.py

Requires: streamlit, requests
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import pyperclip
import hashlib

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Model Converter Tool", layout="centered")
st.title("Model Converter Tool")

# --- Sidebar: System Info ---
st.sidebar.header("System Info")
mlx_available = False
try:
    system_info = requests.get(f"{API_URL}/system/info").json()
    mlx_available = system_info.get("mlx_available", False)
    st.sidebar.json(system_info)
    if not mlx_available:
        st.sidebar.info("MLX conversion not available on this backend.")
except Exception:
    st.sidebar.warning("Could not fetch system info.")

# --- Sidebar: Capabilities ---
st.sidebar.header("Quantization Capabilities")

@st.cache_data(ttl=30)
def get_capabilities():
    try:
        return requests.get(f"{API_URL}/capabilities").json()
    except Exception:
        return {}

@st.cache_data(ttl=600)
def get_supported_formats():
    try:
        resp = requests.get(f"{API_URL}/formats")
        if resp.status_code == 200:
            return resp.json().get("formats", [])
        else:
            st.sidebar.error(f"Failed to fetch formats: {resp.status_code} - {resp.text}")
            return []
    except Exception as e:
        st.sidebar.error(f"Error fetching formats: {e}")
        return []

@st.cache_data(ttl=600)
def get_supported_model_types():
    try:
        resp = requests.get(f"{API_URL}/model-types")
        if resp.status_code == 200:
            return resp.json().get("model_types", [])
        else:
            st.sidebar.error(f"Failed to fetch model types: {resp.status_code} - {resp.text}")
            return []
    except Exception as e:
        st.sidebar.error(f"Error fetching model types: {e}")
        return []

formats = get_supported_formats()
model_types = get_supported_model_types()

# --- Robust sync for dynamic formats ---
formats_hash = hashlib.md5(json.dumps(formats, sort_keys=True).encode()).hexdigest()
if 'formats_hash' not in st.session_state:
    st.session_state['formats_hash'] = formats_hash
else:
    if st.session_state['formats_hash'] != formats_hash:
        # Formats list changed, reset target_format and rerun
        st.session_state['formats_hash'] = formats_hash
        st.session_state['target_format'] = formats[0] if formats else ''
        st.experimental_rerun()

# Debug information
st.sidebar.write("Debug Info:")
st.sidebar.write(f"Formats count: {len(formats)}")
st.sidebar.write(f"Model types count: {len(model_types)}")
if len(formats) == 0:
    st.sidebar.error("No formats loaded!")
if len(model_types) == 0:
    st.sidebar.error("No model types loaded!")

# Remove 'mlx' from formats if not available
if not mlx_available and "mlx" in formats:
    formats = [f for f in formats if f != "mlx"]

# Remove gptq/awq if not supported by backend
capabilities = get_capabilities()
if not capabilities.get('gptq', False) and 'gptq' in formats:
    formats = [f for f in formats if f != 'gptq']
if not capabilities.get('awq', False) and 'awq' in formats:
    formats = [f for f in formats if f != 'awq']

# --- Conversion Form ---
st.header("Model Conversion")
st.markdown("Convert a HuggingFace model or local file to another format. Monitor progress and download the result.")

format_tooltips = {
    "onnx": "ONNX: Open Neural Network Exchange format for interoperability.",
    "torchscript": "TorchScript: PyTorch's serializable, optimizable format.",
    "fp16": "FP16: Half-precision weights for reduced memory.",
    "hf": "HuggingFace: Standard Transformers format.",
    "gptq": "GPTQ: Quantized LLM format (requires CUDA).",
    "awq": "AWQ: Activation-aware quantization (requires CUDA).",
    "gguf": "GGUF: Llama.cpp format for efficient inference.",
    "mlx": "MLX: Apple MLX format for Mac.",
    "test": "Test: Dummy output for testing."
}

# Model source 选择和文件上传都放在表单外部
model_source = st.radio("Select model source:", ["HuggingFace", "Local File"], horizontal=True, key="model_source")

hf_model_name = None
file_path = None
uploaded_file = None

if st.session_state.get("model_source", "HuggingFace") == "HuggingFace":
    hf_model_name = st.text_input("HuggingFace Model Name", key="hf_model_name")
else:
    uploaded_file = st.file_uploader("Upload Model File", type=None, help="Upload a model file from your computer.")
    if uploaded_file is not None:
        # 上传逻辑可在表单提交时处理
        pass

with st.form("convert_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        # Ensure target_format is always valid
        if (
            'target_format' not in st.session_state or
            st.session_state['target_format'] not in formats
        ):
            st.session_state['target_format'] = formats[0] if formats else ''
        format_labels = [f"{fmt} - {format_tooltips.get(fmt, '')}" for fmt in formats]
        st.selectbox(
            "Target Format",
            formats,
            format_func=lambda x: f"{x} - {format_tooltips.get(x, '')}",
            help="Select the output format. Hover for details.",
            key="target_format"
        )
    with col2:
        model_type = st.selectbox("Model Type", model_types, index=0, help="Select the model type.")
    # Warn if user somehow selects a disabled format
    if (st.session_state["target_format"] == 'gptq' and not capabilities.get('gptq', False)) or (st.session_state["target_format"] == 'awq' and not capabilities.get('awq', False)):
        st.warning(f"{st.session_state['target_format'].upper()} quantization is not available on this backend. Please select another format.")
    # Disable convert button if required fields are missing or format is not available
    if st.session_state.get("model_source", "HuggingFace") == "HuggingFace":
        convert_disabled = not st.session_state.get("hf_model_name", "")
    else:
        convert_disabled = uploaded_file is None
    # --- Dynamic advanced options UI ---
    ADVANCED_DEFAULTS = {
        "gptq": {"bits": 4, "group_size": 128},
        "awq": {"bits": 4, "group_size": 120},
        "onnx": {"opset_version": 11, "atol": 1e-4},
        "tensorrt": {"precision": "fp16"},
        "openvino": {"precision": "FP16"},
    }
    # Get defaults for selected format
    adv_defaults = ADVANCED_DEFAULTS.get(st.session_state["target_format"], {})
    adv_params = {}
    # Show sliders/fields for known options
    if st.session_state["target_format"] in ["gptq", "awq"]:
        adv_params["bits"] = st.slider("Bits", 2, 8, int(adv_defaults.get("bits", 4)), help="Quantization bits")
        adv_params["group_size"] = st.slider("Group Size", 8, 2048, int(adv_defaults.get("group_size", 128 if st.session_state["target_format"]=="gptq" else 120)), step=8, help="Quantization group size")
    elif st.session_state["target_format"] == "onnx":
        adv_params["opset_version"] = st.slider("ONNX Opset Version", 7, 20, int(adv_defaults.get("opset_version", 11)), help="ONNX opset version")
        adv_params["atol"] = st.number_input("ONNX ATOL", value=float(adv_defaults.get("atol", 1e-4)), help="ONNX absolute tolerance")
    elif st.session_state["target_format"] == "tensorrt":
        adv_params["precision"] = st.selectbox("TensorRT Precision", ["fp16", "int8"], index=0, help="TensorRT precision mode")
    elif st.session_state["target_format"] == "openvino":
        adv_params["precision"] = st.selectbox("OpenVINO Precision", ["FP16", "FP32"], index=0, help="OpenVINO precision mode")
    # --- Custom key-value pairs ---
    st.markdown("**Custom Advanced Options** (optional)")
    st.info("💡 **Tip**: Use descriptive option names like 'bits', 'group_size', 'precision'. Avoid single letters or numbers.")
    custom_keys = st.session_state.get("custom_keys", [""])
    custom_values = st.session_state.get("custom_values", [""])
    n_custom = st.number_input("Number of custom options", min_value=0, max_value=10, value=len(custom_keys), step=1, key="n_custom")
    # Adjust list lengths
    if len(custom_keys) < n_custom:
        custom_keys += [""] * (n_custom - len(custom_keys))
        custom_values += [""] * (n_custom - len(custom_values))
    elif len(custom_keys) > n_custom:
        custom_keys = custom_keys[:n_custom]
        custom_values = custom_values[:n_custom]
    # Render custom fields
    for i in range(n_custom):
        cols = st.columns([1,2])
        custom_keys[i] = cols[0].text_input(f"Key {i+1}", value=custom_keys[i], key=f"custom_key_{i}")
        custom_values[i] = cols[1].text_input(f"Value {i+1}", value=custom_values[i], key=f"custom_value_{i}")
    # Save to session state
    st.session_state["custom_keys"] = custom_keys
    st.session_state["custom_values"] = custom_values
    # Add custom options to adv_params
    custom_warnings = []
    invalid_options = []
    # Common valid option names for different formats
    valid_options_by_format = {
        "gptq": ["bits", "group_size", "desc_act", "static_groups", "damp_percent", "sym"],
        "awq": ["bits", "group_size", "zero_point", "scale_method"],
        "onnx": ["opset_version", "atol", "rtol", "dynamic_axes", "input_names", "output_names"],
        "tensorrt": ["precision", "max_workspace_size", "fp16_mode", "int8_mode"],
        "openvino": ["precision", "device", "cpu_extension"],
        "mlx": ["dtype", "compile"],
        "gguf": ["model_type", "context_length", "rope_freq_base"]
    }
    for k, v in zip(custom_keys, custom_values):
        if k.strip():
            # Validate key name
            if not k.strip().replace('_', '').replace('-', '').isalnum():
                invalid_options.append(f"Invalid key name: '{k.strip()}' (use only letters, numbers, underscores, hyphens)")
                continue
            # Check if it's a known option for the selected format
            valid_options = valid_options_by_format.get(st.session_state["target_format"], [])
            if k.strip() not in valid_options and len(valid_options) > 0:
                custom_warnings.append(f"'{k.strip()}' is not a standard option for {st.session_state['target_format']} format. Valid options: {', '.join(valid_options)}")
            # Try to parse as int/float/bool, else keep as string
            try:
                v_parsed = json.loads(v)
            except json.JSONDecodeError:
                # If it's not valid JSON, treat as string
                v_parsed = v
                # Warn about potential issues
                if v.strip() and not v.strip().startswith('"'):
                    custom_warnings.append(f"'{k.strip()}': '{v}' will be treated as string (not JSON)")
            except Exception as e:
                # Show error for other parsing issues
                invalid_options.append(f"Error parsing value for '{k.strip()}': {e}")
                continue
            adv_params[k.strip()] = v_parsed
    # Show warnings for custom options
    if custom_warnings:
        st.warning("⚠️ **Custom Options Warnings:**")
        for warning in custom_warnings:
            st.write(f"• {warning}")
    # Show errors for invalid options
    if invalid_options:
        st.error("❌ **Invalid Custom Options:**")
        for error in invalid_options:
            st.write(f"• {error}")
        st.error("Please fix the invalid options above before proceeding.")
        st.stop()
    # Show JSON preview
    st.subheader("JSON Preview")
    st.json(adv_params)
    # Convert button (inside form)
    submitted = st.form_submit_button("🚀 Start Conversion", type="primary", disabled=convert_disabled)

# Show summary only after form is submitted
if 'submitted' in locals() and submitted:
    st.markdown("### Conversion Request Summary")
    st.write({
        "model_source": st.session_state.get("model_source", "HuggingFace"),
        "target_format": st.session_state["target_format"],
        "model_type": model_type
    })

# Handle form submission outside the form
if submitted:
    # Check for invalid options before proceeding
    if invalid_options:
        st.error("❌ Cannot start conversion with invalid options. Please fix the errors above.")
        st.stop()

    try:
        # Validate custom options before sending
        if custom_warnings:
            st.warning("⚠️ Some custom options may cause issues. Proceeding anyway...")

        # Prepare data for conversion
        api_data = {
            "target_format": st.session_state["target_format"],
            "model_type": model_type,
            "conversion_params": json.dumps(adv_params)
        }

        if st.session_state.get("model_source", "HuggingFace") == "HuggingFace":
            if not st.session_state.get("hf_model_name", ""):
                st.error("Please enter a HuggingFace model name.")
                st.stop()
            api_data["model_name"] = st.session_state["hf_model_name"]
        else:
            if not file_path:
                st.error("Please upload a model file.")
                st.stop()
            api_data["file_path"] = file_path

        print("API DATA SENT:", api_data)
        response = requests.post(f"{API_URL}/convert", data=api_data)

        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            st.success(f"✅ Conversion started! Task ID: {task_id}")

            # Store task info for status checking
            st.session_state.current_task = {
                "task_id": task_id,
                "model_name": st.session_state.get("hf_model_name", "") or file_path,
                "target_format": st.session_state["target_format"],
                "start_time": datetime.now()
            }
        else:
            error_msg = response.json().get("detail", "Unknown error")
            st.error(f"❌ Failed to start conversion: {error_msg}")

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")

# Status checking section
if "current_task" in st.session_state:
    st.subheader("📊 Conversion Status")
    task = st.session_state.current_task
    
    # Check status
    try:
        status_response = requests.get(f"{API_URL}/status/{task['task_id']}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            
            # Display status
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", status_data.get("status", "UNKNOWN"))
            with col2:
                st.metric("Progress", f"{status_data.get('progress', 0)}%")
            with col3:
                st.metric("Format", task['target_format'])
            
            # Display message
            message = status_data.get("message", "")
            if message:
                st.info(f"💬 {message}")
            
            # Handle different statuses
            if status_data.get("status") == "SUCCESS":
                result_path = status_data.get("result", "")
                st.success(f"🎉 Conversion completed successfully!")
                st.download_button(
                    label="📥 Download Result",
                    data=f"Result saved to: {result_path}",
                    file_name=f"converted_model.{task['target_format']}",
                    mime="application/octet-stream"
                )
                # Clear task after successful completion
                if st.button("🔄 Start New Conversion"):
                    del st.session_state.current_task
                    st.rerun()
                    
            elif status_data.get("status") == "FAILED":
                error_info = status_data.get("error", "Unknown error")
                st.error(f"❌ Conversion failed: {error_info}")
                # Show error log inline
                try:
                    log_resp = requests.get(f"{API_URL}/error_log/{task['task_id']}")
                    if log_resp.status_code == 200:
                        error_log = log_resp.content.decode()
                        st.code(error_log, language="text")
                        st.button("📋 Copy Error Log", on_click=pyperclip.copy, args=(error_log,))
                    else:
                        st.warning("No error log found for this task.")
                except Exception as e:
                    st.error(f"Failed to fetch error log: {e}")
                # Download error log button
                if st.button("📄 Download Error Log"):
                    try:
                        log_resp = requests.get(f"{API_URL}/error_log/{task['task_id']}")
                        if log_resp.status_code == 200:
                            st.download_button(
                                label="Download Error Log as TXT",
                                data=log_resp.content,
                                file_name=f"error_log_{task['task_id']}.txt",
                                mime="text/plain"
                            )
                        else:
                            st.warning("No error log found for this task.")
                    except Exception as e:
                        st.error(f"Failed to download error log: {e}")
                # Download full task history
                if st.button("🗂️ Download Full Task History (JSON)"):
                    try:
                        hist_resp = requests.get(f"{API_URL}/task_history/{task['task_id']}")
                        if hist_resp.status_code == 200:
                            st.download_button(
                                label="Download Task History as JSON",
                                data=hist_resp.content,
                                file_name=f"task_history_{task['task_id']}.json",
                                mime="application/json"
                            )
                        else:
                            st.warning("No task history found for this task.")
                    except Exception as e:
                        st.error(f"Failed to download task history: {e}")
                if st.button("🔄 Try Again"):
                    del st.session_state.current_task
                    st.rerun()
            elif status_data.get("status") in ["PENDING", "STARTED"]:
                st.info("⏳ Conversion in progress...")
                # Show progress bar and message
                progress = status_data.get("progress", 0)
                st.progress(progress)
                st.write(status_data.get("message", ""))
                # Show timeline of progress messages
                try:
                    timeline_resp = requests.get(f"{API_URL}/progress_timeline/{task['task_id']}")
                    if timeline_resp.status_code == 200:
                        timeline = timeline_resp.json()
                        st.markdown("### Progress Timeline")
                        for entry in timeline:
                            st.write(f"[{entry['timestamp']}] {entry['progress']}% - {entry['message']}")
                    else:
                        st.info("No timeline available yet.")
                except Exception as e:
                    st.error(f"Failed to fetch progress timeline: {e}")
                # Auto-refresh every 2 seconds
                st.button("🔄 Refresh Now", on_click=st.rerun)
                time.sleep(2)
                st.rerun()
            # Download full task history for completed tasks
            if status_data.get("status") == "SUCCESS":
                if st.button("🗂️ Download Full Task History (JSON)"):
                    try:
                        hist_resp = requests.get(f"{API_URL}/task_history/{task['task_id']}")
                        if hist_resp.status_code == 200:
                            st.download_button(
                                label="Download Task History as JSON",
                                data=hist_resp.content,
                                file_name=f"task_history_{task['task_id']}.json",
                                mime="application/json"
                            )
                        else:
                            st.warning("No task history found for this task.")
                    except Exception as e:
                        st.error(f"Failed to download task history: {e}")
        else:
            st.error(f"Failed to get status: {status_response.status_code}")
            
    except Exception as e:
        st.error(f"Error checking status: {e}")
        
    # Add a button to clear the task
    if st.button("🗑️ Clear Task"):
        del st.session_state.current_task
        st.rerun()

# --- Sidebar: Task History Table ---
if 'task_history_table' not in st.session_state:
    st.session_state['task_history_table'] = []
# Add current task to history if completed/failed
if 'current_task' in st.session_state and st.session_state.current_task.get('task_id'):
    task_id = st.session_state.current_task['task_id']
    # Only add if not already in table
    if not any(t['task_id'] == task_id for t in st.session_state['task_history_table']):
        st.session_state['task_history_table'].append({
            'task_id': task_id,
            'model': st.session_state.current_task.get('model_name'),
            'format': st.session_state.current_task.get('target_format'),
            'start_time': str(st.session_state.current_task.get('start_time')),
            'status': status_data.get('status', 'UNKNOWN')
        })
# Filtering/searching UI
st.sidebar.header("Recent Tasks")
filter_status = st.sidebar.selectbox("Filter by Status", ["All"] + sorted({t['status'] for t in st.session_state['task_history_table']}), key="filter_status")
filter_model = st.sidebar.text_input("Search Model Name", "", key="filter_model")
filter_format = st.sidebar.text_input("Search Format", "", key="filter_format")
filtered_tasks = [t for t in st.session_state['task_history_table']
                  if (filter_status == "All" or t['status'] == filter_status)
                  and (filter_model.lower() in (t['model'] or '').lower())
                  and (filter_format.lower() in (t['format'] or '').lower())]
if filtered_tasks:
    import pandas as pd
    df = pd.DataFrame(filtered_tasks)
    st.sidebar.dataframe(df, use_container_width=True)
    # Quick actions
    selected = st.sidebar.selectbox("Select Task for Actions", df['task_id'] if not df.empty else [], key="selected_task_id")
    if selected:
        st.sidebar.write(f"Task: {selected}")
        if st.sidebar.button("View Task History JSON"):
            try:
                hist_resp = requests.get(f"{API_URL}/task_history/{selected}")
                if hist_resp.status_code == 200:
                    json_str = hist_resp.content.decode()
                    st.sidebar.code(json_str, language="json")
                    st.sidebar.button("📋 Copy Task History", on_click=pyperclip.copy, args=(json_str,))
                    # Inline log/timeline search
                    search_term = st.sidebar.text_input("Search Timeline/Log", "", key="search_timeline_log")
                    import json as _json
                    hist = _json.loads(json_str)
                    timeline = hist.get("progress_timeline", [])
                    error_log = hist.get("error_log", "")
                    if search_term:
                        st.sidebar.markdown("**Timeline Matches:**")
                        for entry in timeline:
                            if search_term.lower() in entry['message'].lower():
                                st.sidebar.write(f"[{entry['timestamp']}] {entry['progress']}% - {entry['message']}")
                        if error_log and search_term.lower() in error_log.lower():
                            st.sidebar.markdown("**Error Log Match:**")
                            st.sidebar.code(error_log, language="text")
                else:
                    st.sidebar.warning("No task history found for this task.")
            except Exception as e:
                st.sidebar.error(f"Failed to fetch task history: {e}")
        if st.sidebar.button("Download Task History JSON"):
            try:
                hist_resp = requests.get(f"{API_URL}/task_history/{selected}")
                if hist_resp.status_code == 200:
                    st.sidebar.download_button(
                        label="Download Task History as JSON",
                        data=hist_resp.content,
                        file_name=f"task_history_{selected}.json",
                        mime="application/json"
                    )
                else:
                    st.sidebar.warning("No task history found for this task.")
            except Exception as e:
                st.sidebar.error(f"Failed to download task history: {e}")
# Export all task history at once
if st.sidebar.button("Export All Task History (JSON)"):
    import json as _json
    all_ids = [t['task_id'] for t in st.session_state['task_history_table']]
    all_histories = []
    for tid in all_ids:
        try:
            hist_resp = requests.get(f"{API_URL}/task_history/{tid}")
            if hist_resp.status_code == 200:
                all_histories.append(hist_resp.json())
        except Exception:
            continue
    st.sidebar.download_button(
        label="Download All Task Histories as JSON",
        data=_json.dumps(all_histories, indent=2),
        file_name="all_task_histories.json",
        mime="application/json"
    )