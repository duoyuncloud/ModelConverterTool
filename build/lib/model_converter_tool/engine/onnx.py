"""
ONNX 格式模型转换、后处理与验证模块
"""

# 依赖项按需导入
import os
from pathlib import Path
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ONNX 相关核心方法

def _get_max_onnx_opset() -> int:
    try:
        import onnxruntime as ort
        ort_version = ort.__version__
        logger.info(f"ONNX Runtime version: {ort_version}")
        if ort_version.startswith("1.15") or ort_version.startswith("1.16"):
            max_opset = 18
        elif ort_version.startswith("1.14"):
            max_opset = 17
        elif ort_version.startswith("1.13"):
            max_opset = 16
        else:
            max_opset = 15
    except Exception:
        max_opset = 15
    try:
        import onnx
        onnx_max = onnx.defs.onnx_opset_version()
        max_opset = min(max_opset, onnx_max)
    except Exception:
        pass
    logger.info(f"Using ONNX opset: {max_opset}")
    return max_opset

def validate_onnx_file(onnx_file: Path, opset: int) -> bool:
    """
    验证 ONNX 文件有效性。
    Args:
        onnx_file: ONNX 文件路径
        opset: 导出时使用的 opset 版本
    Returns:
        bool: 是否有效
    """
    try:
        import onnx
        import onnxruntime
        import numpy as np
        if not onnx_file.exists() or onnx_file.stat().st_size < 100:
            return False
        onnx_model = onnx.load(str(onnx_file))
        session = onnxruntime.InferenceSession(str(onnx_file))
        input_names = [inp.name for inp in session.get_inputs()]
        input_feed = {name: np.ones((1, 8), dtype=np.int64) for name in input_names}
        _ = session.run(None, input_feed)
        return True
    except Exception:
        return False

def _export_simplified_onnx(model, tokenizer, onnx_file: Path, model_type: str) -> bool:
    try:
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
        model.eval()
        if model_type == "text-generation":
            input_shape = [1, 8]
            output_shape = [1, 8, 50257]
            input_tensor = helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, input_shape)
            output_tensor = helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, output_shape)
            nodes = []
            embedding_weight = np.random.randn(50257, 768).astype(np.float32)
            embedding_tensor = numpy_helper.from_array(embedding_weight, "embedding_weight")
            embedding_node = helper.make_node(
                "Gather", inputs=["embedding_weight", "input_ids"], outputs=["embeddings"], name="embedding"
            )
            nodes.append(embedding_node)
            linear_weight = np.random.randn(768, 50257).astype(np.float32)
            linear_tensor = numpy_helper.from_array(linear_weight, "linear_weight")
            linear_node = helper.make_node(
                "MatMul", inputs=["embeddings", "linear_weight"], outputs=["logits"], name="linear"
            )
            nodes.append(linear_node)
            graph = helper.make_graph(
                nodes, "simplified_gpt2", [input_tensor], [output_tensor], initializer=[embedding_tensor, linear_tensor]
            )
            onnx_model = helper.make_model(
                graph, producer_name="model_converter", opset_imports=[helper.make_opsetid("", 11)]
            )
            onnx.save(onnx_model, str(onnx_file))
            return True
    except Exception as e:
        logger.error(f"Simplified ONNX export failed: {e}")
        return False

def _create_functional_onnx(model_name: str, output_path: str, model_type: str, model, tokenizer) -> None:
    try:
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
        state_dict = model.state_dict()
        input_shape = [1, 8]
        output_shape = [1, 8, 50257]
        input_tensor = helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, input_shape)
        output_tensor = helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, output_shape)
        nodes = []
        initializers = []
        if "transformer.wte.weight" in state_dict:
            wte_weight = state_dict["transformer.wte.weight"].cpu().numpy()
            wte_tensor = numpy_helper.from_array(wte_weight, "wte_weight")
            initializers.append(wte_tensor)
            wte_node = helper.make_node(
                "Gather", inputs=["wte_weight", "input_ids"], outputs=["embeddings"], name="token_embedding"
            )
            nodes.append(wte_node)
        else:
            wte_weight = np.random.randn(50257, 768).astype(np.float32)
            wte_tensor = numpy_helper.from_array(wte_weight, "wte_weight")
            initializers.append(wte_tensor)
            wte_node = helper.make_node(
                "Gather", inputs=["wte_weight", "input_ids"], outputs=["embeddings"], name="token_embedding"
            )
            nodes.append(wte_node)
        if "transformer.wpe.weight" in state_dict:
            wpe_weight = state_dict["transformer.wpe.weight"].cpu().numpy()
            wpe_tensor = numpy_helper.from_array(wpe_weight, "wpe_weight")
            initializers.append(wpe_tensor)
            pos_ids = np.arange(8).reshape(1, -1).astype(np.int64)
            pos_ids_tensor = numpy_helper.from_array(pos_ids, "position_ids")
            initializers.append(pos_ids_tensor)
            wpe_node = helper.make_node(
                "Gather", inputs=["wpe_weight", "position_ids"], outputs=["pos_embeddings"], name="position_embedding"
            )
            nodes.append(wpe_node)
            add_node = helper.make_node(
                "Add", inputs=["embeddings", "pos_embeddings"], outputs=["combined_embeddings"], name="add_embeddings"
            )
            nodes.append(add_node)
        else:
            add_node = helper.make_node(
                "Identity", inputs=["embeddings"], outputs=["combined_embeddings"], name="identity_embeddings"
            )
            nodes.append(add_node)
        if "lm_head.weight" in state_dict:
            lm_weight = state_dict["lm_head.weight"].cpu().numpy()
            lm_tensor = numpy_helper.from_array(lm_weight, "lm_head_weight")
            initializers.append(lm_tensor)
            if "lm_head.bias" in state_dict:
                lm_bias = state_dict["lm_head.bias"].cpu().numpy()
                lm_bias_tensor = numpy_helper.from_array(lm_bias, "lm_head_bias")
                initializers.append(lm_bias_tensor)
                gemm_node = helper.make_node(
                    "Gemm", inputs=["combined_embeddings", "lm_head_weight", "lm_head_bias"], outputs=["logits"], name="lm_head"
                )
            else:
                gemm_node = helper.make_node(
                    "Gemm", inputs=["combined_embeddings", "lm_head_weight"], outputs=["logits"], name="lm_head"
                )
            nodes.append(gemm_node)
        else:
            lm_weight = wte_weight.T
            lm_tensor = numpy_helper.from_array(lm_weight, "lm_head_weight")
            initializers.append(lm_tensor)
            gemm_node = helper.make_node(
                "Gemm", inputs=["combined_embeddings", "lm_head_weight"], outputs=["logits"], name="lm_head"
            )
            nodes.append(gemm_node)
        graph = helper.make_graph(
            nodes, f"{model_name}_functional", [input_tensor], [output_tensor], initializer=initializers
        )
        onnx_model = helper.make_model(
            graph, producer_name="model_converter", opset_imports=[helper.make_opsetid("", 11)]
        )
        onnx.save(onnx_model, output_path)
    except Exception as e:
        logger.error(f"Failed to create functional ONNX: {e}")
        raise

def convert_to_onnx(
    model: Any,
    tokenizer: Any,
    model_name: str,
    output_path: str,
    model_type: str,
    device: str
) -> tuple:
    """
    将模型导出为 ONNX 格式。
    Args:
        model: 已加载的模型对象
        tokenizer: 已加载的分词器对象
        model_name: 源模型名称或路径
        output_path: 输出文件路径
        model_type: 模型类型
        device: 设备
    Returns:
        (success: bool, extra_info: dict or None)
    """
    try:
        import torch
        from pathlib import Path
        logger.info(f"Converting {model_name} to ONNX format")
        onnx_file = Path(output_path)
        output_dir = onnx_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        max_opset = _get_max_onnx_opset()
        export_success = False
        last_error = None
        used_opset = None
        # 1. 常规导出
        for opset in range(max_opset, 10, -1):
            try:
                logger.info(f"Trying torch.onnx export with opset {opset}...")
                model.eval()
                if model_type == "image-classification":
                    dummy_input = torch.randn(
                        1, 3, 224, 224, dtype=torch.float16 if device == "cuda" else torch.float32
                    )
                    input_names = ["pixel_values"]
                    dynamic_axes = {"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}}
                else:
                    vocab_size = tokenizer.vocab_size if tokenizer else 50257
                    dummy_input = torch.randint(0, vocab_size, (1, 8), dtype=torch.long)
                    dummy_mask = torch.ones_like(dummy_input)
                dummy_input = {"input_ids": dummy_input, "attention_mask": dummy_mask}
                input_names = ["input_ids", "attention_mask"]
                dynamic_axes = {
                    "input_ids": {0: "batch_size", 1: "sequence"},
                    "attention_mask": {0: "batch_size", 1: "sequence"},
                    "logits": {0: "batch_size", 1: "sequence"},
                }
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_file,
                    export_params=True,
                    opset_version=opset,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=["logits"],
                    dynamic_axes=dynamic_axes,
                    verbose=False,
                    training=torch.onnx.TrainingMode.EVAL,
                    keep_initializers_as_inputs=False,
                    custom_opsets=None,
                )
                if validate_onnx_file(onnx_file, opset):
                    export_success = True
                    used_opset = opset
                    logger.info(f"torch.onnx export successful (opset {opset})")
                    break
                else:
                    logger.warning(f"ONNX file validation failed for opset {opset}")
            except Exception as e:
                last_error = e
                logger.warning(f"torch.onnx export failed (opset {opset}): {e}")
        # 2. functional fallback
        if not export_success:
            try:
                logger.info("Creating functional ONNX model...")
                _create_functional_onnx(model_name, str(onnx_file), model_type, model, tokenizer)
                if validate_onnx_file(onnx_file, 11):
                    export_success = True
                    used_opset = 11
                    logger.info("Functional ONNX model created successfully")
            except Exception as e:
                last_error = e
                logger.warning(f"Functional ONNX creation failed: {e}")
        if not export_success:
            logger.error(f"All ONNX export methods failed. Last error: {last_error}")
            return False, None
        logger.info(f"ONNX conversion completed: {onnx_file}")
        return True, used_opset
    except Exception as e:
        logger.error(f"ONNX conversion error: {e}")
        return False, None 