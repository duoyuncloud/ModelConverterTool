"""
api.py
FastAPI routes for model conversion, validation, cache, and monitoring.
"""

from fastapi import APIRouter, Form, HTTPException, BackgroundTasks, WebSocket, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional
import uuid
import os
from app.tasks import convert_model_task, celery_app
from app.websocket import websocket_endpoint, progress_tracker
from app.validator import validator
from app.cache import cache_manager
from app.config import settings
from app.schemas import ConvertResponse, StatusResponse
from app.utils import parse_json_params, generate_output_path
from functools import wraps
import time
import psutil
import platform

UPLOAD_DIR = settings.upload_dir
OUTPUT_DIR = settings.output_dir
SUPPORTED_FORMATS = settings.supported_formats
SUPPORTED_MODEL_TYPES = settings.supported_model_types

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

router = APIRouter()

SERVER_START_TIME = time.time()

@router.get("/formats", summary="Get supported model formats")
def get_formats():
    """Return supported conversion formats."""
    return {"formats": SUPPORTED_FORMATS}

@router.get("/model-types", summary="Get supported model types")
def get_model_types():
    """Return supported model types."""
    return {"model_types": SUPPORTED_MODEL_TYPES}

@router.get("/health", summary="Health check")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.version}

@router.get("/stats", summary="Get system statistics")
def get_stats():
    """Return system statistics."""
    cache_stats = cache_manager.get_cache_stats()
    return {
        "supported_formats": len(SUPPORTED_FORMATS),
        "supported_model_types": len(SUPPORTED_MODEL_TYPES),
        "cache_stats": cache_stats
    }

def handle_exceptions(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return wrapper

@router.post("/validate/model", summary="Validate model info")
@handle_exceptions
def validate_model(model_name: str = Form(...)):
    """Validate HuggingFace model info."""
    return validator.validate_huggingface_model(model_name)

@router.post("/validate/params", summary="Validate conversion parameters")
@handle_exceptions
def validate_conversion_params(
    target_format: str = Form(...),
    conversion_params: str = Form("{}")
):
    """Validate conversion parameters."""
    params = parse_json_params(conversion_params)
    return validator.validate_conversion_params(target_format, params)

@router.post("/upload-model", summary="Upload a local model file")
@handle_exceptions
async def upload_model(file: UploadFile = File(...)):
    upload_dir = settings.upload_dir
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"file_path": file_path}

@router.post("/convert", summary="Submit conversion task")
@handle_exceptions
async def convert_model(
    model_name: str = Form(None),
    file_path: str = Form(None),
    target_format: str = Form(...),
    model_type: str = Form("causal_lm"),
    conversion_params: str = Form("{}")
):
    """Submit a model conversion task. Accepts either a HuggingFace model name or a local file path."""
    print("RECEIVED:", model_name, file_path, target_format, model_type, conversion_params)
    if not model_name and not file_path:
        raise HTTPException(status_code=400, detail="Either model_name or file_path must be provided.")
    model_input = file_path if file_path else model_name
    if target_format.lower() not in settings.supported_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported target format: {target_format}")
    if model_type not in settings.supported_model_types:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
    kwargs = parse_json_params(conversion_params, error_msg="Invalid conversion parameter format")
    file_id = str(uuid.uuid4())
    output_path = generate_output_path(file_id, target_format, settings.output_dir)
    task = convert_model_task.apply_async(args=[model_input, output_path, target_format, model_type, kwargs])
    progress_tracker.set_task_info(task.id, model_input, target_format, model_type, kwargs)
    return ConvertResponse(
        task_id=task.id,
        message="Conversion task submitted"
    )

@router.get("/status/{task_id}", summary="Get conversion task status")
def get_status(task_id: str):
    """Get the status of a conversion task."""
    task_result = celery_app.AsyncResult(task_id)
    progress_info = progress_tracker.get_progress(task_id)
    return StatusResponse(
        task_id=task_id,
        status=task_result.status,
        progress=progress_info.get("progress", 0) if progress_info else 0,
        message=progress_info.get("message", task_result.status) if progress_info else task_result.status,
        result=task_result.result if task_result.successful() else None
    )

@router.get("/download/{task_id}", summary="Download converted model")
def download_model(task_id: str):
    """Download the converted model file."""
    try:
        task_result = celery_app.AsyncResult(task_id)
        if not task_result.successful():
            raise HTTPException(status_code=400, detail="Task not finished or failed")
        output_path = task_result.result
        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Output file not found")
        filename = os.path.basename(output_path)
        return FileResponse(output_path, filename=filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/task/{task_id}", summary="Delete task")
def delete_task(task_id: str, background_tasks: BackgroundTasks):
    """Delete a conversion task."""
    try:
        celery_app.control.revoke(task_id, terminate=True)
        background_tasks.add_task(progress_tracker.cleanup_task, task_id)
        return {"message": "Task deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{task_id}")
async def websocket_connection(websocket: WebSocket, task_id: str):
    """WebSocket connection for task progress updates."""
    await websocket_endpoint(websocket, task_id)

@router.get("/cache/stats", summary="Get cache statistics")
def get_cache_stats():
    """Get cache statistics."""
    try:
        return cache_manager.get_cache_stats()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache/clear", summary="Clear cache")
def clear_cache(prefix: str = None):
    """Clear cache."""
    try:
        cache_manager.clear_cache(prefix)
        return {"message": "Cache cleared"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/system", summary="Get system monitoring info")
def get_system_monitoring():
    """Get system monitoring info."""
    return {"status": "ok"}

@router.get("/system/info", summary="Get system info")
def get_system_info():
    """Get system info."""
    return {
        "version": settings.version,
        "supported_formats": settings.supported_formats,
        "supported_model_types": settings.supported_model_types,
        "system_status": "running",
        "mlx_available": getattr(settings, "mlx_available", False)
    }

@router.get("/capabilities", summary="Get system conversion capabilities")
def get_capabilities():
    """Return system capabilities for quantization formats."""
    import sys
    capabilities = {}
    # Check CUDA
    try:
        import torch
        capabilities["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            capabilities["gpu_name"] = torch.cuda.get_device_name(0)
            capabilities["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        else:
            capabilities["gpu_name"] = None
            capabilities["vram_gb"] = None
    except ImportError:
        capabilities["cuda"] = False
        capabilities["gpu_name"] = None
        capabilities["vram_gb"] = None
    # Check auto-gptq
    try:
        import auto_gptq
        capabilities["gptq"] = capabilities["cuda"]
    except ImportError:
        capabilities["gptq"] = False
    # Check awq
    try:
        import awq
        capabilities["awq"] = capabilities["cuda"]
    except ImportError:
        capabilities["awq"] = False
    # Add Python and OS info
    capabilities["python_version"] = sys.version.split()[0]
    capabilities["os"] = platform.system() + " " + platform.release()
    # RAM info
    capabilities["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    # CPU info
    capabilities["cpu"] = platform.processor() or platform.machine()
    capabilities["cpu_cores"] = psutil.cpu_count(logical=True)
    # Disk space
    disk = psutil.disk_usage("/")
    capabilities["disk_total_gb"] = round(disk.total / 1024**3, 2)
    capabilities["disk_free_gb"] = round(disk.free / 1024**3, 2)
    # Uptime
    capabilities["uptime_minutes"] = int((time.time() - SERVER_START_TIME) / 60)
    return capabilities

@router.get("/error_log/{task_id}", summary="Download error log for a task")
def download_error_log(task_id: str):
    """Download the error log for a failed conversion task."""
    error_log = progress_tracker.get_task_error_log(task_id)
    if not error_log:
        raise HTTPException(status_code=404, detail="No error log found for this task.")
    from io import BytesIO
    return StreamingResponse(BytesIO(error_log.encode()), media_type="text/plain", headers={"Content-Disposition": f"attachment; filename=error_log_{task_id}.txt"})

@router.get("/progress_timeline/{task_id}", summary="Get progress timeline for a task")
def get_progress_timeline(task_id: str):
    timeline = progress_tracker.get_progress_timeline(task_id)
    if not timeline:
        raise HTTPException(status_code=404, detail="No progress timeline found for this task.")
    return timeline

@router.get("/task_history/{task_id}", summary="Download full task history as JSON")
def download_task_history(task_id: str):
    import json as _json
    history = progress_tracker.get_full_task_history(task_id)
    if not history["task_info"]:
        raise HTTPException(status_code=404, detail="No task history found for this task.")
    from io import BytesIO
    return StreamingResponse(BytesIO(_json.dumps(history, indent=2).encode()), media_type="application/json", headers={"Content-Disposition": f"attachment; filename=task_history_{task_id}.json"})