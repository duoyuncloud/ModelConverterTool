"""
tasks.py
Celery task definitions for model conversion and cleanup with improved error handling.
"""

from celery import Celery
from app.converter import ModelConverter
from app.websocket import progress_tracker, progress_callback
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Celery app configuration
celery_app = Celery(
    'model_converter',
    broker=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

# Task configuration
celery_app.conf.update(
    task_time_limit=600,  # 10 minutes
    task_soft_time_limit=540,  # 9 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=100
)

def _validate_output(output_path: str, target_format: str) -> bool:
    """Validate that the output file exists and is valid."""
    try:
        if target_format == "onnx":
            # ONNX always creates model.onnx inside the output directory
            return os.path.isfile(os.path.join(output_path, "model.onnx"))
        # Get the directory containing the output
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            return False
        # Check for files in the output directory based on format
        if target_format == "test":
            # Test format creates a JSON file
            return any(f.endswith('.json') for f in os.listdir(output_dir))
        elif target_format in ["fp16", "hf"]:
            # These formats create directories with config and tokenizer
            return any(f.endswith('_hf') or f.endswith('_fp16') for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f)))
        elif target_format in ["gptq", "awq"]:
            # These formats create directories with config and model files
            return any(f.endswith(f'_{target_format}') for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f)))
        elif target_format == "gguf":
            # GGUF creates directories with .gguf extension
            return any(f.endswith('.gguf') for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f)))
        elif target_format == "mlx":
            # MLX creates directories with .npz extension
            return any(f.endswith('.npz') for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f)))
        else:
            # Other formats create files (torchscript, etc.)
            return any(f.endswith(f'.{target_format}') for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)))
    except Exception as e:
        logger.warning(f"Validation error for {output_path}: {e}")
        return False

def _cleanup_partial_output(output_path: str) -> None:
    """Clean up partial output files on failure."""
    try:
        if os.path.exists(output_path):
            if os.path.isdir(output_path):
                import shutil
                shutil.rmtree(output_path)
            else:
                os.remove(output_path)
        # Also clean up test JSON files
        test_json_path = f"{output_path}.json"
        if os.path.exists(test_json_path):
            os.remove(test_json_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup partial output {output_path}: {e}")

@celery_app.task(bind=True, time_limit=600, soft_time_limit=540)
def convert_model_task(
    self, 
    model_name: str, 
    output_path: str, 
    target_format: str, 
    model_type: str = "causal_lm", 
    conversion_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Celery task for model conversion with improved error handling.
    
    Args:
        model_name: HuggingFace model name or path
        output_path: Output file path
        target_format: Target format
        model_type: Model type
        conversion_params: Conversion parameters
        
    Returns:
        Output path on success
        
    Raises:
        RuntimeError: If conversion fails
    """
    task_id = self.request.id
    conversion_params = conversion_params or {}
    
    try:
        # Initialize task tracking
        progress_tracker.set_task_info(
            task_id=task_id,
            model_name=model_name,
            target_format=target_format,
            model_type=model_type,
            conversion_params=conversion_params
        )
        
        # Update task status
        progress_tracker.set_task_status(task_id, "started")
        progress_tracker.update_progress(task_id, 0, "Starting conversion task")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create progress callback
        callback = progress_callback(task_id)
        
        logger.info(f"Starting conversion task {task_id}: {model_name} -> {target_format}")
        
        # Debug print to confirm latest code is running
        print("DEBUG: I am using output_dir param for convert()")
        # Execute conversion using ModelConverter
        converter = ModelConverter()
        result = converter.convert(
            model_name=model_name,
            target_format=target_format,
            output_dir=output_path,
            options=conversion_params
        )
        
        # Validate output
        if result.get("status") != "success" or not _validate_output(output_path, target_format):
            raise RuntimeError(f"Model conversion failed, output file not found or invalid: {result.get('error')}")
        
        # Update task status to success
        progress_tracker.set_task_status(task_id, "success", output_path)
        progress_tracker.update_progress(task_id, 100, "Conversion completed", {"output_path": output_path})
        
        logger.info(f"Conversion task {task_id} completed: {output_path}")
        return output_path
        
    except Exception as e:
        error_msg = f"Model conversion failed: {str(e)}"
        logger.error(f"Conversion task {task_id} failed: {error_msg}")
        # Store error log for download
        progress_tracker.set_task_error_log(task_id, error_msg)
        # Update task status to failed
        progress_tracker.set_task_status(task_id, "failed")
        progress_tracker.update_progress(task_id, 0, error_msg, {"error": str(e)})
        # Clean up partial output
        _cleanup_partial_output(output_path)
        raise RuntimeError(error_msg)

@celery_app.task
def cleanup_task(task_id: str) -> Dict[str, str]:
    """Clean up task data from cache."""
    try:
        progress_tracker.cleanup_task(task_id)
        logger.info(f"Task data cleanup completed: {task_id}")
        return {"status": "success", "task_id": task_id}
    except Exception as e:
        logger.error(f"Task data cleanup failed: {task_id}, error: {e}")
        return {"status": "failed", "task_id": task_id, "error": str(e)}

@celery_app.task
def health_check() -> Dict[str, Any]:
    """Health check task."""
    return {
        "status": "healthy", 
        "timestamp": "2024-01-01T00:00:00Z",
        "celery_worker": True
    }

@celery_app.task
def clear_model_cache() -> Dict[str, str]:
    """Clear the model cache to free memory."""
    try:
        ModelConverter().clear_cache()
        return {"status": "success", "message": "Model cache cleared"}
    except Exception as e:
        logger.error(f"Failed to clear model cache: {e}")
        return {"status": "failed", "error": str(e)} 