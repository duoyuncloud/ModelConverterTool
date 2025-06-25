"""
schemas.py
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ConvertRequest(BaseModel):
    """Model for a conversion request."""
    model_name: str
    target_format: str
    model_type: str = "causal_lm"
    conversion_params: Optional[Dict[str, Any]] = None

class ConvertResponse(BaseModel):
    """Response for a conversion request."""
    task_id: str
    message: str

class StatusResponse(BaseModel):
    """Response for task status."""
    task_id: str
    status: str
    progress: int
    message: str
    result: Optional[str] = None

class BatchConvertRequest(BaseModel):
    """Request for batch conversion."""
    tasks: List[ConvertRequest]

class BatchConvertResponse(BaseModel):
    """Response for batch conversion."""
    batch_id: str
    task_ids: List[str]

class SystemInfo(BaseModel):
    """System information."""
    version: str
    supported_formats: List[str]
    supported_model_types: List[str]
    system_status: str 