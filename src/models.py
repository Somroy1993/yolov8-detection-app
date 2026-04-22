"""Pydantic models for request/response validation."""
from pydantic import BaseModel
from typing import List, Dict

class DetectionRequest(BaseModel):
    """Request model for object detection."""
    confidence: float = 0.4
    class_filter: List[str] = []

class Detection(BaseModel):
    """Single detection result."""
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    area: int

class DetectionResponse(BaseModel):
    """Response model for detection results."""
    detections: List[Detection]
    annotated_image: str  # base64 encoded
    total_objects: int
