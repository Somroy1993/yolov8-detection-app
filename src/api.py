"""FastAPI router for object detection endpoints."""
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from src.models import DetectionResponse, Detection
from src.core import load_model, run_detection
import base64
import cv2

router = APIRouter(prefix="/api", tags=["detection"])

@router.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = Query(0.4, ge=0.1, le=0.9),
    class_filter: str = Query("")
):
    """
    Detect objects in image using YOLOv8.
    
    Args:
        file: Image file
        confidence: Confidence threshold
        class_filter: Comma-separated class names to filter
        
    Returns:
        DetectionResponse with detections and annotated image
    """
    try:
        file_bytes = await file.read()
        model = load_model("yolov8n.pt")
        
        classes = [c.strip() for c in class_filter.split(",")] if class_filter else []
        annotated_img, detections_df = run_detection(file_bytes, model, confidence, classes)
        
        # Encode image to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode()
        
        # Create detection list
        detections = [
            Detection(
                class_name=row["class"],
                confidence=row["confidence"],
                x1=row["x1"],
                y1=row["y1"],
                x2=row["x2"],
                y2=row["y2"],
                area=row["area"]
            )
            for _, row in detections_df.iterrows()
        ]
        
        return DetectionResponse(
            detections=detections,
            annotated_image=img_base64,
            total_objects=len(detections)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
