"""Pure business logic for YOLOv8 detection (no Streamlit imports)."""
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import plotly.express as px

def load_model(model_name: str) -> YOLO:
    """Load YOLOv8 model (cached automatically by ultralytics)."""
    return YOLO(model_name)

def run_detection(image_bytes: bytes, model: YOLO, confidence: float = 0.4, class_filter: list = None) -> tuple:
    """
    Run YOLOv8 detection on image.
    
    Args:
        image_bytes: Image file as bytes
        model: YOLO model instance
        confidence: Confidence threshold
        class_filter: List of classes to filter by
        
    Returns:
        Tuple of (annotated_image_rgb, detections_dataframe)
    """
    
    # Load image
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Run detection
    results = model(bgr_image, conf=confidence, verbose=False)
    
    # Extract detections
    detections = []
    annotated_image = bgr_image.copy()
    
    if results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]
            
            # Apply class filter if specified
            if class_filter and class_name not in class_filter:
                continue
            
            area = (x2 - x1) * (y2 - y1)
            
            detections.append({
                "class": class_name,
                "confidence": round(conf, 3),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "area": int(area)
            })
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated_image, f"{class_name} {conf:.2f}", 
                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Create DataFrame
    detections_df = pd.DataFrame(detections) if detections else pd.DataFrame()
    
    return rgb_image, detections_df

def generate_chart(detections_df: pd.DataFrame) -> 'plotly.Figure':
    """Generate plotly chart of object counts by class."""
    if detections_df.empty:
        return None
    
    class_counts = detections_df["class"].value_counts().reset_index()
    class_counts.columns = ["class", "count"]
    
    fig = px.bar(
        class_counts,
        x="count",
        y="class",
        orientation="h",
        title="Object Detection Count by Class",
        labels={"count": "Count", "class": "Class"}
    )
    
    return fig

def detections_to_csv(detections_df: pd.DataFrame) -> str:
    """Convert detections to CSV string."""
    detections_df["timestamp"] = pd.Timestamp.now()
    return detections_df.to_csv(index=False)
