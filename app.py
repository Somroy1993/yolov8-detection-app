import streamlit as st
import cv2
import numpy as np
import pandas as pd
import threading
import time
from datetime import datetime
import re
from io import BytesIO
from src.db import insert_user_email, send_notification_email
from src.core import load_model, run_detection, generate_chart, detections_to_csv

# ============================================================================
# USAGE LIMITS
# ============================================================================
MAX_IMAGES_PER_SESSION = 5
MAX_IMAGE_SIZE_MB = 8
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"]

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(page_title="VisionScan", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    body {
        background-color: #ffffff;
    }
    
    .stSidebar {
        background-color: #16213e !important;
    }
    
    h1, h2, h3 {
        color: #1a1a2e;
    }
    
    .stButton > button {
        background-color: #e94560;
        color: white;
        border: none;
        border-radius: 6px;
    }
    
    .stButton > button:hover {
        background-color: #d63447;
    }
    
    .header-bar {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if "email_verified" not in st.session_state:
    st.session_state.email_verified = False
if "detection_results" not in st.session_state:
    st.session_state.detection_results = {}

# ============================================================================
# EMAIL GATE
# ============================================================================
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

if not st.session_state.email_verified:
    st.markdown('<div class="header-bar"><h1>🎯 VisionScan</h1><p>AI Object Detection & Analysis</p></div>', unsafe_allow_html=True)
    
    with st.form("email_gate"):
        st.write("Enter your email to access this free demo tool.")
        email = st.text_input("Email Address", placeholder="your@email.com")
        submitted = st.form_submit_button("Get Started", use_container_width=True)
        
        if submitted:
            if not email:
                st.error("Please enter your email.")
            elif not validate_email(email):
                st.error("Please enter a valid email address.")
            else:
                try:
                    with st.spinner("🔐 Verifying your email..."):
                        insert_user_email(email, "VisionScan")
                        st.session_state.email_verified = True
                        st.session_state.user_email = email
                        threading.Thread(
                            target=send_notification_email,
                            args=(email, "VisionScan"),
                            daemon=True,
                        ).start()
                    st.success("✅ Welcome to VisionScan! Loading app…")
                    time.sleep(0.6)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    # ============================================================================
    # MAIN APP UI
    # ============================================================================
    
    # Header
    st.markdown('<div class="header-bar"><h1>🎯 VisionScan</h1><p>AI Object Detection & Analysis</p></div>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### About VisionScan")
        st.write("""
        Upload images and detect objects using YOLOv8n AI model.
        """)
        
        st.markdown("### Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
        class_filter = st.multiselect("Filter by Class (optional)", 
            ["person", "car", "dog", "cat", "all"], default=[])
        
        st.markdown("### Usage Limits")
        st.info(f"""
        • Max images: {MAX_IMAGES_PER_SESSION}
        • Max size: {MAX_IMAGE_SIZE_MB} MB each
        """)
        
        st.markdown("### Hire Me")
        st.markdown("[🔗 Upwork Profile](https://www.upwork.com/freelancers/~01c2ba71850d2192bb)")
        
        if st.button("🗑️ Clear Session & Results"):
            st.session_state.clear()
            st.rerun()
    
    # Main content
    st.markdown("### Upload Images")
    st.write("Upload one or more images for object detection.")
    
    uploaded_files = st.file_uploader(
        "Choose image(s)",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        key="image_uploader"
    )
    
    # Process button
    if st.button("🔍 Run Detection", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one image.")
        elif len(uploaded_files) > MAX_IMAGES_PER_SESSION:
            st.error(f"Maximum {MAX_IMAGES_PER_SESSION} images allowed.")
        else:
            # Load model
            with st.spinner("Loading YOLOv8n model..."):
                model = load_model("yolov8n.pt")
            
            st.session_state.detection_results = {}
            all_detections = []
            
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(uploaded_files):
                # Validate file size
                file_size_mb = len(file.getvalue()) / (1024 * 1024)
                if file_size_mb > MAX_IMAGE_SIZE_MB:
                    st.warning(f"⚠️ {file.name} exceeds {MAX_IMAGE_SIZE_MB} MB. Skipping.")
                    continue
                
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        # Run detection
                        annotated_img, detections_df = run_detection(
                            file.getvalue(),
                            model,
                            confidence_threshold,
                            class_filter
                        )
                        
                        st.session_state.detection_results[file.name] = {
                            "annotated_image": annotated_img,
                            "detections": detections_df
                        }
                        
                        all_detections.append(detections_df)
                    
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            if st.session_state.detection_results:
                st.success("✅ Detection complete!")
                
                # Tabs for results
                tab1, tab2, tab3 = st.tabs(["📸 Annotated Images", "📊 Detection Table", "📈 Analysis Chart"])
                
                with tab1:
                    st.subheader("Annotated Images with Bounding Boxes")
                    for filename, result in st.session_state.detection_results.items():
                        st.image(result["annotated_image"], caption=filename, use_container_width=True)
                
                with tab2:
                    st.subheader("Detection Results")
                    if all_detections:
                        combined_df = pd.concat(all_detections, ignore_index=True)
                        st.dataframe(combined_df, use_container_width=True)
                        
                        # CSV Download
                        csv_data = combined_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with tab3:
                    st.subheader("Object Count Analysis")
                    if all_detections:
                        combined_df = pd.concat(all_detections, ignore_index=True)
                        chart = generate_chart(combined_df)
                        st.plotly_chart(chart, use_container_width=True)
            
            st.info("All results are cleared when you close or refresh this page.")
    
    # Footer
    st.markdown("---")
    st.markdown("Built by Somnath Roy · [Upwork Profile](https://www.upwork.com/freelancers/~01c2ba71850d2192bb) · This is a free demo with usage limits.")
