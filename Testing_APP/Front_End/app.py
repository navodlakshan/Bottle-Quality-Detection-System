# Bottle-Quality-Detection-System\Testing_APP\Front_End\app.py

import streamlit as st
from PIL import Image
import requests
import base64
import io
import pandas as pd
import cv2
import numpy as np
import threading
from threading import Lock
import time
import queue
import plotly.graph_objects as go
from datetime import datetime
import json
import firebase
from firebase import firebase

# --- FIREBASE CONFIGURATION ---
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyA28GpqF0DWO5rFprJzMbaZa8RYnZQPi3o",
    "authDomain": "milk-bottle-quality-inspector.firebaseapp.com",
    "databaseURL": "https://milk-bottle-quality-inspector-default-rtdb.firebaseio.com",
    "projectId": "milk-bottle-quality-inspector",
    "storageBucket": "milk-bottle-quality-inspector.firebasestorage.app",
    "messagingSenderId": "817815155804",
    "appId": "1:817815155804:web:fbf9d08decac530a512299",
    "measurementId": "G-WY2H2WBLEC"
}

# Database secret (from your message)
DATABASE_SECRET = "EnJfsULs7oiZyaHrUpet6sgQIbvfsIanF3geoWUq"

# Initialize Firebase (using firebase module)
firebase_app = firebase.FirebaseApplication(FIREBASE_CONFIG["databaseURL"], None)

# --- CONFIGURATION ---
FASTAPI_URL = "http://127.0.0.1:8000/predict/"
confidence_threshold = 0.60  # Default confidence threshold for defect detection

# Define the set of classes that trigger a rejection
REJECT_CLASSES = {
    'BodyCrack',
    'LuckyLogoUnclear',
    'OtherLogo',
    'MouthCrack',
    'UnclearLabel'
}

# Transparency threshold (values should be greater than 1.0 for good quality)
TRANSPARENCY_THRESHOLD = 1.0

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #673AB7 0%, #B39DDB 40%, #D1C4E9 70%, #EDE7F6 100%);
        color:#4527A0; 
    }
    
    /* Main styling */
    .main-header {
        font-size: 4rem;
        color: #512DA8;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(81, 45, 168, 0.2);
        background: linear-gradient(135deg, #512DA8 0%, #7E57C2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.6rem;
        color: #512DA8;
        border-bottom: 3px solid #7E57C2;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1.2rem;
        font-weight: 600;
        background: rgba(81, 45, 168, 0.05);
        padding: 1rem;
        border-radius: 8px;
    }
    
    .verdict-accept {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: black;
        padding: 1.5rem;
        border-radius: 12px;
        border: none;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        font-size: 1.2rem;
    }
    
    .verdict-reject {
        background: linear-gradient(135deg, #F44336 0%, #EF5350 100%);
        color: black;
        padding: 1.5rem;
        border-radius: 12px;
        border: none;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        font-size: 1.2rem;
    }
    
    .verdict-warning {
        background: linear-gradient(135deg, #FF9800 0%, #FFB74D 100%);
        color: black;
        padding: 1.5rem;
        border-radius: 12px;
        border: none;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
        font-size: 1.2rem;
    }
    
    .defect-yes {
        background: linear-gradient(135deg, #F44336 0%, #EF5350 100%);
        color: black;
        font-weight: bold;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        border: none;
        font-size: 0.85rem;
        box-shadow: 0 2px 8px rgba(244, 67, 54, 0.2);
    }
    
    .defect-no {
        background: linear-gradient(135deg, #66BB6A 0%, #81C784 100%);
        color: black;
        font-weight: bold;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        border: none;
        font-size: 0.85rem;
        box-shadow: 0 2px 8px rgba(102, 187, 106, 0.2);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #512DA8 0%, #7E57C2 100%);
        color: black;
        border: none;
        padding: 0.8rem 1.8rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(81, 45, 168, 0.3);
        font-size: 1rem;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #4527A0 0%, #673AB7 100%);
        color: white;
        box-shadow: 0 6px 20px rgba(81, 45, 168, 0.4);
        transform: translateY(-2px);
    }
    
    .stButton button:disabled {
        background: linear-gradient(135deg, #BDBDBD 0%, #9E9E9E 100%);
        box-shadow: none;
        transform: none;
    }
    
    .image-preview {
        border-radius: 12px;
        padding: 15px;
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        box-shadow: 0 4px 15px rgba(81, 45, 168, 0.1);
        border: 2px solid #f0ebff;
        transition: transform 0.3s ease;
    }
    
    .image-preview:hover {
        transform: scale(1.02);
    }
    
    .info-box {
        background: linear-gradient(135deg, #EDE7F6 0%, #D1C4E9 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #B39DDB;
        color: #512DA8;
        box-shadow: 0 4px 15px rgba(81, 45, 168, 0.1);
    }
    
    /* Radio button and slider styling */
    .stRadio > div {
        background: linear-gradient(135deg, #512DA8 0%, #7E57C2 100%);
        color: white !important;
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #e8e2ff;
        box-shadow: 0 4px 15px rgba(81, 45, 168, 0.1);
    }
    
    .stSlider {
        background: linear-gradient(135deg, #512DA8 0%, #7E57C2 100%);
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #e8e2ff;
        box-shadow: 0 4px 15px rgba(81, 45, 168, 0.1);
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #FFFFFF 0%, #EDE7F6 100%) !important;
        color: #512DA8 !important;
    }
    
    .stSlider > div > div > div {
        background: #D1C4E9 !important;
        color: #512DA8 !important;
    }
    
    .stSlider > div > div > div > div {
        border: 2px solid #FFFFFF !important;
        color: #512DA8 !important;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        color: #4CAF50;
        border-radius: 12px;
        padding: 10px;
        border: 2px dashed #B39DDB;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #7E57C2;
        color: #4CAF50;
        background: linear-gradient(135deg, #ffffff 0%, #f0ebff 100%);
    }
    
    /* Camera input styling */
    .stCameraInput {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(81, 45, 168, 0.1);
        border: 2px solid #e8e2ff;
    }
    
    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8e2ff 100%);
        padding: 8px;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(81, 45, 168, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 65px;
        white-space: pre-wrap;
        background: transparent;
        border-radius: 16px;
        gap: 8px;
        padding: 20px 24px;
        font-size: 1.5rem;
        font-weight: 700;
        color: #6B7280;
        border: 2px solid #D1C4E9;
        margin: 0 2px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.8);
        color: #512DA8;
        border-color: #D1C4E9;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(81, 45, 168, 0.15);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #512DA8 0%, #7E57C2 100%) !important;
        color: white !important;
        border-color: #512DA8;
        box-shadow: 0 8px 30px rgba(81, 45, 168, 0.3);
        transform: translateY(-2px);
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        border: 2px solid #e8e2ff;
        border-radius: 12px;
        padding: 20px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        box-shadow: 0 4px 15px rgba(81, 45, 168, 0.1);
    }
    
    /* Smaller image previews */
    .small-preview {
        max-height: 180px;
        object-fit: cover;
        border-radius: 10px;
        border: 2px solid #e8e2ff;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        color: #4CAF50;
    }
    
    .status-good {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: #4CAF50;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #FF9800 0%, #FFB74D 100%);
        color: #4CAF50;
    }
    
    .status-bad {
        background: linear-gradient(135deg, #F44336 0%, #EF5350 100%);
        color: #4CAF50;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, #512DA8 0%, #7E57C2 100%);
    }
    
    /* Success and error messages */
    .stSuccess {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: black;
        border-radius: 12px;
        padding: 1rem;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, #F44336 0%, #EF5350 100%);
        color: black;
        border-radius: 12px;
        padding: 1rem;
        border: none;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #2196F3 0%, #42A5F5 100%);
        color: black;
        border-radius: 12px;
        padding: 1rem;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #FF9800 0%, #FFB74D 100%);
        color: black;
        border-radius: 12px;
        padding: 1rem;
        border: none;
    }
    
    /* Toggle styling */
    .stCheckbox {
        padding: 10px;
        border-radius: 12px;
        border: 2px solid #e8e2ff;
    }
    
    /* Settings Panel */
    .settings-panel {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px solid #90CAF9;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.2);
        margin-bottom: 1.5rem;
    }
    
    /* Defect types display */
    .defect-types {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 2px solid #A5D6A7;
    }
    
    /* NEW: Threshold value styling */
    .threshold-value {
        background: linear-gradient(135deg, #FFEB3B 0%, #FFEE58 100%);
        color: #FF6F00;
        font-weight: bold;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #FFA000;
        text-align: center;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(255, 152, 0, 0.2);
    }
    
    /* NEW: Correct threshold styling */
    .correct-threshold {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: #1B5E20;
        font-weight: bold;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #2E7D32;
        text-align: center;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.2);
    }
    
    /* NEW: Defect types list styling */
    .defect-list {
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 2px solid #F48FB1;
        color: #880E4F;
        font-weight: 500;
    }
    
    /* NEW: Settings column styling */
    .settings-column {
        # background: linear-gradient(135deg, #512DA8 0%, #7E57C2 100%);
        color: white;
        padding: 1.5rem 0rem 0rem 1.5rem;
        border-radius: 12px;
        border: 2px solid #B39DDB;
        box-shadow: 0 4px 15px rgba(81, 45, 168, 0.3);
        height: 100%;
    }
    
    /* NEW: Radio button labels in settings */
    .settings-radio label {
        color: white !important;
        font-weight: 500;
    }
    
    /* NEW: Slider label styling */
    .settings-slider label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Transparency specific styles */
    .transparency-good {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: black;
        padding: 0.5rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.2);
    }
    
    .transparency-bad {
        background: linear-gradient(135deg, #F44336 0%, #EF5350 100%);
        color: black;
        padding: 0.5rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 8px rgba(244, 67, 54, 0.2);
    }
    
    .transparency-warning {
        background: linear-gradient(135deg, #FF9800 0%, #FFB74D 100%);
        color: black;
        padding: 0.5rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 8px rgba(255, 152, 0, 0.2);
    }
    
    .system-diagram {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #90CAF9;
    }
</style>
""", unsafe_allow_html=True)

# --- UI SETUP ---
st.set_page_config(
    page_title="Milk Bottle Quality Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for transparency data
if 'transparency_data' not in st.session_state:
    st.session_state.transparency_data = None
if 'transparency_history' not in st.session_state:
    st.session_state.transparency_history = []

# --- HEADER ---
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1>
       🥛 <span  class="main-header"> Milk Bottle Quality Inspector </span>
    </h1>
    <div style="color: #7E57C2; margin-bottom: 2rem; font-size: 1.2rem; font-weight: 500;">
        Advanced AI-powered defect detection & transparency analysis for premium quality assurance
    </div>
</div>
""", unsafe_allow_html=True)

# Use enhanced tabs for better organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 **UPLOAD IMAGES**", "📸 **LIVE INSPECTION**", "🔮 **TRANSPARENCY DETECTION**", "🔍 **ANALYZE IMAGES**", "📊 **ANALYSIS FINAL RESULTS**"])

with tab1:
    # =====================================================
    # 1️⃣ UPLOAD SECTION
    # =====================================================

    st.markdown('<h2 class="sub-header">📁 Upload Bottle Images</h2>', unsafe_allow_html=True)
    st.markdown("Upload **four images** of the bottle: mouth and 3 body angles for comprehensive analysis.")

    up_col1, up_col2, up_col3, up_col4 = st.columns(4)
    
    uploaded_files = {}
    
    with up_col1:
        st.markdown("**<span class='status-warning status-indicator'></span> Bottle Mouth**", unsafe_allow_html=True)
        img_mouth = st.file_uploader("Mouth view", type=["jpg", "jpeg", "png"], key="mouth", 
                                    help="Capture the bottle mouth area")
        if img_mouth:
            uploaded_files["Mouth"] = img_mouth
            st.image(img_mouth, use_container_width=True, caption="Mouth Preview", output_format="JPEG")
    
    with up_col2:
        st.markdown("**<span class='status-good status-indicator'></span> Body Angle 1**", unsafe_allow_html=True)
        img_angle1 = st.file_uploader("Body view 1", type=["jpg", "jpeg", "png"], key="angle1",
                                     help="First body angle")
        if img_angle1:
            uploaded_files["Angle 1"] = img_angle1
            st.image(img_angle1, use_container_width=True, caption="Angle 1 Preview", output_format="JPEG")
    
    with up_col3:
        st.markdown("**<span class='status-good status-indicator'></span> Body Angle 2**", unsafe_allow_html=True)
        img_angle2 = st.file_uploader("Body view 2", type=["jpg", "jpeg", "png"], key="angle2",
                                     help="Second body angle")
        if img_angle2:
            uploaded_files["Angle 2"] = img_angle2
            st.image(img_angle2, use_container_width=True, caption="Angle 2 Preview", output_format="JPEG")
    
    with up_col4:
        st.markdown("**<span class='status-good status-indicator'></span> Body Angle 3**", unsafe_allow_html=True)
        img_angle3 = st.file_uploader("Body view 3", type=["jpg", "jpeg", "png"], key="angle3",
                                     help="Third body angle")
        if img_angle3:
            uploaded_files["Angle 3"] = img_angle3
            st.image(img_angle3, use_container_width=True, caption="Angle 3 Preview", output_format="JPEG")
    
    # Show upload status with better styling
    if uploaded_files:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%); color: white; padding: 1rem; border-radius: 12px; text-align: center;">
            <span style="font-size: 1.2rem; font-weight: bold;">✅ {len(uploaded_files)} image(s) uploaded successfully</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2196F3 0%, #42A5F5 100%); color: white; padding: 1rem; border-radius: 12px; text-align: center;">
            <span style="font-size: 1.1rem;">👆 Upload images using the controls above</span>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    # =====================================================
    # 2️⃣ REAL-TIME VIDEO STREAMING WITH DETECTION
    # =====================================================

    st.markdown('<h2 class="sub-header">📹 Live Bottle Detection</h2>', unsafe_allow_html=True)

    # Initialize session states for streaming
    if "streaming" not in st.session_state:
        st.session_state.streaming = False
    if "stream_stats" not in st.session_state:
        st.session_state.stream_stats = {
            "frames": 0,
            "fps": 0,
            "detections": 0,
        }

    # Start/Stop buttons - Simple and clean
    col_start, col_stop = st.columns(2)
    
    with col_start:
        if st.button("▶️ START CAMERA", use_container_width=True, key="start_stream"):
            st.session_state.streaming = True
    with col_stop:
        if st.button("⏹️ STOP CAMERA", use_container_width=True, key="stop_stream"):
            st.session_state.streaming = False
    
    if st.session_state.streaming:
        st.markdown("**🔴 LIVE - Camera Active**")
        
        # Simple layout
        col_video, col_info = st.columns([3, 1])
        
        with col_video:
            video_display = st.empty()
        
        with col_info:
            stats_display = st.empty()
            classes_display = st.empty()
        
        # Streaming loop
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            frame_counter = 0
            start_time = time.time()
            
            while st.session_state.streaming:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Camera not accessible")
                    break
                
                frame_counter += 1
                st.session_state.stream_stats["frames"] = frame_counter
                
                # Process every frame for real-time detection
                try:
                    # Convert to PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    
                    # Compress frame
                    img_bytes = io.BytesIO()
                    pil_frame.save(img_bytes, format="JPEG", quality=75)
                    img_bytes.seek(0)
                    
                    # Send to API
                    files_to_send = [('files', ("frame.jpg", img_bytes.getvalue(), "image/jpeg"))]
                    response = requests.post(FASTAPI_URL, files=files_to_send, timeout=2)
                    
                    if response.status_code == 200:
                        results = response.json()[0]
                        
                        # Display detected frame
                        encoded = results['image_plotted']
                        decoded = base64.b64decode(encoded)
                        video_display.image(decoded, use_container_width=True)
                        
                        # Get detections
                        detections = results['detections']
                        st.session_state.stream_stats["detections"] = len(detections)
                        
                        # Calculate FPS
                        elapsed = time.time() - start_time
                        st.session_state.stream_stats["fps"] = frame_counter / elapsed if elapsed > 0 else 0
                        
                        # Update stats
                        with stats_display.container():
                            st.metric("🎬 FPS", f"{st.session_state.stream_stats['fps']:.1f}")
                            st.metric("📊 Frames", st.session_state.stream_stats["frames"])
                            st.metric("🔍 Objects", st.session_state.stream_stats["detections"])
                        
                        # Display classes
                        with classes_display.container():
                            st.markdown("**Classes Detected:**")
                            if detections:
                                for det in detections[:3]:
                                    conf = det['confidence']
                                    name = det['class_name']
                                    
                                    if conf > 0.8:
                                        st.success(f"✅ {name}: {conf:.0%}")
                                    elif conf > 0.6:
                                        st.warning(f"⚠️ {name}: {conf:.0%}")
                                    else:
                                        st.info(f"ℹ️ {name}: {conf:.0%}")
                            else:
                                st.info("Scanning...")
                
                except requests.exceptions.Timeout:
                    pass
                except Exception as e:
                    pass
                
                time.sleep(0.01)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            try:
                cap.release()
            except:
                pass
            st.session_state.streaming = False
    
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #512DA8 0%, #7E57C2 100%); color: white; padding: 2rem; border-radius: 12px; text-align: center;">
            <h3 style="margin: 0; font-size: 1.3rem;">📹 Start camera to detect bottles</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem;">Real-time detection starts automatically</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    # =====================================================
    # 3️⃣ TRANSPARENCY DETECTION SECTION
    # =====================================================
    
    st.markdown('<h2 class="sub-header">🔮 Bottle Transparency Detection System</h2>', unsafe_allow_html=True)
    
    # Main control section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Current Bottle Transparency")
        
        # Manual input or Firebase fetch
        option = st.radio("Data Source:", ["📡 Fetch from Firebase", "📝 Manual Input"], horizontal=True)
        
        if option == "📡 Fetch from Firebase":
            if st.button("🔄 Fetch Latest Transparency Data", use_container_width=True):
                with st.spinner("Fetching data from Firebase..."):
                    try:
                        # Fetch data from Firebase Realtime Database
                        # Get the latest reading
                        result = firebase_app.get('/bottleReadings', None)
                        
                        if result:
                            # Find the latest entry by timestamp
                            latest_entry = None
                            latest_timestamp = 0
                            
                            # Handle different data structure formats
                            if isinstance(result, dict):
                                # Check if result is a direct entry with top, mid, bottom
                                if 'top' in result and 'mid' in result and 'bottom' in result:
                                    # Direct single entry format
                                    latest_entry = result
                                    latest_timestamp = result.get('timestamp', int(time.time() * 1000))
                                else:
                                    # Multiple entries format - find latest by timestamp
                                    for key, value in result.items():
                                        if isinstance(value, dict):
                                            # Check for various timestamp field names
                                            entry_timestamp = value.get('timestamp') or value.get('tim') or value.get('time') or 0
                                            
                                            # Verify it has the required fields
                                            if 'top' in value and 'mid' in value and 'bottom' in value:
                                                if entry_timestamp > latest_timestamp:
                                                    latest_timestamp = entry_timestamp
                                                    latest_entry = value
                            
                            if latest_entry and 'top' in latest_entry and 'mid' in latest_entry and 'bottom' in latest_entry:
                                data = {
                                    "top": float(latest_entry.get('top', 0)),
                                    "mid": float(latest_entry.get('mid', 0)),
                                    "bottom": float(latest_entry.get('bottom', 0)),
                                    "timestamp": latest_timestamp if latest_timestamp > 0 else int(time.time() * 1000)
                                }
                                
                                st.session_state.transparency_data = data
                                st.session_state.transparency_history.append(data.copy())
                                
                                st.success("✅ Successfully fetched transparency data from IoT system!")
                                st.info(f"Data: Top={data['top']:.3f}, Mid={data['mid']:.3f}, Bottom={data['bottom']:.3f}")
                            else:
                                st.warning("⚠️ No valid data found in Firebase")
                                st.error("Could not find required fields (top, mid, bottom) in Firebase data")
                                # Show raw data for debugging
                                with st.expander("📋 Debug: Raw Firebase Data"):
                                    st.json(result)
                        else:
                            st.warning("⚠️ No data available in Firebase database")
                            
                    except Exception as e:
                        st.error(f"❌ Error fetching data from Firebase: {str(e)}")
                        st.info("Using sample data for demonstration")
                        
                        # Fallback to sample data
                        data = {
                            "top": 0.95967,
                            "mid": 0.61367,
                            "bottom": 0.79567,
                            "timestamp": int(time.time() * 1000)
                        }
                        st.session_state.transparency_data = data
        
        else:  # Manual Input
            with st.form("manual_transparency"):
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    top_val = st.number_input("Top Transparency", min_value=0.0, max_value=3.0, value=1.2, step=0.01)
                with col_t2:
                    mid_val = st.number_input("Mid Transparency", min_value=0.0, max_value=3.0, value=1.1, step=0.01)
                with col_t3:
                    bottom_val = st.number_input("Bottom Transparency", min_value=0.0, max_value=3.0, value=1.3, step=0.01)
                
                if st.form_submit_button("🔍 Analyze Transparency", use_container_width=True):
                    data = {
                        "top": top_val,
                        "mid": mid_val,
                        "bottom": bottom_val,
                        "timestamp": int(time.time() * 1000)
                    }
                    st.session_state.transparency_data = data
                    st.session_state.transparency_history.append(data.copy())
                    st.success("✅ Analysis complete!")
    
    with col2:
        st.markdown("### ⚙️ Settings")
        st.markdown(f"""
        <div class="info-box">
        <h4>📏 Quality Threshold</h4>
        <p><strong>Minimum Value:</strong> {TRANSPARENCY_THRESHOLD}</p>
        <p>Values > {TRANSPARENCY_THRESHOLD} indicate good transparency</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add option to view raw Firebase data
        if st.button("📊 View Raw Firebase Data", use_container_width=True):
            try:
                result = firebase_app.get('/bottleReadings', None)
                if result:
                    st.markdown("### Raw Firebase Data")
                    st.json(result)
                else:
                    st.info("No data in Firebase database")
            except Exception as e:
                st.error(f"Error accessing Firebase: {str(e)}")
        
        if st.button("🔄 Clear History", use_container_width=True):
            st.session_state.transparency_history = []
            st.success("History cleared!")
    
    # Display results if data exists
    if st.session_state.transparency_data:
        data = st.session_state.transparency_data
        
        st.markdown("---")
        st.markdown("### 📈 Transparency Analysis Results")
        
        # Create metrics columns
        col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
        
        with col_metrics1:
            top_status = "✅ GOOD" if data["top"] > TRANSPARENCY_THRESHOLD else "❌ FAIL"
            top_color = "transparency-good" if data["top"] > TRANSPARENCY_THRESHOLD else "transparency-bad"
            st.markdown(f'<div class="{top_color}"><h4>TOP</h4><h2>{data["top"]:.3f}</h2><p>{top_status}</p></div>', unsafe_allow_html=True)
        
        with col_metrics2:
            mid_status = "✅ GOOD" if data["mid"] > TRANSPARENCY_THRESHOLD else "❌ FAIL"
            mid_color = "transparency-good" if data["mid"] > TRANSPARENCY_THRESHOLD else "transparency-bad"
            st.markdown(f'<div class="{mid_color}"><h4>MID</h4><h2>{data["mid"]:.3f}</h2><p>{mid_status}</p></div>', unsafe_allow_html=True)
        
        with col_metrics3:
            bottom_status = "✅ GOOD" if data["bottom"] > TRANSPARENCY_THRESHOLD else "❌ FAIL"
            bottom_color = "transparency-good" if data["bottom"] > TRANSPARENCY_THRESHOLD else "transparency-bad"
            st.markdown(f'<div class="{bottom_color}"><h4>BOTTOM</h4><h2>{data["bottom"]:.3f}</h2><p>{bottom_status}</p></div>', unsafe_allow_html=True)
        
        with col_metrics4:
            # Determine overall verdict
            all_good = all([
                data["top"] > TRANSPARENCY_THRESHOLD,
                data["mid"] > TRANSPARENCY_THRESHOLD,
                data["bottom"] > TRANSPARENCY_THRESHOLD
            ])
            
            if all_good:
                verdict = "✅ PASS"
                verdict_class = "verdict-accept"
                verdict_text = "All values meet quality standards"
            else:
                fail_count = sum([
                    data["top"] <= TRANSPARENCY_THRESHOLD,
                    data["mid"] <= TRANSPARENCY_THRESHOLD,
                    data["bottom"] <= TRANSPARENCY_THRESHOLD
                ])
                verdict = f"❌ FAIL ({fail_count} sections)"
                verdict_class = "verdict-reject"
                verdict_text = "Does not meet transparency standards"
            
            st.markdown(f'<div class="{verdict_class}"><h4>OVERALL VERDICT</h4><h2>{verdict}</h2><p>{verdict_text}</p></div>', unsafe_allow_html=True)
        
        # Visualization
        st.markdown("### 📊 Transparency Profile")
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                name='Transparency Values',
                x=['Top', 'Mid', 'Bottom'],
                y=[data["top"], data["mid"], data["bottom"]],
                marker_color=['#4CAF50' if val > TRANSPARENCY_THRESHOLD else '#F44336' for val in [data["top"], data["mid"], data["bottom"]]],
                text=[f'{val:.3f}' for val in [data["top"], data["mid"], data["bottom"]]],
                textposition='auto',
            )
        ])
        
        # Add threshold line
        fig.add_hline(y=TRANSPARENCY_THRESHOLD, line_dash="dash", line_color="orange", 
                     annotation_text=f"Threshold ({TRANSPARENCY_THRESHOLD})", 
                     annotation_position="bottom right")
        
        fig.update_layout(
            title="Bottle Transparency Measurements",
            yaxis_title="Transparency Value",
            showlegend=False,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display timestamp
        timestamp = datetime.fromtimestamp(data["timestamp"] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        st.caption(f"📅 Measurement taken at: {timestamp}")
    
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #512DA8 0%, #7E57C2 100%); color: white; padding: 2rem; border-radius: 12px; text-align: center;">
            <h3 style="margin: 0; font-size: 1.3rem;">🔮 No Transparency Data Available</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem;">Fetch data from IoT system or enter manual values to begin analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # History section
    if st.session_state.transparency_history:
        st.markdown("### 📜 Measurement History")
        
        # Create dataframe for history
        history_data = []
        for entry in st.session_state.transparency_history[-10:]:  # Show last 10 entries
            history_data.append({
                'timestamp': datetime.fromtimestamp(entry["timestamp"] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'top': f"{entry['top']:.3f}",
                'mid': f"{entry['mid']:.3f}",
                'bottom': f"{entry['bottom']:.3f}",
                'Overall Status': '✅ PASS' if all([
                    entry['top'] > TRANSPARENCY_THRESHOLD,
                    entry['mid'] > TRANSPARENCY_THRESHOLD,
                    entry['bottom'] > TRANSPARENCY_THRESHOLD
                ]) else '❌ FAIL'
            })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            # Display table
            st.dataframe(
                history_df[['timestamp', 'top', 'mid', 'bottom', 'Overall Status']].sort_values('timestamp', ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
            # Summary statistics
            st.markdown("#### 📊 Historical Summary")
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            
            with col_sum1:
                pass_count = sum(1 for entry in st.session_state.transparency_history if all([
                    entry['top'] > TRANSPARENCY_THRESHOLD,
                    entry['mid'] > TRANSPARENCY_THRESHOLD,
                    entry['bottom'] > TRANSPARENCY_THRESHOLD
                ]))
                st.metric("✅ Passed Bottles", pass_count)
            
            with col_sum2:
                fail_count = len(st.session_state.transparency_history) - pass_count
                st.metric("❌ Failed Bottles", fail_count)
            
            with col_sum3:
                if len(st.session_state.transparency_history) > 0:
                    pass_rate = (pass_count / len(st.session_state.transparency_history)) * 100
                    st.metric("📈 Pass Rate", f"{pass_rate:.1f}%")

with tab4:
    # =====================================================
    # 4️⃣ ANALYZE UPLOADED IMAGES (moved from tab3)
    # =====================================================

    st.markdown('<h2 class="sub-header">🔍 Analyze Uploaded Images</h2>', unsafe_allow_html=True)
    
    # Collect uploaded images
    uploaded_images = [img for img in [img_mouth, img_angle1, img_angle2, img_angle3] if img is not None]
    
    st.markdown("<br>", unsafe_allow_html=True)
    if uploaded_images:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #512DA8 0%, #7E57C2 100%); color: white; padding: 1rem; border-radius: 12px; text-align: center;">
            <span style="font-size: 1.2rem; font-weight: bold;">📸 {len(uploaded_images)} image(s) ready for analysis</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        # Show small previews of uploaded images
        if uploaded_images:
            st.markdown("**Image Previews:**")
            preview_cols = st.columns(min(4, len(uploaded_images)))
            for idx, img in enumerate(uploaded_images):
                with preview_cols[idx % 4]:
                    st.image(img, use_container_width=True, caption=f"Image {idx+1}", output_format="JPEG")
    
    # Enhanced analyze button
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_disabled = len(uploaded_images) == 0
    button_text = "🚀 Start Analysis" if not analyze_disabled else "📝 Please upload images first"
    
    if st.button(button_text, type="primary", use_container_width=True, disabled=analyze_disabled):
        if uploaded_images:  # Proceed even if just one image
            files_to_send = [('files', (img.name, img.getvalue(), img.type)) for img in uploaded_images]

            with st.spinner("🔍 Analyzing images... This may take a few moments."):
                try:
                    response = requests.post(FASTAPI_URL, files=files_to_send)

                    if response.status_code == 200:
                        results_data = response.json()

                        # --- DEFECT ANALYSIS REPORT ---
                        st.markdown('<h2 class="sub-header">📋 Defect Analysis Report</h2>', unsafe_allow_html=True)

                        defect_found = {defect: "No" for defect in REJECT_CLASSES}
                        detected_defects = []
                        final_verdict = "ACCEPT"

                        for result in results_data:
                            for detection in result['detections']:
                                class_name = detection['class_name']
                                confidence = detection['confidence']
                                if class_name in REJECT_CLASSES and confidence > confidence_threshold:
                                    defect_found[class_name] = "Yes"
                                    detected_defects.append({
                                        "defect": class_name,
                                        "confidence": confidence,
                                        "file": result['filename']
                                    })
                                    final_verdict = "REJECT"

                        report_col1, report_col2 = st.columns([2, 1])
                        with report_col1:
                            # Format the defect report with styling
                            defect_report = []
                            for defect, status in defect_found.items():
                                if status == "Yes":
                                    defect_report.append({
                                        "Defect Type": defect,
                                        "Status": f'<span class="defect-yes">{status}</span>'
                                    })
                                else:
                                    defect_report.append({
                                        "Defect Type": defect,
                                        "Status": f'<span class="defect-no">{status}</span>'
                                    })
                            
                            # Convert to DataFrame for display
                            defect_df = pd.DataFrame(defect_report)
                            st.markdown("**Defect Status Summary**")
                            # Use HTML to render styled dataframe
                            st.markdown(defect_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                            
                            # Show detailed defects if any
                            if detected_defects:
                                st.markdown("**Detailed Defect Information**")
                                detail_df = pd.DataFrame([
                                    {
                                        "File": det['file'],
                                        "Defect": det['defect'],
                                        "Confidence": f"{det['confidence']:.2%}"
                                    } for det in detected_defects
                                ])
                                st.dataframe(detail_df, use_container_width=True, hide_index=True)

                        with report_col2:
                            st.markdown("**Final Verdict**")
                            if final_verdict == "REJECT":
                                st.markdown('<div class="verdict-reject">', unsafe_allow_html=True)
                                st.markdown("🔴 **REJECT**")
                                st.markdown(f"**{len(detected_defects)}** defect(s) found")
                                st.markdown("Bottle fails quality check")
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="verdict-accept">', unsafe_allow_html=True)
                                st.markdown("✅ **ACCEPT**")
                                st.markdown("No critical defects")
                                st.markdown("Bottle meets quality standards")
                                st.markdown('</div>', unsafe_allow_html=True)

                        # --- VISUAL RESULTS ---
                        st.markdown('<h2 class="sub-header">🖼️ Visual Inspection Results</h2>', unsafe_allow_html=True)
                        st.write("Below are the original images alongside the model's detection results:")

                        for i, result in enumerate(results_data):
                            filename = result['filename']
                            st.markdown(f"**Image {i+1}: {filename}**")
                            
                            original_image = next((img for img in uploaded_images if img.name == filename), None)

                            if original_image:
                                res_col1, res_col2 = st.columns(2)
                                with res_col1:
                                    st.markdown('<div class="image-preview">', unsafe_allow_html=True)
                                    st.image(original_image, caption="Original Image", use_container_width=True, output_format="JPEG")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with res_col2:
                                    st.markdown('<div class="image-preview">', unsafe_allow_html=True)
                                    encoded_image = result['image_plotted']
                                    decoded_image_bytes = base64.b64decode(encoded_image)
                                    st.image(decoded_image_bytes, caption="Detection Result", use_container_width=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Show detections for this image
                                image_defects = [det for det in result['detections'] if det['confidence'] > confidence_threshold]
                                if image_defects:
                                    defect_list = ", ".join([f"{det['class_name']} ({det['confidence']:.2%})" for det in image_defects])
                                    st.warning(f"Defects in this image: {defect_list}")
                                else:
                                    st.success("No defects detected in this image")
                                    
                            if i < len(results_data) - 1:
                                st.markdown("---")
                    else:
                        st.error(f"Error from server: {response.status_code} - {response.text}")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("⚠️ Please upload at least one image before starting the analysis.")
    elif len(uploaded_images) == 0:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2196F3 0%, #42A5F5 100%); color: white; padding: 1.5rem; border-radius: 12px; text-align: center;">
            <span style="font-size: 1.1rem;">👆 Upload images in the 'Upload Images' tab to enable analysis</span>
        </div>
        """, unsafe_allow_html=True)

with tab5:
    # =====================================================
    # 5️⃣ COMPREHENSIVE ANALYSIS RESULTS
    # =====================================================
    
    st.markdown('<h2 class="sub-header">📊 Comprehensive Quality Analysis</h2>', unsafe_allow_html=True)
    
    # Display combined results if available
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.markdown("### 🖼️ Visual Inspection Summary")
        if 'transparency_data' in st.session_state and st.session_state.transparency_data:
            data = st.session_state.transparency_data
            transparency_status = "✅ PASS" if all([data["top"] > TRANSPARENCY_THRESHOLD, 
                                                  data["mid"] > TRANSPARENCY_THRESHOLD, 
                                                  data["bottom"] > TRANSPARENCY_THRESHOLD]) else "❌ FAIL"
            
            st.metric("Transparency Status", transparency_status)
        else:
            st.info("No transparency data available")
    
    with col_res2:
        st.markdown("### 🔍 Defect Detection Summary")
        if uploaded_images:
            st.metric("Images Ready for Analysis", len(uploaded_images))
        else:
            st.info("No images uploaded for defect analysis")
    
    # Overall Quality Dashboard
    st.markdown("### 📋 Overall Quality Dashboard")
    
    # Create a comprehensive quality report
    quality_report = []
    
    # Check transparency
    if 'transparency_data' in st.session_state and st.session_state.transparency_data:
        data = st.session_state.transparency_data
        transparency_pass = all([data["top"] > TRANSPARENCY_THRESHOLD, 
                               data["mid"] > TRANSPARENCY_THRESHOLD, 
                               data["bottom"] > TRANSPARENCY_THRESHOLD])
        
        quality_report.append({
            "Test": "Transparency Analysis",
            "Status": "✅ PASS" if transparency_pass else "❌ FAIL",
            "Details": f"Top: {data['top']:.3f}, Mid: {data['mid']:.3f}, Bottom: {data['bottom']:.3f}"
        })
    else:
        quality_report.append({
            "Test": "Transparency Analysis",
            "Status": "⚠️ PENDING",
            "Details": "No data available"
        })
    
    # Check defect detection readiness
    quality_report.append({
        "Test": "Defect Detection",
        "Status": "✅ READY" if uploaded_images else "⚠️ AWAITING IMAGES",
        "Details": f"{len(uploaded_images)} images uploaded" if uploaded_images else "Upload images to begin"
    })
    
    # Display quality report
    quality_df = pd.DataFrame(quality_report)
    st.dataframe(quality_df, use_container_width=True, hide_index=True)
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    if uploaded_images and 'transparency_data' in st.session_state and st.session_state.transparency_data:
        data = st.session_state.transparency_data
        transparency_pass = all([data["top"] > TRANSPARENCY_THRESHOLD, 
                               data["mid"] > TRANSPARENCY_THRESHOLD, 
                               data["bottom"] > TRANSPARENCY_THRESHOLD])
        
        if transparency_pass:
            st.success("✅ Bottle transparency meets quality standards. Proceed with visual defect analysis.")
        else:
            st.error("❌ Bottle transparency fails quality standards. Consider rejecting the bottle.")
        
        st.info("📝 For complete quality assessment: Run defect analysis on uploaded images.")
    else:
        st.warning("⚠️ Complete both transparency analysis and upload images for comprehensive quality assessment.")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7E57C2; font-size: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #f8f9ff 0%, #e8e2ff 100%); border-radius: 12px; border: .3rem solid #B39DDB;'>"
    "🥛 <strong>Milk Bottle Quality Inspector</strong> • • Research Project • • Group 34 • • "
    "</div>",
    unsafe_allow_html=True
)