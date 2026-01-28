# 🥛 Bottle Quality Detection System

## 📖 About The Project

**Bottle Quality Detection System** is an advanced AI-powered defect detection and transparency analysis system for premium milk bottle quality assurance. This research project, developed as part of a Bachelor's degree in Information and Communication Technology at the University of Ruhuna, Sri Lanka, provides a comprehensive solution for automating the inspection of reusable glass bottles in the dairy industry.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00C853?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com/yolov8)
[![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://firebase.google.com/)

### 🎓 Research Information
- **University**: University of Ruhuna, Sri Lanka
- **Faculty**: Faculty of Technology
- **Department**: Department of Information and Communication Technology
- **Research Project**: Group 34
- **Students**: 
  - TG/2020/721 - K.N.L. Perera
  - TG/2020/685 - G.A.K. Botheju
- **Supervisor**: Mr. U.G Shashitha Lakal
- **Submission Date**: December 2025

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technical Implementation](#technical-implementation)
- [Performance Metrics](#performance-metrics)
- [Installation Guide](#installation-guide)
- [Usage Instructions](#usage-instructions)
- [Project Structure](#project-structure)
- [Dataset Information](#dataset-information)
- [Results & Findings](#results--findings)
- [Future Enhancements](#future-enhancements)
- [Research Contributions](#research-contributions)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 🔍 Problem Statement

The Sri Lankan dairy industry faces significant challenges in sorting and inspecting reusable glass bottles:

- **Manual Sorting Inefficiency**: Current processes require extensive human labor and are time-consuming
- **High Error Rates**: Human inspection has 15-25% error rates, leading to brand contamination
- **Safety Concerns**: Manual handling of damaged bottles poses safety risks to workers
- **Quality Control Issues**: Defective bottles (cracks, faded labels, poor transparency) can enter production
- **Operational Costs**: Manual processes increase operational expenses and reduce throughput

## 💡 Solution Overview

This project provides an integrated automated solution combining:

1. **AI-Powered Visual Inspection**: YOLOv8-based defect detection for cracks, label quality, and brand identification
2. **IoT Transparency Measurement**: Hardware system using lasers and LDR sensors to quantify bottle clarity
3. **Real-time Web Interface**: Streamlit-based dashboard for comprehensive quality assessment
4. **Cloud Integration**: Firebase Realtime Database for data storage and retrieval

## ✨ Key Features

### 🔬 Multi-Modal Inspection
- **Brand Recognition**: Identifies Lucky Lanka bottles vs. competitor bottles
- **Defect Detection**: Detects body cracks, mouth cracks, and unclear labels
- **Multi-Angle Analysis**: Processes 4 images per bottle (mouth + 3 body angles)
- **Real-time Live Inspection**: Camera integration for on-the-spot quality checking

### 📊 Advanced Analytics
- **Transparency Analysis**: Measures bottle clarity using IoT sensors
- **Quality Dashboard**: Comprehensive quality metrics and compliance status
- **Section-wise Analysis**: Top, Mid, Bottom transparency measurements
- **Automated Decision Making**: ACCEPT/REJECT verdicts with reasoning

### 🔗 System Integration
- **Cloud Database**: Firebase integration for transparency data storage
- **REST API**: FastAPI backend for model inference
- **Web Interface**: Streamlit frontend with intuitive user experience
- **Hardware Integration**: Arduino/ESP32-based transparency measurement system

## 🏗️ System Architecture

### Backend Architecture (FastAPI)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │   YOLOv8 Model  │
│    Frontend     │────▶    Backend      │────▶     (PyTorch)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                            │
                    ┌───────┴────────┐
                    │                │
             ┌─────────────┐  ┌─────────────┐
             │  Firebase   │  │   SQLite    │
             │  Database   │  │   Database  │
             └─────────────┘  └─────────────┘
```

### Frontend Architecture (Streamlit)
```
┌─────────────────────────────────────────────────────┐
│                Streamlit Application                │
├─────────────┬─────────────┬────────────┬────────────┤
│ Upload      │ Live        │ Transparency│ Analysis  │
│ Images      │ Inspection  │ Detection  │ Results   │
├─────────────┴─────────────┴────────────┴────────────┤
│                Real-time Visualization              │
└─────────────────────────────────────────────────────┘
```

## 🛠️ Technical Implementation

### Core Technologies
- **Deep Learning**: YOLOv8 (Ultralytics) for object detection
- **Backend Framework**: FastAPI for REST API development
- **Frontend Framework**: Streamlit for web application
- **Database**: Firebase Realtime Database & SQLite
- **Hardware**: ESP32, Laser modules, LDR sensors, Stepper motors
- **Computer Vision**: OpenCV for image processing

### Model Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Precision** | 92.1% | 92.1% of positive predictions are correct |
| **Recall** | 92.1% | 92.1% of actual defects are detected |
| **mAP50** | 94.9% | Excellent detection accuracy at IoU=0.5 |
| **mAP50-95** | 80.4% | Strong performance across stringent IoU thresholds |
| **F1-Score** | 92.1% | Perfect balance between precision and recall |

### Hardware Components
| Component | Specification | Purpose |
|-----------|---------------|---------|
| Laser Line Module | Red Laser Line Module RYS1230 650nm 5mW | Project focused beam through bottle |
| LDR Sensors | LDR 12mm (RE0155) | Measure transmitted light intensity |
| Microcontroller | ESP32 Dev Module | Read analog values and control rotation |
| Stepper Motor | NEMA 17 | Rotate bottle for full surface inspection |
| Motor Driver | A4988 | Precise stepper motor control |
| Enclosure | Custom-built dark chamber | Eliminate ambient light interference |

## 📈 Performance Metrics

### Model Training Results (75 Epochs)
- **Training Time**: ~2 hours on Google Colab (Tesla T4 GPU)
- **Final mAP50**: 94.9%
- **Final Precision**: 92.1%
- **Final Recall**: 92.1%

### Per-Class Performance
| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| BodyCrack | 93.1% | 95.2% | 97.4% |
| MouthCrack | 94.5% | 92.8% | 97.8% |
| LuckyLogoUnclear | 91.2% | 89.3% | 92.5% |
| OtherLogo | 92.1% | 88.7% | 92.0% |
| UnclearLabel | 89.8% | 90.5% | 93.6% |

### Transparency System Accuracy
- **Classification Accuracy**: 94%
- **False Acceptance Rate**: 3%
- **False Rejection Rate**: 3%
- **Processing Time**: 8-10 seconds per bottle

## 🚀 Installation Guide

### Prerequisites
- Python 3.8 or higher
- Git
- CUDA-capable GPU (optional, for faster inference)
- Arduino IDE (for hardware programming)

### Step 1: Clone the Repository
```bash
git clone https://github.com/navodlakshan/Bottle-Quality-Detection-System.git
cd Bottle-Quality-Detection-System
```

### Step 2: Backend Setup
```bash
cd Testing_APP/Back_End
pip install -r requirements.txt

# Start the FastAPI server
python main.py
```

### Step 3: Frontend Setup
```bash
cd ../Front_End
pip install -r requirements.txt

# Start the Streamlit application
streamlit run app.py
```

### Step 4: Configure Firebase
1. Create a Firebase project at [firebase.google.com](https://firebase.google.com/)
2. Enable Realtime Database
3. Update Firebase configuration in `app.py`:
```python
FIREBASE_CONFIG = {
    "apiKey": "YOUR_API_KEY",
    "authDomain": "YOUR_PROJECT.firebaseapp.com",
    "databaseURL": "https://YOUR_PROJECT-default-rtdb.firebaseio.com",
    "projectId": "YOUR_PROJECT_ID",
    # ... other configuration
}
```

### Step 5: Hardware Setup
1. Assemble the transparency detection system as per the schematic
2. Upload Arduino code to ESP32
3. Connect the system to power and test sensors

## 📁 Project Structure

```
BOTTLE-QUALITY-DETECTION-SYSTEM/
├── Testing_APP/
│   ├── Back_End/
│   │   ├── Models/
│   │   │   └── best.pt                    # Trained YOLOv8 model
│   │   ├── bottle_quality.db              # SQLite database
│   │   ├── main.py                        # FastAPI backend
│   │   ├── NOTES.txt
│   │   └── requirements.txt               # Backend dependencies
│   └── Front_End/
│       ├── app.py                         # Streamlit frontend
│       ├── NOTES.txt
│       └── requirements.txt               # Frontend dependencies
├── Milk_Bottel_Defect_Detection_Model_1.ipynb  # Model training notebook
├── milk-bottle.png                        # Sample bottle image
├── NOTES.txt
├── README.md                              # This file
└── .gitignore
```

## 📊 Dataset Information

### Collection Details
- **Total Images**: 805+ images
- **Physical Bottles**: 200+ bottles
- **Classes**: 7 (BodyCrack, MouthCrack, LuckyLogoUnclear, OtherLogo, UnclearLabel, etc.)
- **Brands Covered**: Lucky Lanka + 10+ competitor brands
- **Lighting Conditions**: Controlled + variable environments

### Data Augmentation
- Horizontal flips (50% probability)
- Random brightness/contrast (30% probability)
- Rotation (±25 degrees, 40% probability)
- Blur (4-pixel limit, 20% probability)
- Gaussian noise (20% probability)

### Class Balancing Strategy
1. **Undersampling**: Majority classes reduced to 800 instances each
2. **Oversampling**: Minority classes augmented to 800 instances each
3. **Validation**: Bounding box validation and sanitization

## 🎯 Usage Instructions

### 1. Image Upload
1. Navigate to the "UPLOAD IMAGES" tab
2. Upload four images: bottle mouth + three body angles
3. Supported formats: JPG, JPEG, PNG
4. Maximum file size: 200MB per file

### 2. Live Inspection
1. Go to the "LIVE INSPECTION" tab
2. Click "START CAMERA" to begin real-time detection
3. View detection results in real-time
4. Click "STOP CAMERA" to end the session

### 3. Transparency Detection
1. Navigate to "TRANSPARENCY DETECTION" tab
2. Choose data source: Firebase fetch or manual input
3. Click "Analyze Transparency" to get results
4. View transparency profile and verdict

### 4. Image Analysis
1. Go to "ANALYZE IMAGES" tab
2. Click "Start Analysis" to process uploaded images
3. View defect analysis report
4. Examine visual inspection results

### 5. Comprehensive Results
1. Navigate to "ANALYSIS FINAL RESULTS" tab
2. View combined quality assessment
3. Check recommendations for next steps

## 📈 Results & Findings

### Key Achievements
1. **High Accuracy**: Achieved 94.9% mAP50, exceeding the 90% target
2. **Real-time Processing**: 1-2 seconds per image for inference
3. **Integrated Solution**: Combined visual inspection with IoT transparency measurement
4. **User-friendly Interface**: Intuitive web application for operators

### Comparative Analysis
| Model | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| YOLOv5 | 92.2% | 91.4% | 94.3% | 72.81% |
| **YOLOv8** | **92.1%** | **92.1%** | **94.90%** | **80.4%** |
| YOLOv11 | 89.5% | 91.0% | 93.5% | 80.7% |

### Industrial Impact
- **Labor Reduction**: Potential to reduce manual sorting workforce by 80%
- **Error Reduction**: Decrease inspection errors from 15-25% to <5%
- **Throughput Increase**: Process bottles 5-10x faster than manual inspection
- **Quality Improvement**: Consistent application of quality standards

## 🔮 Future Enhancements

### Short-term Improvements
1. **Model Optimization**: Quantize model for edge deployment
2. **Speed Enhancement**: Optimize inference time for high-speed production lines
3. **Dataset Expansion**: Include more bottle brands and defect types
4. **Mobile Application**: Develop mobile app for field inspections

### Long-term Vision
1. **Full Automation**: Integrate with robotic arms for automatic bottle handling
2. **Predictive Maintenance**: Use historical data to predict equipment failures
3. **Blockchain Integration**: Implement supply chain tracking for bottles
4. **Multi-factory Deployment**: Scale solution across multiple dairy plants

## 📚 Research Contributions

### Academic Contributions
1. **Novel Dataset**: Created comprehensive dataset of Sri Lankan dairy bottles
2. **Methodology**: Developed two-stage data balancing approach for class imbalance
3. **Integration**: First system combining YOLO-based defect detection with IoT transparency measurement
4. **Validation**: Extensive testing with real industrial data

### Practical Contributions
1. **Industry Solution**: Directly addresses challenges in Sri Lankan dairy industry
2. **Cost Reduction**: Significant potential for operational cost savings
3. **Quality Standardization**: Consistent application of quality criteria
4. **Knowledge Transfer**: Documentation for replication in similar industries

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

We express our sincere gratitude to:

- **Supervisor**: Mr. U.G Shashitha Lakal for continuous guidance and support
- **Department Head**: Prof. P.K.S.C. Jayasinghe for institutional support
- **Industry Partner**: Lucky Lanka Milk Processing Co. PLC for dataset collection support
- **University**: University of Ruhuna for providing research facilities
- **Open Source Community**: For the amazing tools and libraries that made this project possible

---

<div align="center">
  
**🥛 Milk Bottle Quality Inspector • Research Project • Group 34 •**

[![University of Ruhuna](https://img.shields.io/badge/University%20of%20Ruhuna-003366?style=for-the-badge&logo=university&logoColor=white)](https://www.ruh.ac.lk/)
[![B.ICT(Hons)](https://img.shields.io/badge/B.ICT(Hons)-512DA8?style=for-the-badge&logo=academic&logoColor=white)]()
[![DOI](https://img.shields.io/badge/DOI-10.13140/RG.2.2.12345.67890-blue?style=for-the-badge)]()

*"Quality is not an act, it is a habit." - Aristotle*

</div>
