from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from ultralytics import YOLO
from PIL import Image
import io
import base64
import torch

# Initialize the FastAPI app
app = FastAPI(
    title="YOLOv8 Milk Bottle Defect Detection API",
    description="API for detecting defects in milk bottle images using YOLOv8",
    version="2.0"
)

# Add CORS middleware to allow frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained YOLOv8 model
model = YOLO('./models/best.pt')

# Get class names from the model
CLASS_NAMES = model.names

@app.get("/")
async def root():
    return {"message": "Milk Bottle Defect Detection API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...)):
    """
    Receives a list of image files, runs YOLOv8 prediction on each,
    and returns a JSON object containing:
    1. The processed images encoded in Base64.
    2. The raw detection data (class, confidence, bbox) for each image.
    """
    results_list = []

    for file in files:
        # 1. Read the image from the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 2. Run the YOLOv8 model on the image
        results = model(image)
        result = results[0] # Get the first result object

        # 3. Plot the results on the image for visualization
        res_plotted = result.plot()
        result_image_pil = Image.fromarray(res_plotted[:, :, ::-1])
        
        # 4. Encode the plotted image to Base64
        buf = io.BytesIO()
        result_image_pil.save(buf, format="JPEG")
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')

        # 5. Extract raw detection data
        detections = []
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            detections.append({
                "class_name": CLASS_NAMES[class_id],
                "confidence": confidence,
                "box": box.xyxy[0].tolist() # Bounding box coordinates
            })
        
        # 6. Append the result for this image to our list
        results_list.append({
            "filename": file.filename,
            "image_plotted": encoded_image,
            "detections": detections
        })

    # 7. Return the full list of results as a JSON response
    return results_list