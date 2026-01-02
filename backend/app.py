from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()
model = YOLO("../models/best.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    result = model(img, conf=0.25)[0]

    detections = []
    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        detections.append({
            "class": "cucumber",
            "confidence": float(b.conf[0]),
            "bbox": [x1, y1, x2, y2]
        })

    return {"count": len(detections), "detections": detections}
