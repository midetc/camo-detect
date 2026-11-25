from pathlib import Path

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
from ultralytics.models.sam import SAM

FOLDER = Path("C:/Users/Snake/PycharmProjects/camo-detect/")
YOLO_PATH = FOLDER / "./api/weights/yolo26s_5k.pt"
SAM_PATH = FOLDER / "./api/weights/sam2.1_s.pt"

app = FastAPI()

model_yolo = YOLO(YOLO_PATH)
# model_sam = SAM(SAM_PATH)
MODEL = {
    "name": "YOLO26s",
    "task": "detection",
    "classes": ["camouflaged_target", "camouflaged_vehicle"],
    "trained_on": "COD10K, CAMO, NC4K, MHCD",
    "imgsz": 640,
    "precision": 0.76,
    "recall": 0.57,
    "map50": 0.64,
    "map50_95": 0.46,
}


@app.get("/health/")
def health():
    return {
        "ready": model_yolo is not None,
        "weights": str(YOLO_PATH),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": MODEL,
    }


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/analyze/")
async def get_analyze(file: UploadFile):
    with Image.open(file.file) as image:
        image = image.convert("RGB")
        width, height = image.size
        result = model_yolo.predict(source=image, conf=0.25, imgsz=640, verbose=False)[0]

        objects = []
        for box in result.boxes:
            x1, y1, x2, y2 = [round(int(v)) for v in box.xyxy[0]]
            confidence = round(float(box.conf[0]), 4)
            class_name = result.names[int(box.cls[0])]
            objects.append({
                "class": class_name,
                "confidence": confidence,
                "bbox_xyxy": [x1, y1, x2, y2],
            })

        if not objects:
            decision = "unseen"
        elif max(item["confidence"] for item in objects) >= 0.5:
            decision = "seen"
        else:
            decision = "almost seen"

        return {
            "width": width,
            "height": height,
            "conf_threshold": 0.25,
            "imgsz": 640,
            "decision": decision,
            "count": len(objects),
            "objects": objects,
        }