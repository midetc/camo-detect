import json
from pathlib import Path

from PIL import Image
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import torch


FOLDER = Path("C:/Users/Snake/PycharmProjects/camo-detect/")
YOLO_PATH = FOLDER / "./api/weights/yolo26s_5k.pt"

app = FastAPI()


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
        "ready": YOLO_PATH.exists(),
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
        model = YOLO(YOLO_PATH)
        results = model.predict(source=image)
        return {
            "results": json.loads(results[0].to_json()),
        }
