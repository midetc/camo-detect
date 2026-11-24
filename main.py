import json
from pathlib import Path

from PIL import Image
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO

FOLDER = Path("C:/Users/Snake/PycharmProjects/camo-detect/")
YOLO_PATH = FOLDER / "./api/weights/yolo26s_5k.pt"

model = YOLO(YOLO_PATH)
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/analyze/")
def get_analyze(file: UploadFile):
    with Image.open(file.file) as image:
        image = image.convert("RGB")
        results = model.predict(source=image)
        return {
            "results": json.loads(results[0].to_json()),
        }
