import io

from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, Response
import requests

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <form action="/draw/" method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <button>Показать</button>
    </form>
    """


@app.post("/draw/")
async def draw(file: UploadFile):
    content = await file.read()
    response = requests.post(
        "http://127.0.0.1:8000/analyze/",
        files={"file": (file.filename, content, "image/jpeg")},
    )
    response.raise_for_status()
    result = response.json()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    pen = ImageDraw.Draw(image)
    for obj in result["objects"]:
        color = "yellow" if obj["class"] == "camouflaged_target" else "blue"
        pen.rectangle(obj["bbox_xyxy"], outline=color, width=3)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return Response(content=buf.getvalue(), media_type="image/jpeg")