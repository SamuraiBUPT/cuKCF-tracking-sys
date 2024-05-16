# import mkcfup
from fastapi import FastAPI, Request, File, UploadFile
from zipfile import ZipFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
import time
import asyncio
import cv2

import uvicorn

# import kcf
from kcf import inference

app = FastAPI()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# 进度存储
progress = {"value": 0}


class Coords(BaseModel):
    startX: int
    startY: int
    width: int
    height: int


async def progress_generator():
    global progress
    while True:
        await asyncio.sleep(0.2)  # 更新频率
        yield f"data: {progress['value']}\n\n"
        if progress['value'] == 100:
            break


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    context = {'request': request}
    return templates.TemplateResponse("index.html", context)


@app.get("/progress")
async def progress_stream_response(request: Request):
    return StreamingResponse(progress_generator(), media_type="text/event-stream")


@app.post("/slice")
async def slice_video(file: UploadFile = File(...)):
    # save this file
    filepath = os.path.join("uploads/videos", file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # silce this video to images
    video = cv2.VideoCapture(filepath)
    video_name = os.path.splitext(os.path.basename(filepath))[0]

    success, image = video.read()
    count = 0

    out_path = 'uploads/images/' + video_name   # create a folder
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    while success:
        cv2.imwrite(out_path + '/%08d.jpg' % count, image)  # slice
        count += 1
        success, image = video.read()
    video.release()

    base_path = out_path
    first_image = base_path + '/00000001.jpg'

    return {"message": "File uploaded and unpacked successfully!",
            "first_image_url": first_image}


@app.post("/process_coords")
async def process_coords(coords: Coords):
    # 在这里处理接收到的坐标
    print(f"Received coordinates: {coords}")
    return {"message": "Coordinates received successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
