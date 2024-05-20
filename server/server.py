
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
from kcf import ObjectTracker

IMPORT_KCF = True
if IMPORT_KCF:
    import sys
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 获取项目的根路径
    project_root = os.path.abspath(os.path.join(current_path, '..'))
    # 将项目根路径添加到sys.path
    sys.path.append(project_root)
    import KCF


# import kcf


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
    img_path: str


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
    # print(f"Received coordinates: {coords}")

    x, y, w, h, img_path = coords.startX, coords.startY, coords.width, coords.height, coords.img_path

    base_path = os.path.dirname(img_path)
    tracker = KCF.ObjectTracker()

    processedImages = []
    try:
        first_image = cv2.imread(img_path)
        roi = (x, y, w, h)
        tracker.initialize_first_frame(first_image, roi)
        files = os.listdir(base_path)
        files.sort()
        for file in files:
            if file.endswith('.jpg'):
                path = base_path + "/" + file
                img = cv2.imread(path)
                x, y, w, h = tracker.update_tracker(img)
                # print(x, y, w, h)
                cv2.rectangle(img, (int(x), int(y)), (int(
                    x + w), int(y + h)), (0, 255, 255), 1)
                # print(6)
                cv2.imwrite(path, img)
                processedImages.append(path)
    except Exception as e:
        print(e)
        return {"message": "Error occurred while processing coordinates"}

    return {"processed_images": processedImages}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
