import mkcfup
from fastapi import FastAPI, Request, File, UploadFile
from zipfile import ZipFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import time
import asyncio

import uvicorn

# import kcf
from kcf import inference

INPUT_DIR = '/d_workspace/KCFs-tracking-sys/sequences'
OUTPUT_DIR = '/d_workspace/KCFs-tracking-sys/res'

app = FastAPI()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# 进度存储
progress = {"value": 0}


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


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global progress
    progress['value'] = 20
    folder_name = file.filename.split('.')[0]
    upload_folder = f"uploads/zips/{folder_name}"
    unzip_folder = f"uploads/files/{folder_name}"

    # 如果文件夹已经存在，则删除
    if os.path.exists(upload_folder):
        shutil.rmtree(upload_folder)
    if os.path.exists(unzip_folder):
        shutil.rmtree(unzip_folder)

    # 创建文件夹
    os.makedirs(upload_folder)
    os.makedirs(unzip_folder)

    # Correct the file location path
    file_location = os.path.join(upload_folder, file.filename)

    try:
        # 保存文件到 uploads/zips 文件夹
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())

        # 解压缩文件到指定目录
        with ZipFile(file_location, 'r') as zip_ref:
            zip_ref.extractall(unzip_folder)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    # inference
    items = os.listdir(unzip_folder)
    new_path = None
    while len(items) < 2:
        new_path = os.path.join(unzip_folder, items[0])
        items = os.listdir(new_path)
    input_dir = new_path
    output_dir = "result"

    # create a txt file
    with open(os.path.join(output_dir, f"results_{folder_name}.txt"), "w") as f:
        pass

    progress['value'] = 50

    mkcfup.inference(input_dir, output_dir, folder_name)

    progress['value'] = 75

    # TODO: 在这里要把推理结果进行处理，然后进行其他操作，在这里才是100%

    # await asyncio.sleep(3)

    progress['value'] = 100

    return {"message": "File uploaded and unpacked successfully!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
