import mkcfup
from fastapi import FastAPI, Request, File, UploadFile
from zipfile import ZipFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import os, shutil

import uvicorn

INPUT_DIR = '/d_workspace/KCFs-tracking-sys/sequences'
OUTPUT_DIR = '/d_workspace/KCFs-tracking-sys/res'

# if __name__ == '__main__':
#     item = 'Biker'
#     fps = mkcfup.inference(INPUT_DIR, OUTPUT_DIR, item)
#     print(f"FPS of {item} is {fps}")

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    context = {'request': request}
    return templates.TemplateResponse("index.html", context)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
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

    file_location = os.path.join(upload_folder, file.filename)  # Correct the file location path

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
    
    mkcfup.inference(input_dir, output_dir, folder_name)
    
    return {"message": "File uploaded and unpacked successfully!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")