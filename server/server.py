import mkcfup
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

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
    context = {'request': request, 'title': 'Home Page', 'message': 'Hello, FastAPI with Jinja2!'}
    return templates.TemplateResponse("index.html", context)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")