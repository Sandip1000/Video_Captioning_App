from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from video_utils import get_video_tensor

from model import *

app = FastAPI()




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# @app.post("/generate_caption_model1/")
# async def generate_caption(file: UploadFile = File(...)):
#     video_path = os.path.join(UPLOAD_DIR, file.filename)

#     with open(video_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     video_tensor = get_video_tensor(video_path)
#     captions = model1.generate_caption(video_tensor)

#     return {
#         "filename": file.filename,
#         "caption": captions
#     }


@app.post("/generate_caption_model2/")
async def generate_caption(file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    video_tensor = get_video_tensor(video_path)
    captions = model2.generate_caption(video_tensor)

    return {
        "filename": file.filename,
        "caption": captions
    }


# @app.post("/generate_caption_model3/")
# async def generate_caption(file: UploadFile = File(...)):
#     video_path = os.path.join(UPLOAD_DIR, file.filename)

#     with open(video_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     video_tensor = get_video_tensor(video_path)
#     captions = model3.generate_caption(video_tensor)

#     return {
#         "filename": file.filename,
#         "caption": captions
#     }



@app.get("/")
def hello():
    return {"Video Caption Generator In Nepali Language API "}

