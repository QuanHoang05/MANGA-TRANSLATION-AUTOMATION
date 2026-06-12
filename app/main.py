import os
import uuid
import asyncio
import shutil
import traceback
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from app.pipeline import MangaPipeline

# Initialize FastAPI App
app = FastAPI(
    title="Manga Translation Automation Pipeline",
    description="Automated pipeline to extract, translate, and typeset manga pages.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist within workspace
DATA_DIR = os.path.abspath("data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
JOBS_DIR = os.path.join(DATA_DIR, "jobs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)

# Mount data directory for image preview access
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
# Mount static assets
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates for frontend
templates = Jinja2Templates(directory="app/templates")

# Global job store
# Structure: { job_id: { "status": str, "progress": float, "logs": list, "output_zip": str, "images": list } }
jobs = {}


def run_job_in_background(
    job_id: str, 
    zip_path: str, 
    api_key: str, 
    src_lang: str, 
    tone: str, 
    batch_size_pages: int
):
    """
    Executes the translation pipeline in a background thread and updates job state.
    """
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    output_zip_path = os.path.join(job_dir, "translated.zip")
    temp_dir = os.path.join(job_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    def status_callback(message: str, percent: float):
        jobs[job_id]["logs"].append(message)
        jobs[job_id]["progress"] = percent
        
    try:
        pipeline = MangaPipeline(
            api_key=api_key,
            src_lang=src_lang,
            tone=tone,
            batch_size_pages=batch_size_pages,
            status_callback=status_callback
        )
        
        # Run entire pipeline
        pipeline.run_pipeline(zip_path, output_zip_path, temp_dir)
        
        # Find images processed to display in UI comparison
        output_img_dir = os.path.join(temp_dir, "output")
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        processed_images = []
        if os.path.exists(output_img_dir):
            for file in sorted(os.listdir(output_img_dir)):
                if file.lower().endswith(image_extensions) and not file.startswith('._'):
                    processed_images.append(file)
                    
        jobs[job_id]["images"] = processed_images
        jobs[job_id]["output_zip"] = output_zip_path
        jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        error_msg = f"LỖI HỆ THỐNG: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        jobs[job_id]["logs"].append(error_msg)
        jobs[job_id]["status"] = "failed"
    finally:
        # Clean uploaded zip file to save space
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """
    Serves the main application landing page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_zip(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Form(...),
    src_lang: str = Form("en"),
    tone: str = Form("tự nhiên"),
    batch_size_pages: int = Form(10)
):
    """
    Accepts the manga archive ZIP upload, initializes job, and starts background runner.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận tệp tin định dạng .zip")
        
    job_id = str(uuid.uuid4())
    zip_filename = f"{job_id}.zip"
    zip_path = os.path.join(UPLOAD_DIR, zip_filename)
    
    # Save uploaded file
    try:
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể lưu tệp tải lên: {str(e)}")
        
    # Create job entry
    jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "logs": ["HỆ THỐNG: Đã nhận tệp ZIP. Bắt đầu phiên làm việc..."],
        "output_zip": None,
        "images": []
    }
    
    # Run in background
    background_tasks.add_task(
        run_job_in_background,
        job_id,
        zip_path,
        api_key,
        src_lang,
        tone,
        batch_size_pages
    )
    
    return {"job_id": job_id}


@app.get("/api/stream-progress")
async def stream_progress(job_id: str):
    """
    Streams job processing progress logs and status updates to the client in real-time.
    """
    async def event_generator():
        if job_id not in jobs:
            yield f"data: {{\"error\": \"Không tìm thấy phiên làm việc {job_id}\"}}\n\n"
            return
            
        last_log_idx = 0
        while True:
            job = jobs.get(job_id)
            if not job:
                break
                
            # Send any new logs
            logs_count = len(job["logs"])
            if last_log_idx < logs_count:
                for idx in range(last_log_idx, logs_count):
                    data = {
                        "status": job["status"],
                        "progress": job["progress"],
                        "log": job["logs"][idx],
                        "images": job["images"] if job["status"] == "completed" else []
                    }
                    import json
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                last_log_idx = logs_count
                
            if job["status"] in ("completed", "failed"):
                break
                
            await asyncio.sleep(0.3)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/download/{job_id}")
async def download_translated(job_id: str):
    """
    Downloads the processed ZIP containing the translated images.
    """
    job = jobs.get(job_id)
    if not job or job["status"] != "completed" or not job["output_zip"] or not os.path.exists(job["output_zip"]):
        raise HTTPException(status_code=404, detail="Tệp kết quả dịch không tồn tại hoặc chưa hoàn thành.")
        
    return FileResponse(
        path=job["output_zip"],
        filename="manga_translated.zip",
        media_type="application/zip"
    )
