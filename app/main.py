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

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Manga Translation Automation Pipeline",
    description="Hệ thống tự động dịch truyện tranh và chèn chữ thông minh.",
    version="1.0.0"
)

# Cấu hình CORS (Cho phép chia sẻ tài nguyên chéo nguồn để dễ dàng debug và tích hợp)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đảm bảo các thư mục cần thiết tồn tại trong không gian làm việc của dự án
DATA_DIR = os.path.abspath("data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
JOBS_DIR = os.path.join(DATA_DIR, "jobs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)

# Gắn thư mục data tĩnh để cho phép giao diện Client truy cập xem trước ảnh gốc và ảnh đã dịch
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
# Gắn thư mục static phục vụ các tệp tĩnh của giao diện (CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Cấu hình thư mục chứa các mẫu giao diện (Jinja2 Templates)
templates = Jinja2Templates(directory="app/templates")

# Bộ lưu trữ trạng thái tiến trình công việc toàn cục (Global Job Store)
# Cấu trúc dữ liệu: { job_id: { "status": str, "progress": float, "logs": list, "output_zip": str, "images": list } }
jobs = {}


def run_job_in_background(
    job_id: str, 
    zip_path: str, 
    api_key: str, 
    src_lang: str, 
    tone: str, 
    batch_size_pages: int,
    additional_instructions: str = ""
):
    """
    Thực thi quy trình dịch truyện tranh trong luồng chạy nền (background task) và cập nhật trạng thái phiên làm việc.
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
            additional_instructions=additional_instructions,
            status_callback=status_callback
        )
        
        # Chạy toàn bộ tiến trình dịch thuật (OCR -> Dịch -> Inpainting -> Vẽ chữ -> Đóng gói)
        pipeline.run_pipeline(zip_path, output_zip_path, temp_dir)
        
        # Tìm danh sách ảnh đã xử lý xong để trả về giao diện phục vụ so sánh kết quả gốc/dịch
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
        # Xóa tệp ZIP đã tải lên để giải phóng dung lượng bộ nhớ đệm trên máy chủ
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """
    Trả về giao diện trang chủ chính của ứng dụng dịch truyện tranh.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_zip(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Form(...),
    src_lang: str = Form("en"),
    tone: str = Form("tự nhiên"),
    batch_size_pages: int = Form(10),
    additional_instructions: str = Form("")
):
    """
    Tiếp nhận tệp ZIP truyện tranh tải lên từ Client, khởi tạo tiến trình xử lý ngầm và trả về ID phiên làm việc.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận tệp tin định dạng .zip")
        
    job_id = str(uuid.uuid4())
    zip_filename = f"{job_id}.zip"
    zip_path = os.path.join(UPLOAD_DIR, zip_filename)
    
    # Lưu tệp tin tải lên vào bộ nhớ đệm tạm thời trên máy chủ
    try:
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể lưu tệp tải lên: {str(e)}")
        
    # Khởi tạo thông tin phiên làm việc trong danh sách quản lý trạng thái
    jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "logs": ["HỆ THỐNG: Đã nhận tệp ZIP. Bắt đầu khởi chạy tiến trình dịch ngầm..."],
        "output_zip": None,
        "images": []
    }
    
    # Kích hoạt tác vụ dịch chạy ngầm thông qua BackgroundTasks của FastAPI
    background_tasks.add_task(
        run_job_in_background,
        job_id,
        zip_path,
        api_key,
        src_lang,
        tone,
        batch_size_pages,
        additional_instructions
    )
    
    return {"job_id": job_id}


@app.get("/api/stream-progress")
async def stream_progress(job_id: str):
    """
    Truyền phát trực tiếp logs tiến trình và cập nhật phần trăm hoàn thành về Client theo thời gian thực (SSE).
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
                
            # Gửi đi các dòng log mới phát sinh trong tiến trình chạy ngầm
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
    Cho phép tải về tệp ZIP chứa toàn bộ các ảnh truyện tranh kết quả đã được dịch và vẽ chữ hoàn chỉnh.
    """
    job = jobs.get(job_id)
    if not job or job["status"] != "completed" or not job["output_zip"] or not os.path.exists(job["output_zip"]):
        raise HTTPException(status_code=404, detail="Tệp kết quả dịch không tồn tại hoặc phiên làm việc chưa hoàn thành.")
        
    return FileResponse(
        path=job["output_zip"],
        filename="manga_translated.zip",
        media_type="application/zip"
    )
