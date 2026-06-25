import os
import uuid
import json
import asyncio
import shutil
import traceback
import threading
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
    version="2.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đảm bảo các thư mục cần thiết tồn tại
DATA_DIR = os.path.abspath("data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
JOBS_DIR = os.path.join(DATA_DIR, "jobs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)

# Gắn thư mục data tĩnh để giao diện xem trước ảnh gốc và ảnh đã dịch
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
# Gắn thư mục static phục vụ CSS, JS
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Cấu hình Jinja2 Templates
templates = Jinja2Templates(directory="app/templates")

# Bộ lưu trữ trạng thái tiến trình công việc toàn cục
# Cấu trúc: { job_id: { "status", "progress", "logs", "output_zip", "images", "ocr_results", "translated_results" } }
jobs = {}


def run_job_in_background(
    job_id: str,
    zip_path: str,
    api_key: str,
    src_lang: str,
    tgt_lang: str,
    tone: str,
    batch_size_pages: int,
    additional_instructions: str = "",
    custom_translation: str = "",
    det_db_unclip_ratio: float = 1.6,
    det_db_box_thresh: float = 0.6
):
    """
    Thực thi quy trình dịch truyện tranh trong luồng nền và cập nhật trạng thái phiên làm việc.
    """
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    output_zip_path = os.path.join(job_dir, "translated.zip")
    temp_dir = os.path.join(job_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    def status_callback(message: str, percent: float, event_type: str = None, data=None):
        jobs[job_id]["logs"].append(message)
        jobs[job_id]["progress"] = percent
        if event_type == "ocr_completed":
            jobs[job_id]["ocr_results"] = data
        elif event_type == "translation_completed":
            jobs[job_id]["translated_results"] = data
        elif event_type == "paused":
            jobs[job_id]["status"] = "paused"
            event = jobs[job_id].get("resume_event")
            if event:
                event.wait()
                event.clear()
                jobs[job_id]["status"] = "processing"
                return jobs[job_id].get("custom_translation_input", "")
        return None

    try:
        pipeline = MangaPipeline(
            api_key=api_key,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            tone=tone,
            batch_size_pages=batch_size_pages,
            additional_instructions=additional_instructions,
            status_callback=status_callback,
            custom_translation=custom_translation,
            det_db_unclip_ratio=det_db_unclip_ratio,
            det_db_box_thresh=det_db_box_thresh
        )

        pipeline.run_pipeline(zip_path, output_zip_path, temp_dir)

        # Thu thập danh sách ảnh đã xử lý để giao diện so sánh trước/sau
        output_img_dir = os.path.join(temp_dir, "output")
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        processed_images = []
        if os.path.exists(output_img_dir):
            for file in sorted(os.listdir(output_img_dir)):
                if file.lower().endswith(image_extensions) and not file.startswith('._'):
                    processed_images.append(file)

        jobs[job_id]["images"] = processed_images
        jobs[job_id]["output_zip"] = output_zip_path
        jobs[job_id]["src_lang"] = pipeline.src_lang
        jobs[job_id]["status"] = "completed"

    except Exception as e:
        error_msg = f"LỖI HỆ THỐNG: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        jobs[job_id]["logs"].append(error_msg)
        jobs[job_id]["status"] = "failed"
    finally:
        # Xóa tệp upload tạm thời sau khi xử lý xong
        if os.path.exists(zip_path):
            try:
                if os.path.isdir(zip_path):
                    shutil.rmtree(zip_path)
                else:
                    os.remove(zip_path)
            except Exception:
                pass


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Trả về giao diện trang chủ chính."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    api_key: Optional[str] = Form(""),
    src_lang: str = Form("en"),
    tgt_lang: str = Form("vi"),
    tone: str = Form("tự nhiên"),
    batch_size_pages: int = Form(10),
    additional_instructions: str = Form(""),
    custom_translation: Optional[str] = Form(""),
    det_db_unclip_ratio: float = Form(1.6),
    det_db_box_thresh: float = Form(0.6)
):
    """
    Tiếp nhận tệp tin tải lên (1 file ZIP hoặc nhiều file ảnh đơn lẻ),
    khởi tạo tiến trình xử lý ngầm và trả về ID phiên làm việc.
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="Không nhận được tệp tin nào.")

    job_id = str(uuid.uuid4())

    # Kiểm tra xem có phải một file ZIP duy nhất không
    is_zip = len(files) == 1 and files[0].filename.lower().endswith('.zip')
    pipeline_input_path = ""

    if is_zip:
        zip_filename = f"{job_id}.zip"
        zip_path = os.path.join(UPLOAD_DIR, zip_filename)
        try:
            with open(zip_path, "wb") as buffer:
                shutil.copyfileobj(files[0].file, buffer)
            pipeline_input_path = zip_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Không thể lưu tệp ZIP: {str(e)}")
    else:
        # Nhiều file ảnh đơn lẻ → lưu vào thư mục riêng
        job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
        os.makedirs(job_upload_dir, exist_ok=True)

        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        saved_count = 0

        for uploaded_file in files:
            ext = os.path.splitext(uploaded_file.filename)[1].lower()
            if ext not in image_extensions:
                continue
            file_path = os.path.join(job_upload_dir, uploaded_file.filename)
            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(uploaded_file.file, buffer)
                saved_count += 1
            except Exception:
                continue

        if saved_count == 0:
            raise HTTPException(status_code=400, detail="Không tìm thấy tệp ảnh hợp lệ nào (.png, .jpg, .jpeg, .webp, .bmp)")

        pipeline_input_path = job_upload_dir

    # Khởi tạo trạng thái phiên làm việc mới (reset hoàn toàn)
    jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "logs": ["HỆ THỐNG: Đã nhận tệp tin đầu vào. Bắt đầu khởi chạy tiến trình dịch..."],
        "output_zip": None,
        "images": [],
        "ocr_results": None,
        "translated_results": None,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "resume_event": threading.Event(),
        "custom_translation_input": ""
    }

    background_tasks.add_task(
        run_job_in_background,
        job_id,
        pipeline_input_path,
        api_key,
        src_lang,
        tgt_lang,
        tone,
        batch_size_pages,
        additional_instructions,
        custom_translation,
        det_db_unclip_ratio,
        det_db_box_thresh
    )

    return {"job_id": job_id}


@app.get("/api/stream-progress")
async def stream_progress(job_id: str):
    """
    Truyền phát logs tiến trình và phần trăm hoàn thành về Client theo thời gian thực (SSE).
    """
    async def event_generator():
        if job_id not in jobs:
            yield f"data: {{\"error\": \"Không tìm thấy phiên làm việc {job_id}\"}}\n\n"
            return

        last_log_idx = 0
        sent_final_state = False

        while True:
            job = jobs.get(job_id)
            if not job:
                break

            # Gửi các dòng log mới
            logs_count = len(job["logs"])
            if last_log_idx < logs_count:
                for idx in range(last_log_idx, logs_count):
                    data = {
                        "status": job["status"],
                        "progress": job["progress"],
                        "log": job["logs"][idx],
                        "images": job["images"] if job["status"] == "completed" else [],
                        "ocr_results": job.get("ocr_results"),
                        "translated_results": job.get("translated_results"),
                        "src_lang": job.get("src_lang"),
                        "tgt_lang": job.get("tgt_lang")
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    if data["status"] in ("completed", "failed"):
                        sent_final_state = True
                last_log_idx = logs_count

            if job["status"] in ("completed", "failed"):
                if not sent_final_state:
                    data = {
                        "status": job["status"],
                        "progress": job["progress"],
                        "log": None,
                        "images": job["images"] if job["status"] == "completed" else [],
                        "ocr_results": job.get("ocr_results"),
                        "translated_results": job.get("translated_results"),
                        "src_lang": job.get("src_lang"),
                        "tgt_lang": job.get("tgt_lang")
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/continue/{job_id}")
async def continue_job(job_id: str, custom_translation: Optional[str] = Form("")):
    """
    Tiếp nhận bản dịch JSON từ người dùng và giải phóng sự kiện block để tiếp tục tiến trình.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên làm việc.")

    jobs[job_id]["custom_translation_input"] = custom_translation

    event = jobs[job_id].get("resume_event")
    if event:
        event.set()

    return {"status": "resumed"}


@app.get("/api/download/{job_id}")
async def download_translated(job_id: str):
    """
    Cho phép tải về tệp ZIP chứa toàn bộ các ảnh đã được dịch và vẽ chữ hoàn chỉnh.
    """
    job = jobs.get(job_id)
    if not job or job["status"] != "completed" or not job["output_zip"] or not os.path.exists(job["output_zip"]):
        raise HTTPException(status_code=404, detail="Tệp kết quả dịch không tồn tại hoặc phiên làm việc chưa hoàn thành.")

    return FileResponse(
        path=job["output_zip"],
        filename="manga_translated.zip",
        media_type="application/zip"
    )


@app.get("/api/download-stitched/{job_id}")
async def download_stitched(job_id: str):
    """
    Cho phép tải về ảnh ghép dọc duy nhất (Webtoon stitched image),
    hỗ trợ cả định dạng JPEG (.jpg) và PNG (.png).
    """
    job_dir = os.path.join(JOBS_DIR, job_id)
    temp_dir = os.path.join(job_dir, "temp")

    jpg_path = os.path.join(temp_dir, "translated_stitched.jpg")
    png_path = os.path.join(temp_dir, "translated_stitched.png")

    if os.path.exists(jpg_path):
        return FileResponse(path=jpg_path, filename="manga_translated_stitched.jpg", media_type="image/jpeg")
    elif os.path.exists(png_path):
        return FileResponse(path=png_path, filename="manga_translated_stitched.png", media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp ảnh ghép dọc. Vui lòng chạy lại tiến trình.")
