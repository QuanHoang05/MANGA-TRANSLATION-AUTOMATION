# Manga Translation Automation Pipeline v2.0 🚀

Hệ thống **tự động dịch truyện tranh bằng AI** — tích hợp hoàn chỉnh từ quét chữ OCR, dịch thuật gộp ngữ cảnh, xóa chữ cũ thông minh đến vẽ chữ mới. Hỗ trợ cả hai luồng: **dịch tự động qua Gemini API** và **nhập bản dịch thủ công JSON**.

## 📊 Thống kê tương tác Repo
![GitHub Traffic](https://raw.githubusercontent.com/QuanHoang05/MANGA-TRANSLATION-AUTOMATION/main/traffic.svg)

---

## ✨ Tính Năng v2.0

| Tính năng | Mô tả |
|-----------|-------|
| 🧠 **OCR đa ngôn ngữ** | PaddleOCR hỗ trợ Tiếng Anh, Trung, Nhật, Hàn. Tự phát hiện ngôn ngữ (Auto-detect) |
| 🔗 **Stitching & Smart Slicing** | Khâu dọc tất cả trang → cắt thông minh theo khoảng trắng (Row Variance) để OCR không bỏ sót bong bóng thoại bị cắt ngang |
| 🤖 **Dịch gộp ngữ cảnh** | Gửi cụm trang (batch) lên Gemini Flash để AI hiểu xưng hô và mạch truyện, tránh dịch rời rạc |
| ✏️ **Inpainting lai** | Bong bóng thoại → tô đè màu nền gốc; Chữ SFX lơ lửng → xóa bằng LaMa-ONNX hoặc OpenCV |
| 🔤 **Typesetting thông minh** | Tự chọn cỡ font vừa khít, ngắt dòng, căn giữa. Font Nunito Bold hỗ trợ Unicode đầy đủ |
| 📦 **Hỗ trợ ảnh lẻ & ZIP** | Kéo thả 1 file ZIP hoặc chọn nhiều ảnh trực tiếp |
| 🔄 **Dịch thủ công (không API)** | OCR xong → tải JSON → tự dịch → paste lại → pipeline tiếp tục vẽ chữ |
| 🔁 **Auto-reset sau hoàn thành** | Giao diện tự reset sau 3s, giữ nguyên API Key và Prompt phụ |
| 📜 **Prompt phụ xưng hô** | Truyền hướng dẫn xưng hô riêng (VD: "Muội - Huynh") vào prompt Gemini |

---

## 🛠️ Quy Trình 5 Bước (Workflow)

```
ZIP / Ảnh lẻ
     │
     ▼ BƯỚC 1 — Thu thập & Stitching
  Khâu dọc toàn bộ ảnh → Smart Slice tại khoảng trắng
     │
     ▼ BƯỚC 2 — OCR (PaddleOCR)
  Quét chữ từng trang → Group bong bóng → Xuất JSON OCR
     │
     ▼ BƯỚC 3 — Dịch thuật
  [Có API Key] → Gửi lên Gemini Flash theo batch
  [Không API]  → Tạm dừng → Người dùng paste JSON bản dịch
  [Có JSON sẵn] → Áp dụng ngay bản dịch tùy chỉnh
     │
     ▼ BƯỚC 4 — Inpainting & Typesetting
  Xóa chữ cũ (LaMa / OpenCV) → Vẽ chữ Việt mới (Pillow)
     │
     ▼ BƯỚC 5 — Đóng gói
  Ghép ảnh dọc (Webtoon) + Nén ZIP → Sẵn sàng tải về
```

---

## 💻 Cách Khởi Chạy

### Cách 1: Docker (Khuyên dùng)

```bash
# Build image
docker build -t manga-translator .

# Chạy container
docker run -d -p 8000:8000 --name manga-translator-container manga-translator
```

Truy cập giao diện tại: [http://localhost:8000](http://localhost:8000)

---

### Cách 2: Local Development

```bash
# Cài thư viện
pip install -r requirements.txt

# Chạy server (tự động tải font Nunito Bold nếu chưa có)
python run.py
```

Mở trình duyệt tại: [http://localhost:8000](http://localhost:8000)

---

### Cách 3: Google Colab (Dùng GPU T4 miễn phí)

```python
# 1. Clone và cài đặt
!git clone https://github.com/QuanHoang05/MANGA-TRANSLATION-AUTOMATION.git
%cd MANGA-TRANSLATION-AUTOMATION
!pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
!pip install -r requirements.txt
!pip install pyngrok

# 2. Chạy server với ngrok tunnel
from pyngrok import ngrok
ngrok.set_auth_token("NGROK_TOKEN_CỦA_BẠN")
public_url = ngrok.connect(8000)
print("🌐 Truy cập tại:", public_url.public_url)
!python run.py
```

---

## 🎮 Hướng Dẫn Sử Dụng

### Chế độ 1: Dịch tự động (Gemini API)

1. Nhập **Gemini API Key** (miễn phí tại [Google AI Studio](https://aistudio.google.com/))
2. Chọn ngôn ngữ gốc / dịch, văn phong
3. *(Tuỳ chọn)* Nhập **Prompt phụ** hướng dẫn xưng hô đặc biệt
4. Tải lên file ZIP hoặc nhiều ảnh → Nhấn **Bắt Đầu Dịch Tự Động**
5. Khi hoàn thành: tải ZIP kết quả hoặc ảnh ghép dọc Webtoon

### Chế độ 2: OCR + Dịch thủ công (không cần API Key)

1. Để trống API Key → Tải file lên → Nhấn **Bắt Đầu**
2. Xác nhận chạy ở chế độ **Chỉ quét OCR**
3. Hệ thống quét xong và tạm dừng → Tải JSON OCR về
4. Tự dịch từng ô thoại → Paste JSON bản dịch vào ô **"Bản dịch JSON"**
5. Nhấn **Tiếp Tục Dịch** → Pipeline tự vẽ chữ và đóng gói

### Chế độ 3: Nhập bản dịch JSON trước khi chạy

1. Paste JSON bản dịch vào ô **"Bản dịch JSON"** trước khi nhấn Start
2. Pipeline sẽ dùng bản dịch đó, bỏ qua bước gọi API

---

## 📁 Cấu Trúc Dự Án

```
MANGA-TRANSLATION-AUTOMATION/
├── .github/
│   └── workflows/
│       └── traffic.yml    # File cấu hình kịch bản cho Bot thống kê
├── app/
│   ├── main.py            # FastAPI routes & job management
│   ├── pipeline.py        # MangaPipeline (5 bước xử lý)
│   ├── static/
│   │   ├── css/style.css  # Giao diện Premium Dark Theme
│   │   └── js/app.js      # Frontend logic & SSE client
│   └── templates/
│       └── index.html     # Giao diện chính
├── fonts/
│   └── Nunito-Bold.ttf    # Font Unicode cho vẽ chữ Việt
├── data/                  # Thư mục lưu uploads & jobs (auto-created)
├── Dockerfile
├── requirements.txt
├── run.py                 # Entry point (tải font nếu chưa có)
├── traffic.svg            # Biểu đồ thống kê lượng truy cập (do Bot tự vẽ)
└── README.md
```

---

## 🔧 Biến Môi Trường & Cấu Hình

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| Batch size | 10 trang | Số trang gộp trong 1 lần gọi Gemini |
| Ngôn ngữ gốc | `en` | `en`, `ch`, `japan`, `korean`, `auto` |
| Ngôn ngữ dịch | `vi` | `vi`, `en`, `ch`, `japan`, `korean` |
| Văn phong | `tự nhiên` | `tự nhiên`, `cổ trang`, `dễ thương`, `lịch sự` |

---

## 📦 Thư Viện Chính

| Thư viện | Mục đích |
|----------|----------|
| `fastapi` + `uvicorn` | Web server & REST API |
| `paddleocr` | Nhận diện chữ OCR đa ngôn ngữ |
| `google-generativeai` | Gọi Gemini API dịch thuật |
| `opencv-python` | Xử lý ảnh & Inpainting |
| `Pillow` | Vẽ chữ tiếng Việt lên ảnh |
| `simple-lama-inpainting` | Xóa chữ SFX lơ lửng (AI) |
| `numpy` | Tính toán mảng/ma trận |

---

## ⚠️ Lưu Ý

- **RAM**: OCR và inpainting tốn RAM, khuyên dùng tối thiểu **4GB RAM** (8GB cho chương dài)
- **GPU**: PaddleOCR sẽ tự dùng GPU nếu có CUDA. Docker chạy trên CPU
- **Font**: `run.py` tự tải `Nunito-Bold.ttf` về thư mục `fonts/` khi khởi động
- **API Key**: Gemini API miễn phí có giới hạn 15 req/min, pipeline tự thử lại khi lỗi rate limit
