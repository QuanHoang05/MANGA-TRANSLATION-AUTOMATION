# Manga Translation Automation Pipeline 🚀

Hệ thống dịch thuật truyện tranh tự động bằng AI, thực hiện qua 5 bước tối ưu hóa: giải nén, quét chữ OCR (PaddleOCR), dịch thuật gộp ngữ cảnh bằng mô hình **Gemini 1.5 Flash API** (Structured JSON Output), tẩy xóa chữ cũ bằng **OpenCV Inpainting**, và tự động căn chỉnh vẽ chữ tiếng Việt mới (Pillow Text Wrapping & Auto-scaling).

---

## 🛠️ Quy Trình 5 Bước Hoạt Động (Workflow)

1. **Thu Thập & Giải Nén (Input Ingestion)**: Giải nén tệp ZIP chứa các trang ảnh truyện tranh (được sắp xếp theo thứ tự ví dụ `001.png`, `002.jpg`...).
2. **Quét Chữ AI (OCR Extraction)**: Sử dụng mô hình **PaddleOCR** quét qua toàn bộ ảnh, bóc tách tọa độ polygon và nội dung chữ gốc. Gán mã định danh ID duy nhất cho từng ô thoại theo định dạng `[TênẢnh-SốThứTự]`.
3. **Dịch Thuật Gộp (Context-Aware Translation)**: Gộp các câu thoại theo cụm trang ảnh (mặc định 10 trang) kèm Prompt hướng dẫn xưng hô. Gửi một request duy nhất lên **Google AI Studio (Gemini Flash)** để giữ tính đồng bộ nhân vật (ví dụ: Ta - Ngươi, Thiếu chủ - Đại nhân...). API được cấu hình bắt buộc trả về định dạng **JSON có cấu trúc**.
4. **Xử Lý Đồ Họa (Inpainting & Typesetting)**:
   - **Xóa chữ cũ**: Tạo mặt nạ (mask) từ tọa độ OCR, giãn nở (dilate) mặt nạ để xóa sạch mép chữ, sau đó dùng thuật toán `cv2.inpaint` để tái tạo nền tự nhiên 99%.
   - **Vẽ chữ mới**: Tự động ngắt dòng (Text Wrapping) theo độ rộng bong bóng thoại gốc, giảm dần kích thước font chữ (font size) để văn bản Việt vừa khít với chiều cao ô thoại. Vẽ chữ đen viền trắng để tăng độ tương phản rõ nét.
5. **Đóng Gói & Tải Về**: Lưu các ảnh hoàn chỉnh vào thư mục kết quả và nén lại thành tệp ZIP cho phép tải về qua giao diện web.

---

## 💻 Cách Khởi Chạy Hệ Thống

### Cách 1: Chạy cục bộ trên máy tính (Local Development)

#### 1. Cài đặt các thư viện Python:
Đảm bảo bạn đã cài đặt Python 3.10 trở lên. Trong cửa sổ terminal, chạy lệnh:
```bash
pip install -r requirements.txt
```

#### 2. Khởi chạy Server:
Chạy script `run.py` (tự động tải xuống font chữ hỗ trợ tiếng Việt **Nunito Bold** và mở server Uvicorn):
```bash
python run.py
```

#### 3. Sử dụng:
Mở trình duyệt web và truy cập địa chỉ: [http://localhost:8000](http://localhost:8000)

---

### Cách 2: Chạy trong Google Colab (Khuyên dùng - Sử dụng GPU T4 miễn phí)

Khi xử lý nhiều ảnh hoặc ảnh dung lượng lớn, nên chạy trên Google Colab để tận dụng GPU T4 giúp tăng tốc độ quét PaddleOCR đáng kể.

#### 1. Tạo một Notebook mới trên Colab, đổi loại môi trường (Runtime) sang **GPU T4**.
#### 2. Chạy khối mã cài đặt môi trường:
```python
# 1. Clone code dự án
!git clone https://github.com/QuanHoang05/MANGA-TRANSLATION-AUTOMATION.git
%cd MANGA-TRANSLATION-AUTOMATION

# 2. Cài đặt thư viện hỗ trợ GPU cho PaddlePaddle
!pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
!pip install -r requirements.txt

# 3. Cài đặt ngrok hoặc localtunnel để tạo đường link truy cập giao diện Web
!pip install pyngrok
```

#### 3. Khởi chạy server và lấy link truy cập:
```python
from pyngrok import ngrok
import os

# Cấu hình ngrok auth token (lấy từ https://dashboard.ngrok.com)
NGROK_TOKEN = "ĐIỀN_TOKEN_NGROK_CỦA_BẠN_VÀO_ĐÂY"
ngrok.set_auth_token(NGROK_TOKEN)

# Mở cổng tunnel 8000
public_url = ngrok.connect(8000)
print("👉 TRUY CẬP ĐƯỜNG LINK WEB GIAO DIỆN TẠI ĐÂY:", public_url.public_url)

# Chạy server FastAPI
!python run.py
```

---

### Cách 3: Đóng gói Docker Image (Docker Production)

Hệ thống đã có sẵn `Dockerfile` đóng gói các thư viện hệ thống cần thiết cho OpenCV và PaddleOCR chạy ổn định trên Linux.

#### 1. Build Docker Image:
```bash
docker build -t manga-translator .
```

#### 2. Chạy container:
```bash
docker run -d -p 8000:8000 --name manga-translator-container manga-translator
```
Truy cập qua trình duyệt tại: [http://localhost:8000](http://localhost:8000)

---

## 🌟 Các Tính Năng Cao Cấp Được Tích Hợp

- **Structured Output JSON**: Ngăn chặn hoàn toàn lỗi lệch định dạng hoặc mất mát ID thoại bằng cách cấu hình `response_schema` cứng trong API Gemini.
- **Micro-animations & Dark Theme Layout**: Giao diện Premium được thiết kế với phong cách Glassmorphic hiện đại, hỗ trợ drag-and-drop tải file trực quan và các thanh tiến trình nhịp nhàng.
- **Double Image Comparison**: Sau khi dịch xong, giao diện hiển thị bảng so sánh 2 nửa màn hình (Ảnh gốc chứa viền đỏ OCR và Ảnh kết quả dịch sạch sẽ) giúp người dùng dễ dàng thẩm định chất lượng bản dịch từng trang.
- **Auto-Font Scaling & Outline Stroke**: Chữ tiếng Việt được vẽ bằng màu đen cùng viền trắng dày 2px, tự động giảm size từ 28 xuống 10 cho đến khi vừa khít bong bóng thoại, cam kết dễ đọc trên mọi hình nền.
