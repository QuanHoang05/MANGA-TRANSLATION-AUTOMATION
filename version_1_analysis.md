# Phân Tích Nhược Điểm Phiên Bản 1 (Version 1 Analysis)

Tài liệu này tổng hợp danh sách các nhược điểm, lỗi đồ họa, và hạn chế kỹ thuật được ghi nhận trong phiên bản 1 (Version 1) của hệ thống Manga Translation Automation Pipeline. Đây là cơ sở để định hình và phát triển các tính năng đột phá cho phiên bản 2 (Version 2).

---

## 🐛 Danh sách nhược điểm của Version 1 (Vấn đề về Xử lý Ảnh & Đồ họa)

### 1. Lỗi định vị và nhét chữ (Text Typesetting)
* **Hiện trạng:** Chữ tiếng Việt mới chèn chưa có thuật toán tự động xuống dòng và căn giữa tối ưu.
* **Hậu quả:** Chữ hay bị tràn ra ngoài viền, không khớp với không gian hình học của bong bóng thoại, hoặc font chữ bị bóp co quá nhỏ đến mức không đọc được khi cố nhét chuỗi dài vào khung giới hạn.

### 2. Lỗi xóa nền bong bóng thoại (Bubble Inpainting)
* **Hiện trạng:** Việc tẩy sạch chữ Trung Quốc cũ trước khi ghi đè chữ mới chưa triệt để.
* **Hậu quả:** Khi box vẽ chữ mới đè lên box chữ cũ, đôi khi các nét chữ cũ thò ra ngoài rìa hoặc tạo các mảng màu lem nhem, chồng chéo cực kỳ rối mắt.

### 3. Lỗi chữ không nền (Floating Text / SFX)
* **Hiện trạng:** Chưa xử lý được các đoạn chữ viết lơ lửng trên nền phong cảnh, hiệu ứng hành động (SFX - Sound Effects).
* **Hậu quả:** Hiện tại hệ thống không thể xóa được chữ cũ lơ lửng này nếu không làm loang lổ, hỏng cấu trúc tranh gốc phía dưới.

### 4. Lỗi ảnh bị cắt ngang (Sliced Images)
* **Hiện trạng:** Ảnh tải từ web truyện thường bị cắt ngang xương (webtoon dọc bị phân tách ngẫu nhiên thành các tệp ảnh riêng lẻ), làm đứt đôi bong bóng thoại nằm ở mép ảnh cắt.
* **Hậu quả:** Mô hình OCR không thể nhận diện được các phần bong bóng thoại bị đứt gãy này, dẫn đến việc bỏ sót không dịch các dòng chữ ở ranh giới ảnh.
