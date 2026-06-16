import uvicorn
import os

if __name__ == "__main__":
    # Đảm bảo thư mục fonts tồn tại trên máy cục bộ
    os.makedirs("fonts", exist_ok=True)
    font_path = os.path.join("fonts", "Nunito-Bold.ttf")
    if not os.path.exists(font_path):
        print("Đang tự động tải về font chữ Nunito-Bold hỗ trợ tiếng Việt...")
        import urllib.request
        try:
            urllib.request.urlretrieve(
                "https://github.com/google/fonts/raw/main/ofl/nunito/Nunito%5Bwght%5D.ttf",
                font_path
            )
            print("Đã tải xong font chữ Nunito-Bold thành công.")
        except Exception as e:
            print(f"Cảnh báo: Không thể tải tự động font chữ: {e}. Hệ thống sẽ dùng font mặc định của Pillow.")

    # Lấy cổng mạng (port) từ cấu hình môi trường hoặc đặt mặc định là 8000
    port = int(os.environ.get("PORT", 8000))
    # Mặc định tắt reload khi chạy server để tránh việc ghi ảnh tạm làm khởi động lại server giữa chừng
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    print(f"Đang khởi động máy chủ dịch thuật tại cổng {port} (Reload={reload})...")
    # Khởi chạy server uvicorn trỏ tới ứng dụng app trong thư mục app/main.py
    # Sử dụng reload_dirs=["app"] để uvicorn chỉ theo dõi thay đổi trong thư mục app/ và tránh việc tự động khởi động lại khi ghi tệp tạm vào thư mục data/
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=reload, reload_dirs=["app"] if reload else None)

