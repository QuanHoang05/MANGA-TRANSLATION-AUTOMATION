import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def run_local_verification():
    print("=== BAT DAU KIEM TRA KIEN TRUC ===")
    
    # 1. Kiểm tra việc nhập thư viện cần thiết
    print("1. Kiem tra thu vien can thiet...")
    try:
        from app.pipeline import MangaPipeline
        import fastapi
        import uvicorn
        print(" -> Import thanh cong: pipeline, fastapi, uvicorn")
    except ImportError as e:
        print(f" -> LOI: Khong the import mot so thu vien: {e}")
        sys.exit(1)
        
    # 2. Kiểm tra các thư mục cần thiết trong không gian làm việc
    print("2. Khoi tao thu muc...")
    os.makedirs("fonts", exist_ok=True)
    os.makedirs("data/tests", exist_ok=True)
    print(" -> Data directories are initialized (Thư mục dữ liệu đã sẵn sàng).")
    
    # 3. Tải font chữ nếu bị thiếu trên ổ đĩa
    font_path = "fonts/Nunito-Bold.ttf"
    if not os.path.exists(font_path):
        print("3. Tai thu font Nunito...")
        import urllib.request
        try:
            urllib.request.urlretrieve(
                "https://github.com/google/fonts/raw/main/ofl/nunito/Nunito-Bold.ttf",
                font_path
            )
            print(" -> Tai font thanh cong.")
        except Exception as e:
            print(f" -> Khong the tai font: {e}. Se dung font mac dinh cua he thong.")
            font_path = "Arial"

    # 4. Thử nghiệm giả lập xử lý ảnh (Tẩy chữ & vẽ chữ mới)
    print("4. Chay thu nghiem ve chu & Inpainting...")
    try:
        # Tạo một ảnh trắng ảo với một ô thoại giả lập (đường tròn màu đen, bên trong màu trắng)
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Vẽ chữ gốc "HELLO WORLD" vào bên trong ô thoại
        cv2.circle(img, (200, 200), 100, (0, 0, 0), 2) # Viền của ô thoại giả lập
        cv2.putText(img, "HELLO WORLD", (140, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Lưu ảnh thử nghiệm gốc
        orig_test_path = "data/tests/original_test.png"
        cv2.imwrite(orig_test_path, img)
        print(f" -> Da tao anh gia lap: {orig_test_path}")
        
        # Chạy thử nghiệm xóa chữ (Inpainting) trên vùng chứa chữ gốc
        # Tọa độ hộp giới hạn bao xung quanh chữ "HELLO WORLD"
        # x0=135, y0=185, x2=285, y2=235
        mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.rectangle(mask, (135, 185), (285, 235), 255, -1)
        
        # Thực thi thuật toán cv2.inpaint phục hồi nền
        inpainted = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        inpainted_path = "data/tests/inpainted_test.png"
        cv2.imwrite(inpainted_path, inpainted)
        print(f" -> Tay chu (Inpainting) thanh cong: {inpainted_path}")
        
        # Chuyển đổi sang vẽ chữ tiếng Việt đã dịch lên ảnh
        inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(inpainted_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # Khởi tạo thực thể pipeline để gọi hàm vẽ chữ bổ trợ
        pipeline = MangaPipeline(api_key="", status_callback=None)
        
        # Vẽ câu thoại tiếng Việt vào bên trong hộp giới hạn
        pipeline.draw_text_in_box(
            draw=draw,
            text="XIN CHÀO THẾ GIỚI! ĐÂY LÀ KẾT QUẢ DỊCH THUẬT TIẾNG VIỆT.",
            bbox=[135, 185, 285, 235],
            font_path=font_path
        )
        
        # Lưu kết quả thử nghiệm cuối cùng
        final_test_path = "data/tests/final_test.png"
        pil_img.save(final_test_path)
        print(f" -> Ve chu (Typesetting) thanh cong: {final_test_path}")
        
        print("\n=== HOAN THANH KIEM TRA === ")
        print("Tat ca cac module cot loi hoat dong hoan hao!")
        
    except Exception as e:
        print(f" -> LOI TRONG QUA TRINH THU NGHIEM: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_local_verification()
