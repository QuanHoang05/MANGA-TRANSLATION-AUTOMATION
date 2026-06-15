"""
Script test nhanh pipeline với ảnh từ data/tests/ và API key từ file linkAPIggStudio
Chạy: python test_local.py
"""
import os
import sys
import shutil

# ── Đọc API key từ file ─────────────────────────────────────────────────────
API_KEY_FILE = os.path.join("data", "tests", "linkAPIggStudio")
api_key = ""
if os.path.exists(API_KEY_FILE):
    with open(API_KEY_FILE, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
    print(f"[OK] Đã đọc API key: {api_key[:10]}...")
else:
    print("[LỖI] Không tìm thấy file API key!")
    sys.exit(1)

# ── Cấu hình test ────────────────────────────────────────────────────────────
TEST_IMG_DIR   = os.path.join("data", "tests")         # Thư mục chứa 1.jpg - 6.jpg
OUTPUT_ZIP     = os.path.join("data", "tests", "output_test", "result_test.zip")
TEMP_DIR       = os.path.join("data", "tests", "temp_test")

os.makedirs(os.path.dirname(OUTPUT_ZIP), exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

print(f"[OK] Thư mục ảnh test: {os.path.abspath(TEST_IMG_DIR)}")
print(f"[OK] Output zip sẽ lưu tại: {os.path.abspath(OUTPUT_ZIP)}")
print(f"[OK] Temp dir: {os.path.abspath(TEMP_DIR)}")

# ── Kiểm tra ảnh test có tồn tại không ──────────────────────────────────────
imgs = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("_")]
imgs_sorted = sorted(imgs)
print(f"[OK] Tìm thấy {len(imgs_sorted)} file ảnh test: {imgs_sorted}")

if not imgs_sorted:
    print("[LỖI] Không tìm thấy ảnh nào trong data/tests!")
    sys.exit(1)

# ── Callback in log ──────────────────────────────────────────────────────────
def status_callback(message, percent, event_type=None, data=None):
    print(f"  [{percent:5.1f}%] {message}")
    if event_type == "ocr_completed" and data:
        print(f"    → OCR: Phát hiện {len(data)} ô thoại")
        for item in data[:5]:  # In 5 cái đầu
            print(f"      [{item['id']}] {item['text'][:60]}")
        if len(data) > 5:
            print(f"      ... và {len(data)-5} ô khác")
    elif event_type == "translation_completed" and data:
        print(f"    → Dịch: Nhận {len(data)} bản dịch")
        for k, v in list(data.items())[:5]:
            print(f"      [{k}] → {v[:60]}")

# ── Chạy pipeline ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BẮT ĐẦU CHẠY PIPELINE TEST")
print("="*60)

# Xóa temp cũ để test sạch
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR, exist_ok=True)

try:
    from app.pipeline import MangaPipeline

    pipeline = MangaPipeline(
        api_key=api_key,
        src_lang="ch",          # Ảnh test thường là tiếng Trung/Nhật
        tone="tự nhiên",
        batch_size_pages=6,     # 6 ảnh → 1 batch
        additional_instructions="",
        status_callback=status_callback,
        custom_weights_path="models/yolov8_comic.pt"  # Sẽ bỏ qua nếu không có
    )

    pipeline.run_pipeline(
        zip_path=TEST_IMG_DIR,        # Truyền thẳng thư mục ảnh (không cần zip)
        output_zip_path=OUTPUT_ZIP,
        temp_dir=TEMP_DIR
    )

    print("\n" + "="*60)
    print("✅ PIPELINE HOÀN THÀNH THÀNH CÔNG!")
    print(f"   Ảnh kết quả: {os.path.join(TEMP_DIR, 'output')}")
    print(f"   File ZIP: {OUTPUT_ZIP}")
    
    # Liệt kê ảnh output
    output_dir = os.path.join(TEMP_DIR, "output")
    if os.path.exists(output_dir):
        out_files = sorted(os.listdir(output_dir))
        print(f"   Số trang đầu ra: {len(out_files)}")
        for f in out_files:
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"     - {f} ({size/1024:.1f} KB)")
    print("="*60)

except ImportError as e:
    print(f"\n[LỖI IMPORT] Thiếu thư viện: {e}")
    print("Hãy chạy: pip install paddleocr ultralytics google-generativeai opencv-python pillow")
except Exception as e:
    import traceback
    print(f"\n[LỖI PIPELINE] {e}")
    traceback.print_exc()
