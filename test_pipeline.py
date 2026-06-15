"""
Script kiểm thử end-to-end Pipeline dịch truyện tranh
Sử dụng ảnh thật từ data/tests/ và API key Gemini thực tế
"""
import os
import sys
import shutil
import zipfile
import json
import traceback

# Đảm bảo import từ thư mục gốc của dự án
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── CẤU HÌNH TEST ─────────────────────────────────────────────────────────────
TEST_IMAGES_DIR = os.path.join("data", "tests")
API_KEY_FILE    = os.path.join(TEST_IMAGES_DIR, "linkAPIggStudio")
OUTPUT_DIR      = os.path.join("data", "tests", "output_test")
TEMP_DIR        = os.path.join("data", "tests", "temp_test")
# ────────────────────────────────────────────────────────────────────────────────


def read_api_key():
    """Đọc API key từ file."""
    try:
        with open(API_KEY_FILE, "r", encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            print(f"[OK] Đọc API key thành công (ký tự đầu: {key[:8]}...)")
            return key
        else:
            print("[WARN] File API key trống!")
            return ""
    except Exception as e:
        print(f"[WARN] Không đọc được file API key: {e}")
        return ""


def collect_test_images():
    """Thu thập danh sách ảnh test (1.jpg đến 6.jpg)."""
    exts = ('.jpg', '.jpeg', '.png', '.webp')
    images = []
    for f in sorted(os.listdir(TEST_IMAGES_DIR)):
        if f.lower().endswith(exts) and not f.startswith('.'):
            images.append(os.path.join(TEST_IMAGES_DIR, f))
    print(f"[OK] Tìm thấy {len(images)} ảnh test: {[os.path.basename(p) for p in images]}")
    return images


def pack_images_to_zip(image_paths, zip_path):
    """Đóng gói danh sách ảnh thành file ZIP."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in image_paths:
            zf.write(p, os.path.basename(p))
    print(f"[OK] Đã tạo ZIP test: {zip_path}")


def run_pipeline_test(api_key, zip_path, output_zip):
    """Chạy toàn bộ pipeline và ghi log."""
    from app.pipeline import MangaPipeline

    logs = []
    def status_callback(msg, pct, event_type=None, data=None):
        line = f"[{pct:5.1f}%] {msg}"
        logs.append(line)
        print(line)
        if event_type == "ocr_completed" and data:
            print(f"\n══ OCR RAW ({len(data)} items) ══")
            for item in data[:5]:
                print(f"  [{item['id']}] {item['text'][:60]}")
            if len(data) > 5:
                print(f"  ... và {len(data)-5} mục khác")
        elif event_type == "translation_completed" and data:
            print(f"\n══ DỊCH THUẬT ({len(data)} items) ══")
            for k, v in list(data.items())[:5]:
                print(f"  [{k}] → {v[:60]}")
            if len(data) > 5:
                print(f"  ... và {len(data)-5} mục khác")

    pipeline = MangaPipeline(
        api_key=api_key,
        src_lang="ch",          # Truyện Manhua - tiếng Trung
        tone="tự nhiên",
        batch_size_pages=6,     # Dịch tất cả 6 trang cùng 1 batch
        additional_instructions="Đây là truyện Manhua (truyện tranh Trung Quốc). Dịch thành tiếng Việt tự nhiên.",
        status_callback=status_callback,
    )

    os.makedirs(TEMP_DIR, exist_ok=True)
    pipeline.run_pipeline(zip_path, output_zip, TEMP_DIR)
    return logs


def save_result_images(output_zip):
    """Giải nén ảnh kết quả ra thư mục output."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with zipfile.ZipFile(output_zip, 'r') as zf:
        zf.extractall(OUTPUT_DIR)
    result_imgs = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
    print(f"\n[OK] Đã giải nén {len(result_imgs)} ảnh kết quả vào: {OUTPUT_DIR}")
    for img in sorted(result_imgs):
        print(f"  → {img}")
    return result_imgs


def main():
    print("=" * 60)
    print("  KIỂM THỬ END-TO-END PIPELINE DỊCH TRUYỆN TRANH")
    print("=" * 60)

    # 0. Dọn thư mục cũ nếu có
    for d in [OUTPUT_DIR, TEMP_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 1. Đọc API key
    api_key = read_api_key()

    # 2. Thu thập ảnh test
    images = collect_test_images()
    if not images:
        print("[FAIL] Không tìm thấy ảnh test nào trong data/tests/")
        sys.exit(1)

    # 3. Đóng gói ZIP
    zip_path = os.path.join(TEMP_DIR, "test_input.zip")
    pack_images_to_zip(images, zip_path)

    # 4. Đường dẫn output ZIP
    output_zip = os.path.join(TEMP_DIR, "test_output.zip")

    # 5. Chạy pipeline
    print("\n" + "─" * 60)
    print("  BẮT ĐẦU CHẠY PIPELINE...")
    print("─" * 60)
    try:
        logs = run_pipeline_test(api_key, zip_path, output_zip)
    except Exception as e:
        print(f"\n[FAIL] Pipeline gặp lỗi nghiêm trọng: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 6. Lưu kết quả
    print("\n" + "─" * 60)
    print("  KẾT QUẢ")
    print("─" * 60)
    if os.path.exists(output_zip):
        result_imgs = save_result_images(output_zip)
        print(f"\n✅ PIPELINE HOÀN THÀNH THÀNH CÔNG!")
        print(f"   Ảnh kết quả: {OUTPUT_DIR}")
    else:
        print("\n[FAIL] Không tìm thấy file output ZIP. Pipeline thất bại.")
        sys.exit(1)


if __name__ == "__main__":
    main()
