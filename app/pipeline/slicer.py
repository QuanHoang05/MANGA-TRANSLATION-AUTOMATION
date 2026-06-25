import os
import re
import cv2
import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None

def natural_sort_key(s: str) -> list:
    """Khóa sắp xếp tự nhiên để tránh nhảy thứ tự số (1, 10, 2 → 1, 2, 10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def stitch_and_smart_slice(
    raw_images: list, 
    output_folder: str, 
    log_fn, 
    target_max_height: int = 2000
) -> list:
    """
    Nối dọc toàn bộ ảnh thành 1 dải, sau đó cắt thông minh tại khoảng trắng
    dựa trên phương sai hàng ngang (Row Variance) để PaddleOCR quét hiệu quả.
    """
    log_fn(f"Bắt đầu khâu (stitching) {len(raw_images)} ảnh...", 6.0)

    loaded_images = []
    for path in raw_images:
        try:
            img = Image.open(path)
            loaded_images.append((img, path))
        except Exception as e:
            log_fn(f"Bỏ qua ảnh lỗi: {path} - {e}", 6.0)

    if not loaded_images:
        raise ValueError("Không thể tải được bất kỳ ảnh hợp lệ nào.")

    w_common = loaded_images[0][0].width
    resized_images = []
    total_height = 0

    for img, _ in loaded_images:
        w, h = img.size
        if w != w_common:
            h_new = int(h * (w_common / w))
            img = img.resize((w_common, h_new), Image.Resampling.LANCZOS)
            total_height += h_new
        else:
            total_height += h
        resized_images.append(img)

    log_fn(f"Tổng chiều cao sau khâu: {total_height}px. Chiều rộng chung: {w_common}px", 7.0)

    # Tạo Mega Image
    mega_img = Image.new("RGB", (w_common, total_height), (255, 255, 255))
    current_y = 0
    for img in resized_images:
        mega_img.paste(img, (0, current_y))
        current_y += img.height

    # Phân tích phương sai dòng để tìm điểm cắt tối ưu
    log_fn("Đang phân tích phương sai dòng để tìm điểm cắt thông minh...", 8.0)
    mega_np = np.array(mega_img)
    mega_gray = cv2.cvtColor(mega_np, cv2.COLOR_RGB2GRAY)
    row_variances = np.var(mega_gray, axis=1)

    y = 0
    part_idx = 1
    slice_paths = []
    min_slice_h = int(target_max_height * 0.7)

    while y < total_height:
        if total_height - y <= target_max_height:
            slice_y = total_height
        else:
            search_start = y + min_slice_h
            search_end = min(y + target_max_height, total_height)
            local_variances = row_variances[search_start:search_end]
            best_local_y = int(np.argmin(local_variances))
            slice_y = search_start + best_local_y

        slice_img = mega_img.crop((0, y, w_common, slice_y))

        # Phóng to ảnh lên tối thiểu 1600px để PaddleOCR quét tốt hơn
        w_s, h_s = slice_img.size
        if w_s < 1600:
            ratio = 1600.0 / w_s
            slice_img = slice_img.resize((1600, int(h_s * ratio)), Image.Resampling.LANCZOS)
            slice_img = slice_img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
        else:
            slice_img = slice_img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=3))

        part_name = f"{part_idx:03d}_slice.png"
        part_path = os.path.join(output_folder, part_name)
        slice_img.save(part_path)
        slice_paths.append(part_path)

        log_fn(f"Đã cắt phân mảnh {part_idx}: dòng {y}→{slice_y} (Cao: {slice_y - y}px)", 9.0)
        y = slice_y
        part_idx += 1

    return slice_paths

def stitch_output_images(output_folder: str, temp_dir: str, log_fn):
    """
    Ghép dọc tất cả ảnh kết quả thành 1 ảnh Webtoon duy nhất (JPEG hoặc PNG).
    """
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    image_files = [
        os.path.join(output_folder, f)
        for f in sorted(os.listdir(output_folder))
        if f.lower().endswith(image_extensions) and not f.startswith('._')
    ]

    if not image_files:
        return

    images = [cv2.imread(p) for p in image_files]
    images = [img for img in images if img is not None]

    if not images:
        return

    w_common = images[0].shape[1]
    resized = []
    for img in images:
        h, w = img.shape[:2]
        if w != w_common:
            h_new = int(h * (w_common / w))
            img = cv2.resize(img, (w_common, h_new), interpolation=cv2.INTER_LANCZOS4)
        resized.append(img)

    stitched = np.vstack(resized)
    h_total = stitched.shape[0]

    if h_total > 65535:
        stitched_path = os.path.join(temp_dir, "translated_stitched.png")
        cv2.imwrite(stitched_path, stitched)
        log_fn(f"HỆ THỐNG: Ảnh ghép quá dài ({h_total}px). Đã lưu dạng PNG.", 92.0)
    else:
        stitched_path = os.path.join(temp_dir, "translated_stitched.jpg")
        cv2.imwrite(stitched_path, stitched, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        log_fn(f"HỆ THỐNG: Đã ghép dọc thành công: {os.path.basename(stitched_path)} ({h_total}px)", 92.0)
