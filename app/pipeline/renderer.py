import cv2
import numpy as np
import textwrap
from PIL import ImageFont, ImageDraw

def apply_adaptive_padding(bbox: list, img_w: int, img_h: int) -> list:
    """Cộng thêm padding cho bbox nhỏ để tránh tràn chữ."""
    x0, y0, x2, y2 = bbox
    w, h = x2 - x0, y2 - y0
    if w < 250 or h < 120:
        pad_w = max(4, min(25, int(w * 0.12)))
        pad_h = max(4, min(25, int(h * 0.12)))
        x0 = max(0, x0 - pad_w)
        y0 = max(0, y0 - pad_h)
        x2 = min(img_w, x2 + pad_w)
        y2 = min(img_h, y2 + pad_h)
    return [int(x0), int(y0), int(x2), int(y2)]

def max_inscribed_rectangle(mask_binary) -> tuple:
    """
    Tìm hình chữ nhật có diện tích lớn nhất nằm trong mặt nạ nhị phân.
    Thuật toán quy hoạch động O(H × W).
    """
    H, W = mask_binary.shape
    heights = np.zeros(W, dtype=np.int32)
    max_area = 0
    best_rect = (0, 0, 0, 0)

    for r in range(H):
        heights = np.where(mask_binary[r], heights + 1, 0)
        stack = []
        for c in range(W + 1):
            h = heights[c] if c < W else 0
            while stack and heights[stack[-1]] >= h:
                curr_h = heights[stack.pop()]
                width = c if not stack else c - stack[-1] - 1
                area = curr_h * width
                if area > max_area:
                    max_area = area
                    start_c = stack[-1] + 1 if stack else 0
                    best_rect = (start_c, r - curr_h + 1, c - 1, r)
            stack.append(c)

    return best_rect

def find_bubble_contour_and_rect(cv2_img, bbox: list):
    """
    Dò ranh giới bong bóng thoại bằng floodFill từ tâm bbox,
    tính hình chữ nhật nội tiếp lớn nhất để vẽ chữ.
    """
    x0, y0, x2, y2 = map(int, bbox)
    box_w, box_h = x2 - x0, y2 - y0
    H_img, W_img = cv2_img.shape[:2]

    pad_x = int(box_w * 0.3) + 15
    pad_y = int(box_h * 0.3) + 15
    xmin = max(0, x0 - pad_x)
    xmax = min(W_img, x2 + pad_x)
    ymin = max(0, y0 - pad_y)
    ymax = min(H_img, y2 + pad_y)

    crop = cv2_img[ymin:ymax, xmin:xmax].copy()
    crop_h, crop_w = crop.shape[:2]

    if crop_h <= 0 or crop_w <= 0:
        return False, None, [x0, y0, x2, y2], (255, 255, 255)

    seed_x = max(0, min(crop_w - 1, int((x0 + x2) / 2 - xmin)))
    seed_y = max(0, min(crop_h - 1, int((y0 + y2) / 2 - ymin)))

    # Chọn điểm sáng nếu pixel trung tâm quá tối (chữ)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray_crop[seed_y, seed_x] < 80:
        ny0, ny1 = max(0, seed_y - 15), min(crop_h, seed_y + 15)
        nx0, nx1 = max(0, seed_x - 15), min(crop_w, seed_x + 15)
        sub = gray_crop[ny0:ny1, nx0:nx1]
        if sub.size > 0:
            _, _, _, max_loc = cv2.minMaxLoc(sub)
            seed_x = nx0 + max_loc[0]
            seed_y = ny0 + max_loc[1]

    ff_mask = np.zeros((crop_h + 2, crop_w + 2), dtype=np.uint8)
    cv2.floodFill(
        image=crop,
        mask=ff_mask,
        seedPoint=(seed_x, seed_y),
        newVal=(255, 255, 255),
        loDiff=(5, 5, 5),
        upDiff=(5, 5, 5),
        flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    )
    bubble_mask = ff_mask[1:-1, 1:-1]
    bubble_pixels = int(np.sum(bubble_mask == 255))

    if bubble_pixels < box_w * box_h * 0.35:
        return False, None, [x0, y0, x2, y2], (255, 255, 255)

    contours, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, [x0, y0, x2, y2], (255, 255, 255)

    bubble_cnt = None
    max_cnt_area = 0
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, (seed_x, seed_y), False) >= 0:
            a = cv2.contourArea(cnt)
            if a > max_cnt_area:
                max_cnt_area = a
                bubble_cnt = cnt

    if bubble_cnt is None:
        return False, None, [x0, y0, x2, y2], (255, 255, 255)

    best_contour_abs = bubble_cnt + np.array([xmin, ymin])

    # Màu nền trung vị của bong bóng
    bg_pixels = crop[bubble_mask == 255]
    bg_color = (255, 255, 255)
    if bg_pixels.size > 0:
        m = np.median(bg_pixels, axis=0)
        bg_color = (int(m[0]), int(m[1]), int(m[2]))

    # Xói mòn mask để chữ không dính viền
    k_size = max(3, int(min(box_w, box_h) * 0.08))
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    eroded = cv2.erode(bubble_mask, kernel, iterations=1)

    # Lấy bounding box của eroded mask (đại diện cho ranh giới an toàn của bong bóng)
    ys, xs = np.where(eroded > 0)
    if xs.size > 0 and ys.size > 0:
        b_x0 = xs.min() + xmin
        b_y0 = ys.min() + ymin
        b_x2 = xs.max() + xmin
        b_y2 = ys.max() + ymin
    else:
        b_x0, b_y0, b_x2, b_y2 = xmin, ymin, xmax, ymax

    # Mở rộng OCR box 35% chiều rộng và 20% chiều cao để tạo không gian chứa chữ dịch
    exp_w = int(box_w * 0.35)
    exp_h = int(box_h * 0.20)
    
    x0_exp = x0 - exp_w
    x2_exp = x2 + exp_w
    y0_exp = y0 - exp_h
    y2_exp = y2 + exp_h

    # Clip (cắt tỉa) hộp mở rộng theo biên an toàn của bong bóng
    x0_new = max(x0_exp, b_x0)
    y0_new = max(y0_exp, b_y0)
    x2_new = min(x2_exp, b_x2)
    y2_new = min(y2_exp, b_y2)

    # Đảm bảo box hợp lệ (không bị đảo ngược tọa độ)
    if x2_new > x0_new and y2_new > y0_new:
        best_rect = [x0_new, y0_new, x2_new, y2_new]
    else:
        best_rect = [x0, y0, x2, y2]

    return True, best_contour_abs, best_rect, bg_color

def wrap_text(text: str, font, max_width: float) -> list:
    """Ngắt dòng văn bản để vừa chiều rộng bbox."""
    try:
        bbox = font.getbbox("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        avg_char_w = (bbox[2] - bbox[0]) / 62
    except Exception:
        avg_char_w = 10.0
    avg_char_w = max(avg_char_w, 1.0)
    chars_per_line = max(1, int(max_width / avg_char_w))
    return textwrap.wrap(text, width=chars_per_line)

def draw_text_in_box(draw: ImageDraw.Draw, text: str, bbox: list, font_path: str, is_sfx: bool = False):
    """
    Tự động chọn cỡ font tối ưu, ngắt dòng và vẽ văn bản căn giữa trong bbox.
    """
    x0, y0, x2, y2 = bbox
    box_w, box_h = x2 - x0, y2 - y0

    optimal_font = None
    optimal_lines = []
    optimal_line_heights = []

    # Tự động điều chỉnh kích cỡ font bắt đầu dựa trên chiều cao của box (tối thiểu 38, tối đa 85)
    start_font_size = max(38, min(85, int(box_h * 0.8)))
    for font_size in range(start_font_size, 5, -2):
        try:
            font = ImageFont.load_default() if font_path == "Arial" else ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        lines = wrap_text(text, font, box_w)
        line_heights = []
        max_line_w = 0

        for line in lines:
            lb = font.getbbox(line)
            line_heights.append(lb[3] - lb[1])
            max_line_w = max(max_line_w, lb[2] - lb[0])

        spacing = 4
        total_h = sum(line_heights) + spacing * max(0, len(lines) - 1)

        if max_line_w <= box_w and total_h <= box_h:
            optimal_font = font
            optimal_lines = lines
            optimal_line_heights = line_heights
            break

    # Fallback: dùng cỡ font tối thiểu 6
    if optimal_font is None:
        try:
            optimal_font = ImageFont.load_default() if font_path == "Arial" else ImageFont.truetype(font_path, 6)
        except Exception:
            optimal_font = ImageFont.load_default()
        optimal_lines = wrap_text(text, optimal_font, box_w)
        optimal_line_heights = [optimal_font.getbbox(l)[3] - optimal_font.getbbox(l)[1] for l in optimal_lines]

    spacing = 4
    total_h = sum(optimal_line_heights) + spacing * max(0, len(optimal_lines) - 1)
    current_y = y0 + (box_h - total_h) / 2

    for i, line in enumerate(optimal_lines):
        lb = optimal_font.getbbox(line)
        line_w = lb[2] - lb[0]
        current_x = x0 + (box_w - line_w) / 2

        if is_sfx:
            draw.text((current_x, current_y), line, fill=(255, 255, 255), font=optimal_font,
                      stroke_width=4, stroke_fill=(0, 0, 0))
        else:
            draw.text((current_x, current_y), line, fill=(0, 0, 0), font=optimal_font,
                      stroke_width=2, stroke_fill=(255, 255, 255))

        current_y += optimal_line_heights[i] + spacing
