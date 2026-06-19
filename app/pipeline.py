import os
import re
import gc
import json
import zipfile
import shutil
import textwrap
import traceback
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

Image.MAX_IMAGE_PIXELS = None

import google.generativeai as genai


class MangaPipeline:
    def __init__(
        self,
        api_key: str,
        src_lang: str = "en",
        tgt_lang: str = "vi",
        tone: str = "tự nhiên",
        batch_size_pages: int = 10,
        additional_instructions: str = "",
        status_callback=None,
        custom_translation: str = ""
    ):
        self.api_key = api_key
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tone = tone
        self.batch_size_pages = batch_size_pages
        self.additional_instructions = additional_instructions
        self.status_callback = status_callback

        # Cấu hình Google Gemini API
        if api_key:
            genai.configure(api_key=api_key)

        # Phân tích bản dịch tùy chỉnh do người dùng cung cấp
        self.custom_translation_map = self._parse_translation_json(custom_translation)

    # ──────────────────────────────────────────────────────────────────────────
    # Tiện ích nội bộ
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_translation_json(self, json_str: str) -> dict:
        """
        Phân tích cú pháp chuỗi JSON bản dịch thành từ điển {id: translated_text}.
        Hỗ trợ cả 2 định dạng:
          - {"id1": "text1", "id2": "text2"}
          - {"translations": [{"id": "id1", "translated_text": "text1"}, ...]}
        """
        if not json_str:
            return {}
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                if "translations" in data and isinstance(data["translations"], list):
                    return {
                        item["id"]: item["translated_text"]
                        for item in data["translations"]
                        if isinstance(item, dict) and "id" in item and "translated_text" in item
                    }
                return data
            if isinstance(data, list):
                return {
                    item["id"]: item["translated_text"]
                    for item in data
                    if isinstance(item, dict) and "id" in item and "translated_text" in item
                }
        except Exception as e:
            print(f"Cảnh báo: Không thể phân tích bản dịch JSON: {e}")
        return {}

    def log(self, message: str, percent: float, event_type: str = None, data=None):
        """Ghi log và gọi callback cập nhật trạng thái."""
        if self.status_callback:
            try:
                return self.status_callback(message, percent, event_type, data)
            except TypeError:
                try:
                    return self.status_callback(message, percent)
                except Exception:
                    pass
        else:
            print(f"[{percent:.1f}%] {message}")
        return None

    @staticmethod
    def _natural_sort_key(s: str) -> list:
        """Khóa sắp xếp tự nhiên để tránh nhảy thứ tự số (1, 10, 2 → 1, 2, 10)."""
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    # ──────────────────────────────────────────────────────────────────────────
    # Bước 1 – Chuẩn bị ảnh đầu vào
    # ──────────────────────────────────────────────────────────────────────────

    def stitch_and_smart_slice(self, raw_images: list, output_folder: str, target_max_height: int = 2000) -> list:
        """
        Nối dọc toàn bộ ảnh thành 1 dải, sau đó cắt thông minh tại khoảng trắng
        dựa trên phương sai hàng ngang (Row Variance) để PaddleOCR quét hiệu quả.
        """
        self.log(f"Bắt đầu khâu (stitching) {len(raw_images)} ảnh...", 6.0)

        loaded_images = []
        for path in raw_images:
            try:
                img = Image.open(path)
                loaded_images.append((img, path))
            except Exception as e:
                self.log(f"Bỏ qua ảnh lỗi: {path} - {e}", 6.0)

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

        self.log(f"Tổng chiều cao sau khâu: {total_height}px. Chiều rộng chung: {w_common}px", 7.0)

        # Tạo Mega Image
        mega_img = Image.new("RGB", (w_common, total_height), (255, 255, 255))
        current_y = 0
        for img in resized_images:
            mega_img.paste(img, (0, current_y))
            current_y += img.height

        # Phân tích phương sai dòng để tìm điểm cắt tối ưu
        self.log("Đang phân tích phương sai dòng để tìm điểm cắt thông minh...", 8.0)
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

            self.log(f"Đã cắt phân mảnh {part_idx}: dòng {y}→{slice_y} (Cao: {slice_y - y}px)", 9.0)
            y = slice_y
            part_idx += 1

        return slice_paths

    # ──────────────────────────────────────────────────────────────────────────
    # Bước 2 – Nhận diện chữ OCR
    # ──────────────────────────────────────────────────────────────────────────

    def _get_ocr_model(self, lang: str):
        """Khởi tạo (hoặc lấy từ cache) đối tượng PaddleOCR cho ngôn ngữ chỉ định."""
        if not hasattr(self, "_ocr_cache"):
            self._ocr_cache = {}
        if lang not in self._ocr_cache:
            from paddleocr import PaddleOCR
            self._ocr_cache[lang] = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                enable_mkldnn=False,
                det_limit_side_len=3000,
                det_limit_type='max',
                drop_score=0.3,
                cpu_threads=2
            )
        return self._ocr_cache[lang]

    def detect_image_language(self, img_path: str) -> str:
        """
        Dùng model OCR 'ch' đọc thử 1 ảnh để nhận dạng tập ký tự và suy ra ngôn ngữ.
        """
        try:
            ocr = self._get_ocr_model("ch")
            try:
                result = ocr.ocr(img_path)
            except TypeError:
                result = ocr.ocr(img_path, cls=True)

            if not result or not result[0]:
                return "en"

            full_text = "".join(line[1][0] for line in result[0])

            if any('\u3040' <= c <= '\u30ff' for c in full_text):
                return "japan"
            if any('\uac00' <= c <= '\ud7a3' for c in full_text):
                return "korean"
            if any('\u4e00' <= c <= '\u9fff' for c in full_text):
                return "ch"
            return "en"
        except Exception as e:
            self.log(f"Lỗi khi dò ngôn ngữ tự động: {e}. Mặc định dùng 'en'.", 15.5)
            return "en"

    def group_ocr_boxes(self, ocr_items: list, lang: str) -> list:
        """
        Gom nhóm các dòng chữ OCR thuộc cùng một bong bóng thoại bằng Union-Find.
        """
        if not ocr_items:
            return []

        n = len(ocr_items)
        parent = list(range(n))

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            for j in range(i + 1, n):
                x0A, y0A, x2A, y2A = ocr_items[i]["bbox"]
                x0B, y0B, x2B, y2B = ocr_items[j]["bbox"]
                h_avg = ((y2A - y0A) + (y2B - y0B)) / 2.0
                v_gap = max(0, y0B - y2A) if y0B >= y2A else max(0, y0A - y2B)
                overlap_x = min(x2A, x2B) - max(x0A, x0B)
                if v_gap <= h_avg * 1.5 and overlap_x > -(h_avg * 0.8):
                    union(i, j)

        groups: dict = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(ocr_items[i])

        is_cjk = lang.lower() in ("ch", "chinese", "jp", "japan", "japanese", "ko", "korean")
        merged_items = []

        for g_idx, group in enumerate(groups.values()):
            group.sort(key=lambda item: item["bbox"][1])
            texts = [item["original_text"] for item in group]
            combined_text = "".join(texts) if is_cjk else " ".join(texts)

            x0 = min(item["bbox"][0] for item in group)
            y0 = min(item["bbox"][1] for item in group)
            x2 = max(item["bbox"][2] for item in group)
            y2 = max(item["bbox"][3] for item in group)

            img_prefix = group[0]["id"].split("-")[0]
            merged_items.append({
                "id": f"{img_prefix}-B{g_idx}",
                "original_text": combined_text,
                "box_points": [item["box_points"] for item in group],
                "bbox": [x0, y0, x2, y2],
                "confidence": sum(item["confidence"] for item in group) / len(group)
            })

        merged_items.sort(key=lambda item: item["bbox"][1])
        return merged_items

    def _run_ocr_on_image(self, ocr_model, img_path: str) -> list:
        """
        Chạy OCR trên một ảnh và trả về danh sách raw detection items đã chuẩn hóa.
        Hỗ trợ cả PaddleOCR 2.x và 3.x+.
        """
        try:
            raw_result = ocr_model.ocr(img_path)
        except TypeError:
            raw_result = ocr_model.ocr(img_path, cls=True)

        if not raw_result or not raw_result[0]:
            return []

        # Chuẩn hóa định dạng PaddleOCR 3.x+ (dict với rec_texts)
        if isinstance(raw_result[0], dict) and 'rec_texts' in raw_result[0]:
            normalized = []
            for page_data in raw_result:
                rec_texts = page_data.get('rec_texts', [])
                rec_scores = page_data.get('rec_scores', [])
                rec_polys = page_data.get('rec_polys', [])
                for i in range(len(rec_texts)):
                    box = rec_polys[i] if i < len(rec_polys) else []
                    if hasattr(box, 'tolist'):
                        box = box.tolist()
                    score = rec_scores[i] if i < len(rec_scores) else 1.0
                    normalized.append([box, (rec_texts[i], score)])
            return normalized

        return raw_result[0] or []

    # ──────────────────────────────────────────────────────────────────────────
    # Bước 3 – Dịch thuật
    # ──────────────────────────────────────────────────────────────────────────

    def translate_batch(self, items: list) -> dict:
        """
        Gửi cụm thoại lên Gemini API và nhận về từ điển {id: bản_dịch}.
        """
        lang_display = {
            "ko": "tiếng Hàn", "korean": "tiếng Hàn",
            "japan": "tiếng Nhật", "jp": "tiếng Nhật",
            "ch": "tiếng Trung", "zh": "tiếng Trung",
            "en": "tiếng Anh", "english": "tiếng Anh"
        }
        tgt_display = {
            "vi": "tiếng Việt", "vietnamese": "tiếng Việt",
            "en": "tiếng Anh", "english": "tiếng Anh",
            "ch": "tiếng Trung", "chinese": "tiếng Trung",
            "japan": "tiếng Nhật", "japanese": "tiếng Nhật",
            "ko": "tiếng Hàn", "korean": "tiếng Hàn"
        }

        src_name = lang_display.get(self.src_lang.lower(), self.src_lang)
        tgt_name = tgt_display.get(self.tgt_lang.lower(), self.tgt_lang)

        prompt = f"""Bạn là một dịch giả truyện tranh (manga/webtoon) chuyên nghiệp. Hãy dịch các câu thoại từ {src_name} sang {tgt_name}.

HƯỚNG DẪN XƯNG HÔ & PHONG CÁCH:
- Đảm bảo xưng hô đồng bộ, tự nhiên và phù hợp với ngữ cảnh câu chuyện.
- Tông giọng dịch yêu cầu: {self.tone}.
- Dựa vào mạch truyện của các câu thoại liên tiếp để suy luận mối quan hệ nhân vật chính xác.
"""
        if self.additional_instructions:
            prompt += f"\nYÊU CẦU ĐẶC BIỆT TỪ NGƯỜI DÙNG:\n- {self.additional_instructions}\n"

        prompt += f"""
DỮ LIỆU ĐẦU VÀO (JSON):
{json.dumps({"items": items}, ensure_ascii=False, indent=2)}

Hãy dịch toàn bộ danh sách và trả về kết quả JSON:
{{
  "translations": [
    {{"id": "ID gốc", "translated_text": "nội dung đã dịch"}}
  ]
}}
"""
        schema = {
            "type": "OBJECT",
            "properties": {
                "translations": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "STRING"},
                            "translated_text": {"type": "STRING"}
                        },
                        "required": ["id", "translated_text"]
                    }
                }
            },
            "required": ["translations"]
        }

        # Lấy danh sách model flash khả dụng, ưu tiên model mới nhất
        model_names = []
        try:
            supported = list(genai.list_models())
            flash_models = [
                m.name.replace("models/", "")
                for m in supported
                if "flash" in m.name.lower()
                and "generatecontent" in "".join(m.supported_generation_methods).lower()
            ]
            flash_models.sort(reverse=True)
            model_names.extend(flash_models)
        except Exception as e:
            print(f"Cảnh báo: Không thể liệt kê model ({e})")

        for fallback in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash"]:
            if fallback not in model_names:
                model_names.append(fallback)

        last_err = None
        for m_name in model_names:
            try:
                print(f"Đang thử dịch bằng model: {m_name}...")
                model = genai.GenerativeModel(
                    model_name=m_name,
                    generation_config={
                        "response_mime_type": "application/json",
                        "response_schema": schema
                    }
                )
                response = model.generate_content(prompt)
                response_data = json.loads(response.text)
                print(f"Dịch thành công bằng model: {m_name}!")
                return {
                    item["id"]: item["translated_text"]
                    for item in response_data.get("translations", [])
                }
            except Exception as e:
                last_err = e
                print(f"Model {m_name} lỗi: {str(e)}")
                continue

        print("Tất cả các model thử nghiệm đều thất bại.")
        if last_err:
            raise last_err
        raise Exception("Không thể khởi tạo dịch thuật Gemini.")

    # ──────────────────────────────────────────────────────────────────────────
    # Bước 4 – Xử lý đồ họa (Inpainting & Typesetting)
    # ──────────────────────────────────────────────────────────────────────────

    def apply_adaptive_padding(self, bbox: list, img_w: int, img_h: int) -> list:
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

    def max_inscribed_rectangle(self, mask_binary) -> tuple:
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

    def find_bubble_contour_and_rect(self, cv2_img, bbox: list):
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

        inscribed = self.max_inscribed_rectangle(eroded > 0)
        if inscribed[2] > inscribed[0] and inscribed[3] > inscribed[1]:
            best_rect = [
                inscribed[0] + xmin,
                inscribed[1] + ymin,
                inscribed[2] + xmin,
                inscribed[3] + ymin
            ]
        else:
            best_rect = [x0, y0, x2, y2]

        return True, best_contour_abs, best_rect, bg_color

    # ──────────────────────────────────────────────────────────────────────────
    # Vẽ chữ
    # ──────────────────────────────────────────────────────────────────────────

    def _wrap_text(self, text: str, font, max_width: float) -> list:
        """Ngắt dòng văn bản để vừa chiều rộng bbox."""
        try:
            bbox = font.getbbox("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            avg_char_w = (bbox[2] - bbox[0]) / 62
        except Exception:
            avg_char_w = 10.0
        avg_char_w = max(avg_char_w, 1.0)
        chars_per_line = max(1, int(max_width / avg_char_w))
        return textwrap.wrap(text, width=chars_per_line)

    def draw_text_in_box(self, draw: ImageDraw.Draw, text: str, bbox: list, font_path: str, is_sfx: bool = False):
        """
        Tự động chọn cỡ font tối ưu, ngắt dòng và vẽ văn bản căn giữa trong bbox.
        """
        x0, y0, x2, y2 = bbox
        box_w, box_h = x2 - x0, y2 - y0

        optimal_font = None
        optimal_lines = []
        optimal_line_heights = []

        for font_size in range(38, 5, -2):
            try:
                font = ImageFont.load_default() if font_path == "Arial" else ImageFont.truetype(font_path, font_size)
            except Exception:
                font = ImageFont.load_default()

            lines = self._wrap_text(text, font, box_w)
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
            optimal_lines = self._wrap_text(text, optimal_font, box_w)
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

    # ──────────────────────────────────────────────────────────────────────────
    # Bước 5 – Ghép ảnh đầu ra
    # ──────────────────────────────────────────────────────────────────────────

    def stitch_output_images(self, output_folder: str, temp_dir: str):
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
            self.log(f"HỆ THỐNG: Ảnh ghép quá dài ({h_total}px). Đã lưu dạng PNG.", 92.0)
        else:
            stitched_path = os.path.join(temp_dir, "translated_stitched.jpg")
            cv2.imwrite(stitched_path, stitched, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            self.log(f"HỆ THỐNG: Đã ghép dọc thành công: {os.path.basename(stitched_path)} ({h_total}px)", 92.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline chính
    # ──────────────────────────────────────────────────────────────────────────

    def run_pipeline(self, zip_path: str, output_zip_path: str, temp_dir: str):
        """
        Thực thi quy trình 5 bước tự động dịch truyện tranh:
        1. Chuẩn bị ảnh đầu vào
        2. Nhận diện chữ OCR
        3. Dịch thuật (Gemini API hoặc bản dịch tùy chỉnh)
        4. Xóa chữ cũ & Vẽ chữ mới (Inpainting + Typesetting)
        5. Ghép ảnh & Đóng gói ZIP
        """
        # ── Bước 1: Chuẩn bị ảnh ──
        self.log("BƯỚC 1: Thu thập và chuẩn bị ảnh đầu vào...", 5.0)
        raw_extract_folder = os.path.join(temp_dir, "raw_extract")
        input_folder = os.path.join(temp_dir, "input")
        output_folder = os.path.join(temp_dir, "output")
        for folder in (raw_extract_folder, input_folder, output_folder):
            os.makedirs(folder, exist_ok=True)

        if os.path.isdir(zip_path):
            for file in os.listdir(zip_path):
                src = os.path.join(zip_path, file)
                if os.path.isfile(src):
                    shutil.copy(src, os.path.join(raw_extract_folder, file))
        else:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(raw_extract_folder)

        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        raw_images = []
        for root, _, files in os.walk(raw_extract_folder):
            for file in files:
                if file.lower().endswith(image_extensions) and not file.startswith('._'):
                    raw_images.append(os.path.join(root, file))

        if not raw_images:
            raise ValueError("Không tìm thấy ảnh hợp lệ nào trong dữ liệu tải lên.")

        raw_images.sort(key=self._natural_sort_key)
        images = self.stitch_and_smart_slice(raw_images, input_folder)
        total_images = len(images)
        self.log(f"Đã khâu và cắt thông minh thành {total_images} trang ảnh.", 10.0)

        # ── Bước 2: OCR ──
        self.log("BƯỚC 2: Đang tải mô hình nhận diện chữ (PaddleOCR)...", 15.0)

        if self.src_lang.lower() == "auto":
            self.log("Đang dò ngôn ngữ tự động từ ảnh đầu tiên...", 15.5)
            self.src_lang = self.detect_image_language(images[0])
            self.log(f"Đã nhận diện ngôn ngữ: {self.src_lang}", 16.0)

        lang_map = {"en": "en", "japan": "japan", "korean": "korean", "ch": "ch"}
        ocr_lang = lang_map.get(self.src_lang, "en")
        ocr_model = self._get_ocr_model(ocr_lang)

        ocr_data = {}  # {img_path: [OCR item, ...]}

        for idx, img_path in enumerate(images):
            img_name = os.path.basename(img_path)
            pct = 15.0 + (idx / total_images) * 25.0
            self.log(f"Đang quét OCR ảnh {idx + 1}/{total_images}: {img_name}...", pct)

            try:
                detections = self._run_ocr_on_image(ocr_model, img_path)
                img_ocr_items = []

                for bubble_idx, det in enumerate(detections):
                    box = det[0]
                    text, conf = det[1][0], det[1][1]
                    xs = [pt[0] for pt in box]
                    ys = [pt[1] for pt in box]
                    img_ocr_items.append({
                        "id": f"{img_name.split('.')[0]}-O{bubble_idx}",
                        "original_text": text,
                        "box_points": box,
                        "bbox": [min(xs), min(ys), max(xs), max(ys)],
                        "confidence": float(conf)
                    })

                ocr_data[img_path] = self.group_ocr_boxes(img_ocr_items, self.src_lang)

            except Exception as e:
                self.log(f"Lỗi khi quét OCR ảnh {img_name}: {str(e)}", pct)
                ocr_data[img_path] = []
            finally:
                gc.collect()

        # Xuất kết quả OCR về client
        ocr_results_list = [
            {"id": item["id"], "text": item["original_text"]}
            for img_path in images
            for item in ocr_data.get(img_path, [])
        ]
        self.log("Đã hoàn thành quét OCR toàn bộ các trang.", 45.0,
                 event_type="ocr_completed", data=ocr_results_list)

        # ── Bước 3: Dịch thuật ──
        self.log("BƯỚC 3: Dịch thuật gộp qua API Gemini Studio...", 45.0)
        all_ocr_items = [item for img_path in images for item in ocr_data[img_path]]
        translated_texts = {}

        if self.custom_translation_map:
            # Người dùng đã cung cấp bản dịch sẵn
            self.log("HỆ THỐNG: Sử dụng bản dịch JSON tùy chỉnh do người dùng cung cấp.", 48.0)
            for item in all_ocr_items:
                if item["id"] in self.custom_translation_map:
                    translated_texts[item["id"]] = self.custom_translation_map[item["id"]]
            self.log("Đã áp dụng bản dịch tùy chỉnh.", 70.0,
                     event_type="translation_completed", data=translated_texts)

        elif not self.api_key:
            # Không có API key và không có bản dịch → tạm dừng chờ người dùng nhập
            user_translation_json = self.log(
                "HỆ THỐNG: Không có API Key. Tạm dừng tiến trình để chờ bạn cung cấp bản dịch JSON...",
                45.0,
                event_type="paused"
            )
            if user_translation_json:
                self.custom_translation_map = self._parse_translation_json(user_translation_json)
                for item in all_ocr_items:
                    if item["id"] in self.custom_translation_map:
                        translated_texts[item["id"]] = self.custom_translation_map[item["id"]]
            self.log("Đã áp dụng bản dịch từ người dùng.", 70.0,
                     event_type="translation_completed", data=translated_texts)

        elif all_ocr_items:
            # Có API key → gửi lên Gemini dịch theo batch
            page_clusters = []
            current_cluster = []
            for i, img_path in enumerate(images):
                current_cluster.append(img_path)
                if len(current_cluster) >= self.batch_size_pages or i == len(images) - 1:
                    page_clusters.append(current_cluster)
                    current_cluster = []

            total_clusters = len(page_clusters)
            self.log(f"Tổng {len(all_ocr_items)} ô thoại. Chia làm {total_clusters} nhóm trang gửi Gemini...", 48.0)

            for c_idx, cluster in enumerate(page_clusters):
                pct = 48.0 + (c_idx / total_clusters) * 22.0
                self.log(f"Đang gửi nhóm trang {c_idx + 1}/{total_clusters} lên Gemini...", pct)

                cluster_items = [
                    {"id": item["id"], "text": item["original_text"]}
                    for img_path in cluster
                    for item in ocr_data[img_path]
                ]
                if cluster_items:
                    translated_texts.update(self.translate_batch(cluster_items))

            self.log("Đã nhận bản dịch từ Gemini.", 70.0,
                     event_type="translation_completed", data=translated_texts)
        else:
            self.log("Không phát hiện văn bản nào cần dịch.", 70.0,
                     event_type="translation_completed", data={})

        # ── Bước 4: Inpainting & Typesetting ──
        self.log("BƯỚC 4: Xóa chữ cũ (Inpainting) & Ráp chữ mới (Typesetting)...", 70.0)

        font_path = os.path.join("fonts", "Nunito-Bold.ttf")
        if not os.path.exists(font_path):
            font_path = "../fonts/Nunito-Bold.ttf"
            if not os.path.exists(font_path):
                font_path = "Arial"

        for idx, img_path in enumerate(images):
            img_name = os.path.basename(img_path)
            pct = 70.0 + (idx / total_images) * 20.0
            self.log(f"Đang xử lý đồ họa ảnh {idx + 1}/{total_images}: {img_name}...", pct)

            try:
                cv2_img = cv2.imread(img_path)
                if cv2_img is None:
                    self.log(f"Lỗi: Không đọc được ảnh {img_name}", pct)
                    shutil.copy(img_path, os.path.join(output_folder, img_name))
                    continue

                items = ocr_data[img_path]
                typeset_info = {}

                if items:
                    sfx_mask = np.zeros(cv2_img.shape[:2], dtype=np.uint8)
                    has_sfx = False
                    H_img, W_img = cv2_img.shape[:2]

                    for item in items:
                        t_text = translated_texts.get(item["id"])
                        # Chỉ xử lý những ô thoại có bản dịch khác với văn bản gốc
                        if not t_text or t_text == item["original_text"]:
                            continue

                        is_bubble, _, best_rect, bg_color = self.find_bubble_contour_and_rect(cv2_img, item["bbox"])
                        padded_rect = self.apply_adaptive_padding(item["bbox"], W_img, H_img)

                        if is_bubble:
                            # Tô đè màu nền đơn sắc của bong bóng thoại
                            sub_mask = np.zeros(cv2_img.shape[:2], dtype=np.uint8)
                            for poly in item["box_points"]:
                                pts = np.array(poly, dtype=np.int32)
                                cv2.fillPoly(sub_mask, [pts], 255)
                            sub_mask = cv2.dilate(sub_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
                            cv2_img[sub_mask == 255] = bg_color
                            typeset_info[item["id"]] = {"bbox": padded_rect, "is_sfx": False}
                        else:
                            # Chữ SFX lơ lửng → Inpaint nâng cao
                            for poly in item["box_points"]:
                                pts = np.array(poly, dtype=np.int32)
                                cv2.fillPoly(sfx_mask, [pts], 255)
                            has_sfx = True
                            typeset_info[item["id"]] = {"bbox": padded_rect, "is_sfx": True}

                    if has_sfx:
                        sfx_mask = cv2.dilate(sfx_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
                        lama = self._get_lama_instance()

                        if lama is not None:
                            # Inpaint cục bộ từng vùng để tiết kiệm RAM và tăng tốc
                            contours, _ = cv2.findContours(sfx_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in contours:
                                x, y, w, h = cv2.boundingRect(cnt)
                                pad = 15
                                xmin, ymin = max(0, x - pad), max(0, y - pad)
                                xmax = min(cv2_img.shape[1], x + w + pad)
                                ymax = min(cv2_img.shape[0], y + h + pad)

                                if (xmax - xmin) > 0 and (ymax - ymin) > 0:
                                    crop_img = cv2_img[ymin:ymax, xmin:xmax]
                                    crop_mask = sfx_mask[ymin:ymax, xmin:xmax]
                                    pil_c = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                                    pil_m = Image.fromarray(crop_mask)
                                    inpainted = lama(pil_c, pil_m)
                                    result_cv = cv2.cvtColor(np.array(inpainted), cv2.COLOR_RGB2BGR)
                                    h_c, w_c = crop_img.shape[:2]
                                    if result_cv.shape[:2] != (h_c, w_c):
                                        result_cv = cv2.resize(result_cv, (w_c, h_c), interpolation=cv2.INTER_LANCZOS4)
                                    cv2_img[ymin:ymax, xmin:xmax] = result_cv
                        else:
                            cv2_img = cv2.inpaint(cv2_img, sfx_mask, inpaintRadius=8, flags=cv2.INPAINT_TELEA)

                # Chuyển sang PIL để vẽ chữ tiếng Việt
                pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)

                for item in items:
                    t_text = translated_texts.get(item["id"])
                    if t_text and t_text != item["original_text"]:
                        info = typeset_info.get(item["id"], {"bbox": item["bbox"], "is_sfx": False})
                        self.draw_text_in_box(draw, t_text, info["bbox"], font_path, is_sfx=info["is_sfx"])

                # Làm nét ảnh nếu quá nhỏ
                w_img, h_img = pil_img.size
                if w_img < 1200:
                    new_w = 1200
                    new_h = int(h_img * (1200 / w_img))
                    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    pil_img = pil_img.filter(ImageFilter.SHARPEN)

                pil_img.save(os.path.join(output_folder, img_name))

            except Exception as e:
                self.log(f"Lỗi khi vẽ chữ trên ảnh {img_name}: {str(e)}", pct)
                traceback.print_exc()
                shutil.copy(img_path, os.path.join(output_folder, img_name))
            finally:
                cv2_img = None
                pil_img = None
                gc.collect()

        # ── Bước 5: Ghép & Đóng gói ──
        self.log("BƯỚC 5: Nén file kết quả và chuẩn bị tải về...", 90.0)
        self.stitch_output_images(output_folder, temp_dir)

        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(output_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    zf.write(file_path, os.path.relpath(file_path, output_folder))

        self.log("HỆ THỐNG: Đã hoàn thành toàn bộ quy trình dịch truyện thành công!", 100.0)

    def _get_lama_instance(self):
        """Lấy instance SimpleLama (cache) hoặc None nếu không khả dụng."""
        if hasattr(self, "_lama_instance"):
            return self._lama_instance
        try:
            from simple_lama_inpainting import SimpleLama
            import torch
            if not hasattr(torch.jit, "_patched_for_lama"):
                torch.set_num_threads(2)
                torch.set_num_interop_threads(2)
                orig = torch.jit.load

                def patched(f, map_location=None, *args, **kwargs):
                    return orig(f, map_location="cpu", *args, **kwargs)

                torch.jit.load = patched
                torch.jit._patched_for_lama = True
            self.log("Đang khởi tạo mô hình AI LaMa-ONNX cho SFX...", 72.0)
            self._lama_instance = SimpleLama("cpu")
            return self._lama_instance
        except Exception as le:
            print(f"Cảnh báo: Không dùng được LaMa ({le}). Sử dụng OpenCV Inpaint.")
            self._lama_instance = None
            return None
