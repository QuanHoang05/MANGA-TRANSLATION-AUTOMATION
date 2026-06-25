import os
import gc
import json
import zipfile
import shutil
import traceback
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

import google.generativeai as genai

# Import các hàm mô-đun hóa
from app.pipeline.slicer import natural_sort_key, stitch_and_smart_slice, stitch_output_images
from app.pipeline.ocr import get_ocr_model, detect_image_language, run_ocr_on_image
from app.pipeline.grouper import group_ocr_boxes
from app.pipeline.translator import parse_translation_json, translate_batch
from app.pipeline.renderer import apply_adaptive_padding, find_bubble_contour_and_rect, draw_text_in_box

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
        custom_translation: str = "",
        det_db_unclip_ratio: float = 1.6,
        det_db_box_thresh: float = 0.6
    ):
        self.api_key = api_key
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tone = tone
        self.batch_size_pages = batch_size_pages
        self.additional_instructions = additional_instructions
        self.status_callback = status_callback
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.det_db_box_thresh = det_db_box_thresh

        # Cấu hình Google Gemini API
        if api_key:
            genai.configure(api_key=api_key)

        # Phân tích bản dịch tùy chỉnh do người dùng cung cấp
        self.custom_translation_map = parse_translation_json(custom_translation)

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

        raw_images.sort(key=natural_sort_key)
        images = stitch_and_smart_slice(raw_images, input_folder, self.log)
        total_images = len(images)
        self.log(f"Đã khâu và cắt thông minh thành {total_images} trang ảnh.", 10.0)

        # ── Bước 2: OCR ──
        self.log("BƯỚC 2: Đang tải mô hình nhận diện chữ (PaddleOCR)...", 15.0)

        if self.src_lang.lower() == "auto":
            self.log("Đang dò ngôn ngữ tự động từ ảnh đầu tiên...", 15.5)
            self.src_lang = detect_image_language(images[0], self.log)
            self.log(f"Đã nhận diện ngôn ngữ: {self.src_lang}", 16.0)

        lang_map = {"en": "en", "japan": "japan", "korean": "korean", "ch": "ch"}
        ocr_lang = lang_map.get(self.src_lang, "en")
        ocr_model = get_ocr_model(ocr_lang, self.det_db_unclip_ratio, self.det_db_box_thresh)

        ocr_data = {}  # {img_path: [OCR item, ...]}

        for idx, img_path in enumerate(images):
            img_name = os.path.basename(img_path)
            pct = 15.0 + (idx / total_images) * 25.0
            self.log(f"Đang quét OCR ảnh {idx + 1}/{total_images}: {img_name}...", pct)

            try:
                cv2_img = cv2.imread(img_path)
                detections = run_ocr_on_image(ocr_model, img_path)
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

                ocr_data[img_path] = group_ocr_boxes(img_ocr_items, self.src_lang, cv2_img)

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
                self.custom_translation_map = parse_translation_json(user_translation_json)
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
                    translated_texts.update(translate_batch(
                        cluster_items, 
                        self.src_lang, 
                        self.tgt_lang, 
                        self.tone, 
                        self.additional_instructions
                    ))

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
                        # Nếu không có bản dịch hoặc bản dịch giống hệt bản gốc, ta giữ nguyên và không xóa
                        if not t_text or t_text.strip() == "" or t_text == item["original_text"]:
                            continue

                        is_bubble, _, best_rect, bg_color = find_bubble_contour_and_rect(cv2_img, item["bbox"])
                        padded_rect = apply_adaptive_padding(item["bbox"], W_img, H_img)

                        # Kiểm tra xem LaMa có khả dụng không và kích thước box có nhỏ không
                        lama_available = self._get_lama_instance() is not None
                        x0, y0, x2, y2 = item["bbox"]
                        box_w, box_h = x2 - x0, y2 - y0
                        is_small_box = (box_w < 250 or box_h < 120)

                        if is_bubble and not (is_small_box and lama_available):
                            # Tô đè màu nền đơn sắc của bong bóng thoại (cho bong bóng lớn hoặc khi không có LaMa)
                            sub_mask = np.zeros(cv2_img.shape[:2], dtype=np.uint8)
                            for poly in item["box_points"]:
                                pts = np.array(poly, dtype=np.int32)
                                cv2.fillPoly(sub_mask, [pts], 255)
                            sub_mask = cv2.dilate(sub_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
                            cv2_img[sub_mask == 255] = bg_color
                            typeset_info[item["id"]] = {"bbox": best_rect, "is_sfx": False}
                        else:
                            # Chữ SFX lơ lửng, hoặc bong bóng nhỏ khi có LaMa → Dùng LaMa / OpenCV Inpaint nâng cao để xóa chữ
                            item_mask = np.zeros(cv2_img.shape[:2], dtype=np.uint8)
                            for poly in item["box_points"]:
                                pts = np.array(poly, dtype=np.int32)
                                cv2.fillPoly(item_mask, [pts], 255)
                            
                            # Tối ưu hóa độ giãn nở (dilation) động cho từng chữ SFX để che phủ hết viền/bóng chữ mà không lem nét vẽ tranh
                            box_h = y2 - y0
                            dilate_size = max(4, min(16, int(box_h * 0.10)))
                            if dilate_size % 2 == 0:
                                dilate_size += 1
                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
                            item_mask = cv2.dilate(item_mask, kernel)
                            
                            sfx_mask = cv2.bitwise_or(sfx_mask, item_mask)
                            has_sfx = True
                            
                            # Giữ nguyên kiểu vẽ chữ (is_sfx=False nếu ban đầu là bubble, dùng best_rect)
                            if is_bubble:
                                typeset_info[item["id"]] = {"bbox": best_rect, "is_sfx": False}
                            else:
                                typeset_info[item["id"]] = {"bbox": padded_rect, "is_sfx": True}

                    if has_sfx:
                        lama = self._get_lama_instance()

                        if lama is not None:
                            # Inpaint cục bộ từng vùng để tiết kiệm RAM và tăng tốc
                            contours, _ = cv2.findContours(sfx_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in contours:
                                x, y, w, h = cv2.boundingRect(cnt)
                                
                                # Tăng kích thước vùng đệm (padding) để LaMa nhận dạng bối cảnh xung quanh tốt hơn
                                pad_w = max(45, int(w * 0.35))
                                pad_h = max(45, int(h * 0.35))
                                
                                xmin, ymin = max(0, x - pad_w), max(0, y - pad_h)
                                xmax = min(cv2_img.shape[1], x + w + pad_w)
                                ymax = min(cv2_img.shape[0], y + h + pad_h)
                                
                                # Đảm bảo kích thước tối thiểu của crop là 160x160 để mô hình AI hoạt động tối ưu nhất
                                crop_w = xmax - xmin
                                crop_h = ymax - ymin
                                if crop_w < 160:
                                    diff = 160 - crop_w
                                    xmin = max(0, xmin - diff // 2)
                                    xmax = min(cv2_img.shape[1], xmax + diff // 2)
                                if crop_h < 160:
                                    diff = 160 - crop_h
                                    ymin = max(0, ymin - diff // 2)
                                    ymax = min(cv2_img.shape[0], ymax + diff // 2)

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
                        draw_text_in_box(draw, t_text, info["bbox"], font_path, is_sfx=info["is_sfx"])

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
        stitch_output_images(output_folder, temp_dir, self.log)

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
