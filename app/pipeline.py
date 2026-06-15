import os
import zipfile
import shutil
import json
import traceback
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
import google.generativeai as genai

# Cache các đối tượng OCR toàn cục để tránh việc tải lại mô hình trong mỗi yêu cầu
_ocr_instances = {}

def get_ocr(lang):
    global _ocr_instances
    # Ánh xạ ngôn ngữ viết tắt sang định dạng PaddleOCR hỗ trợ
    if lang.lower() == "ko":
        lang = "korean"
    elif lang.lower() == "jp":
        lang = "japan"
        
    if lang not in _ocr_instances:
        # Import bên trong hàm để giúp ứng dụng khởi động nhanh hơn và tránh lỗi crash
        # nếu thư viện paddleocr chưa được cài đặt
        from paddleocr import PaddleOCR
        import paddle
        
        # Tự động kiểm tra xem thiết bị hiện tại có hỗ trợ GPU/CUDA không để tối ưu tốc độ
        use_gpu = False
        try:
            use_gpu = paddle.device.is_compiled_with_cuda()
            print(f"Hệ thống phát hiện hỗ trợ GPU: {use_gpu}")
        except Exception as e:
            print(f"Không thể kiểm tra GPU: {str(e)}. Mặc định chuyển sang chạy bằng CPU.")
            
        print(f"Đang khởi tạo đối tượng PaddleOCR cho ngôn ngữ: {lang} (Chạy trên GPU = {use_gpu})...")
        
        # Thử khởi tạo theo chuẩn PaddleOCR phiên bản mới (3.x+) trước
        try:
            device = "gpu" if use_gpu else "cpu"
            # Lưu ý: PaddleOCR 3.x+ gặp lỗi với enable_mkldnn=True trên một số hệ thống CPU, đặt enable_mkldnn=False cho an toàn
            _ocr_instances[lang] = PaddleOCR(
                use_textline_orientation=True, 
                lang=lang, 
                device=device,
                enable_mkldnn=False
            )
        except Exception as e:
            print(f"Không thể khởi tạo bằng tham số PaddleOCR 3.x+ ({str(e)}). Thử bằng tham số PaddleOCR 2.x...")
            try:
                # Dự phòng cho PaddleOCR 2.x
                _ocr_instances[lang] = PaddleOCR(
                    use_angle_cls=True, 
                    lang=lang, 
                    show_log=False, 
                    use_gpu=use_gpu
                )
            except Exception as e2:
                print(f"Lỗi khi khởi tạo PaddleOCR 2.x: {str(e2)}")
                raise e2
    return _ocr_instances[lang]


class MangaPipeline:
    def __init__(self, api_key: str, src_lang: str = "en", tone: str = "tự nhiên", batch_size_pages: int = 10, additional_instructions: str = "", status_callback=None, custom_translation: str = "", use_yolo: bool = False):
        self.api_key = api_key
        self.src_lang = src_lang
        self.tone = tone
        self.batch_size_pages = batch_size_pages
        self.additional_instructions = additional_instructions
        self.status_callback = status_callback
        self.use_yolo = False
        self.yolo_model = None

        
        # Cấu hình API cho Google Gemini
        if api_key:
            genai.configure(api_key=api_key)
            
        # Phân tích cú pháp bản dịch tùy chỉnh do người dùng nhập vào (nếu có)
        self.custom_translation_map = {}
        if custom_translation:
            try:
                data = json.loads(custom_translation)
                if isinstance(data, dict):
                    if "translations" in data and isinstance(data["translations"], list):
                        for item in data["translations"]:
                            if isinstance(item, dict) and "id" in item and "translated_text" in item:
                                self.custom_translation_map[item["id"]] = item["translated_text"]
                    else:
                        self.custom_translation_map = data
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "id" in item and "translated_text" in item:
                            self.custom_translation_map[item["id"]] = item["translated_text"]
            except Exception as pe:
                print(f"Cảnh báo: Không thể phân tích cú pháp custom_translation JSON: {pe}")

    def load_yolo_model(self):
        """
        Khởi tạo mô hình YOLOv8 cho việc nhận diện bong bóng thoại nếu thư viện và file model tồn tại.
        """
        try:
            from ultralytics import YOLO
            import os
            model_path = os.path.join("models", "yolov8_comic.pt")
            if os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
                print(f"HỆ THỐNG: Đã nạp thành công mô hình YOLOv8 từ {model_path}.")
            else:
                print(f"HỆ THỐNG CẢNH BÁO: Không tìm thấy file model YOLO tại {model_path}.")
                self.use_yolo = False
        except ImportError:
            print("HỆ THỐNG CẢNH BÁO: Thư viện ultralytics chưa được cài đặt. Tắt tính năng YOLO.")
            self.use_yolo = False
        except Exception as e:
            print(f"Lỗi khi nạp mô hình YOLO: {e}")
            self.use_yolo = False

    def detect_bubbles_yolo(self, cv_img) -> list:
        """
        Sử dụng YOLOv8 để trả về danh sách các bounding box của bong bóng thoại.
        Trả về list các list tọa độ: [[x1, y1, x2, y2], ...]
        """
        if not self.yolo_model:
            return []
            
        try:
            results = self.yolo_model.predict(cv_img, verbose=False, conf=0.25)
            boxes = []
            if len(results) > 0 and results[0].boxes:
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    boxes.append([x1, y1, x2, y2])
            return boxes
        except Exception as e:
            print(f"Lỗi khi nhận diện bong bóng thoại bằng YOLO: {e}")
            return []
            
    def log(self, message: str, percent: float, event_type: str = None, data = None):
        if self.status_callback:
            try:
                self.status_callback(message, percent, event_type, data)
            except TypeError:
                try:
                    self.status_callback(message, percent)
                except Exception:
                    pass
        else:
            print(f"[{percent:.1f}%] {message}")

    def group_ocr_boxes(self, ocr_items: list, lang: str) -> list:
        """
        Gom nhóm các dòng chữ quét được từ OCR thuộc cùng một ô thoại (speech bubble)
        để dịch trọn vẹn câu và vẽ chữ cân đối, tránh lỗi đè chữ.
        """
        if not ocr_items:
            return []
            
        n = len(ocr_items)
        parent = list(range(n))
        
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                
        # Duyệt và liên kết các hộp thoại gần nhau
        for i in range(n):
            for j in range(i + 1, n):
                boxA = ocr_items[i]["bbox"]
                boxB = ocr_items[j]["bbox"]
                
                x0_A, y0_A, x2_A, y2_A = boxA
                x0_B, y0_B, x2_B, y2_B = boxB
                
                h_A = y2_A - y0_A
                h_B = y2_B - y0_B
                h_avg = (h_A + h_B) / 2.0
                
                # Khoảng cách dọc giữa hai ô chữ
                v_gap = max(0, y0_B - y2_A) if y0_B >= y2_A else max(0, y0_A - y2_B)
                # Khoảng cách ngang (độ chồng chéo ngang)
                overlap_x = min(x2_A, x2_B) - max(x0_A, x0_B)
                
                # Điều kiện gom nhóm:
                # 1. Khoảng cách dọc nhỏ (dưới 1.5 lần chiều cao chữ trung bình)
                # 2. Có chồng chéo chiều ngang hoặc khoảng cách ngang rất bé (dưới 0.8 lần chiều cao chữ)
                is_close_v = v_gap <= h_avg * 1.5
                is_overlapping_h = overlap_x > - (h_avg * 0.8)
                
                if is_close_v and is_overlapping_h:
                    union(i, j)
                    
        # Nhóm các mục theo gốc liên kết
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(ocr_items[i])
            
        merged_items = []
        for g_idx, group in enumerate(groups.values()):
            # Sắp xếp các dòng chữ trong nhóm từ trên xuống dưới
            group.sort(key=lambda item: item["bbox"][1])
            
            # Ghép chuỗi văn bản gốc
            texts = [item["original_text"] for item in group]
            if lang.lower() in ["ch", "chinese", "jp", "japan", "japanese", "ko", "korean"]:
                combined_text = "".join(texts)
            else:
                combined_text = " ".join(texts)
                
            # Tính toán hộp bao (bbox) gộp cho cả nhóm
            x0_comb = min(item["bbox"][0] for item in group)
            y0_comb = min(item["bbox"][1] for item in group)
            x2_comb = max(item["bbox"][2] for item in group)
            y2_comb = max(item["bbox"][3] for item in group)
            
            # Gộp danh sách đa giác chữ (polygons) để phục vụ xóa chữ
            combined_box_points = []
            for item in group:
                combined_box_points.append(item["box_points"])
                
            first_id = group[0]["id"]
            img_prefix = first_id.split("-")[0]
            merged_id = f"{img_prefix}-B{g_idx}"
            
            merged_items.append({
                "id": merged_id,
                "original_text": combined_text,
                "box_points": combined_box_points,
                "bbox": [x0_comb, y0_comb, x2_comb, y2_comb],
                "confidence": sum(item["confidence"] for item in group) / len(group)
            })
            
        # Sắp xếp lại danh sách các bong bóng thoại gộp theo thứ tự xuất hiện trên trang ảnh
        merged_items.sort(key=lambda item: item["bbox"][1])
        return merged_items

    def stitch_and_smart_slice(self, raw_images: list[str], output_folder: str, target_max_height: int = 2000) -> list[str]:
        """
        Nối dọc toàn bộ ảnh của 1 chương thành 1 dải duy nhất để chữa lành bong bóng thoại bị cắt ngang,
        sau đó sử dụng thuật toán quét phương sai ngang (Row Variance) để cắt thông minh tại khoảng trắng
        phù hợp với tiêu chuẩn PaddleOCR (2000px segment).
        """
        from PIL import Image
        import numpy as np
        import cv2
        import os
        
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
            
        # Chọn chiều rộng chung là chiều rộng của ảnh đầu tiên
        w_common = loaded_images[0][0].width
        
        resized_images = []
        total_height = 0
        for img, path in loaded_images:
            w, h = img.size
            if w != w_common:
                h_new = int(h * (w_common / w))
                img_resized = img.resize((w_common, h_new), Image.Resampling.LANCZOS)
                resized_images.append(img_resized)
                total_height += h_new
            else:
                resized_images.append(img)
                total_height += h
                
        self.log(f"Tổng chiều cao sau khi khâu: {total_height}px. Chiều rộng chung: {w_common}px", 7.0)
        
        # Nối tất cả thành 1 Mega Image
        mega_img = Image.new("RGB", (w_common, total_height), (255, 255, 255))
        current_y = 0
        for img in resized_images:
            mega_img.paste(img, (0, current_y))
            current_y += img.height
            
        # Chuyển sang ảnh grayscale để tính phương sai ngang nhanh hơn
        self.log("Đang phân tích phương sai dòng để tìm điểm cắt thông minh...", 8.0)
        mega_np = np.array(mega_img)
        mega_gray = cv2.cvtColor(mega_np, cv2.COLOR_RGB2GRAY)
        
        # Tính phương sai hàng ngang
        row_variances = np.var(mega_gray, axis=1)
        
        y = 0
        part_idx = 1
        slice_paths = []
        
        min_slice_h = int(target_max_height * 0.7)
        max_slice_h = target_max_height
        
        while y < total_height:
            if total_height - y <= max_slice_h:
                slice_y = total_height
            else:
                search_start = y + min_slice_h
                search_end = min(y + max_slice_h, total_height)
                
                # Quét và tìm dòng có phương sai nhỏ nhất (tức là khoảng trống trơn màu)
                local_variances = row_variances[search_start:search_end]
                best_local_y = np.argmin(local_variances)
                slice_y = search_start + best_local_y
                
            # Cắt ảnh
            slice_img = mega_img.crop((0, y, w_common, slice_y))
            
            part_name = f"{part_idx:03d}_slice.png"
            part_path = os.path.join(output_folder, part_name)
            slice_img.save(part_path)
            slice_paths.append(part_path)
            
            self.log(f"Đã cắt phân mảnh {part_idx}: dòng {y} đến {slice_y} (Cao: {slice_y - y}px)", 9.0)
            
            y = slice_y
            part_idx += 1
            
        return slice_paths

    def max_inscribed_rectangle(self, mask_binary) -> tuple:
        """
        Tìm hình chữ nhật có diện tích lớn nhất nằm hoàn toàn trong mặt nạ nhị phân.
        Sử dụng thuật toán quy hoạch động tối ưu O(H * W).
        """
        H, W = mask_binary.shape
        heights = np.zeros(W, dtype=np.int32)
        max_area = 0
        best_rect = (0, 0, 0, 0) # x0, y0, x2, y2
        
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
                        end_c = c - 1
                        start_r = r - curr_h + 1
                        end_r = r
                        best_rect = (start_c, start_r, end_c, end_r)
                stack.append(c)
        return best_rect

    def find_bubble_contour_and_rect(self, cv2_img, bbox):
        """
        Dò tìm ranh giới thực sự của bong bóng thoại bằng cv2.floodFill từ tâm bbox
        để tránh tràn mặt nạ, sau đó tính Hình chữ nhật nội tiếp lớn nhất.
        """
        import cv2
        import numpy as np
        
        x0, y0, x2, y2 = map(int, bbox)
        box_w = x2 - x0
        box_h = y2 - y0
        H_img, W_img = cv2_img.shape[:2]
        
        # Định vị vùng crop rộng xung quanh bounding box để dò biên bong bóng thoại
        pad_x = int(box_w * 0.3) + 15
        pad_y = int(box_h * 0.3) + 15
        xmin = max(0, x0 - pad_x)
        xmax = min(W_img, x2 + pad_x)
        ymin = max(0, y0 - pad_y)
        ymax = min(H_img, y2 + pad_y)
        
        crop = cv2_img[ymin:ymax, xmin:xmax]
        crop_h, crop_w = crop.shape[:2]
        
        if crop_h <= 0 or crop_w <= 0:
            return False, None, [x0, y0, x2, y2], (255, 255, 255)
            
        # Tìm điểm bắt đầu (seedPoint) tại trung tâm của Bounding Box
        seed_x = int((x0 + x2) / 2 - xmin)
        seed_y = int((y0 + y2) / 2 - ymin)
        
        # Đảm bảo seedPoint nằm trong khung ảnh crop
        seed_x = max(0, min(crop_w - 1, seed_x))
        seed_y = max(0, min(crop_h - 1, seed_y))
        
        # Tránh việc chọn phải pixel chữ (màu tối), tìm pixel sáng nhất trong vùng lân cận
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray_crop[seed_y, seed_x] < 80:
            ny_min = max(0, seed_y - 15)
            ny_max = min(crop_h, seed_y + 15)
            nx_min = max(0, seed_x - 15)
            nx_max = min(crop_w, seed_x + 15)
            sub_gray = gray_crop[ny_min:ny_max, nx_min:nx_max]
            if sub_gray.size > 0:
                _, _, _, max_loc = cv2.minMaxLoc(sub_gray)
                seed_x = nx_min + max_loc[0]
                seed_y = ny_min + max_loc[1]
                
        # Khởi tạo mask cho floodFill (kích thước lớn hơn ảnh gốc 2 pixel ở mỗi chiều)
        ff_mask = np.zeros((crop_h + 2, crop_w + 2), dtype=np.uint8)
        
        # Chạy thuật toán floodFill loang màu khắt khe loDiff=(5,5,5) và upDiff=(5,5,5)
        # để chống tràn màu ra các khu vực quần áo/bầu trời bên ngoài bong bóng thoại
        cv2.floodFill(
            image=crop,
            mask=ff_mask,
            seedPoint=(seed_x, seed_y),
            newVal=(255, 255, 255),
            loDiff=(5, 5, 5),
            upDiff=(5, 5, 5),
            flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        )
        
        # Lấy mask bong bóng thoại sau khi loang màu (loại bỏ lề 1px của floodFill)
        bubble_mask = ff_mask[1:-1, 1:-1]
        
        # Tính toán diện tích bong bóng thoại quét được
        bubble_pixels = np.sum(bubble_mask == 255)
        min_area = box_w * box_h * 0.35
        
        is_bubble = False
        best_rect = [x0, y0, x2, y2]
        best_contour_abs = None
        bg_color = (255, 255, 255)
        
        if bubble_pixels >= min_area:
            # Tìm contours từ mask của floodFill để trích xuất đa giác bong bóng thoại
            contours, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Tìm contour lớn nhất chứa điểm seedPoint
                cx_crop = seed_x
                cy_crop = seed_y
                bubble_cnt = None
                max_cnt_area = 0
                for cnt in contours:
                    if cv2.pointPolygonTest(cnt, (cx_crop, cy_crop), False) >= 0:
                        cnt_area = cv2.contourArea(cnt)
                        if cnt_area > max_cnt_area:
                            max_cnt_area = cnt_area
                            bubble_cnt = cnt
                            
                if bubble_cnt is not None:
                    is_bubble = True
                    best_contour_abs = bubble_cnt + np.array([xmin, ymin])
                    
                    # Xác định màu nền trung vị (median BGR color) của bong bóng thoại
                    bg_pixels = crop[bubble_mask == 255]
                    if bg_pixels.size > 0:
                        median_bgr = np.median(bg_pixels, axis=0)
                        bg_color = (int(median_bgr[0]), int(median_bgr[1]), int(median_bgr[2]))
                        
                    # Áp dụng xói mòn (erosion) trên mặt nạ để chữ không chạm sát viền đen bong bóng
                    k_size = max(3, int(min(box_w, box_h) * 0.08))
                    if k_size % 2 == 0:
                        k_size += 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                    eroded_mask = cv2.erode(bubble_mask, kernel, iterations=1)
                    
                    # Chạy thuật toán quy hoạch động tìm Hình chữ nhật nội tiếp lớn nhất
                    inscribed = self.max_inscribed_rectangle(eroded_mask > 0)
                    if inscribed[2] > inscribed[0] and inscribed[3] > inscribed[1]:
                        best_rect = [
                            inscribed[0] + xmin,
                            inscribed[1] + ymin,
                            inscribed[2] + xmin,
                            inscribed[3] + ymin
                        ]
                        
        return is_bubble, best_contour_abs, best_rect, bg_color

    def run_pipeline(self, zip_path: str, output_zip_path: str, temp_dir: str):
        """
        Thực thi quy trình 5 bước tự động dịch truyện tranh.
        """
        # Bước 1: Thu thập và chuẩn bị đầu vào (Giải nén hoặc đọc thư mục chứa ảnh thô)
        self.log("BƯỚC 1: Thu thập và chuẩn bị ảnh đầu vào...", 5.0)
        raw_extract_folder = os.path.join(temp_dir, "raw_extract")
        input_folder = os.path.join(temp_dir, "input")
        output_folder = os.path.join(temp_dir, "output")
        os.makedirs(raw_extract_folder, exist_ok=True)
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        
        # Nếu zip_path thực chất là thư mục chứa các ảnh đơn lẻ do người dùng chọn trực tiếp
        if os.path.isdir(zip_path):
            for file in os.listdir(zip_path):
                src = os.path.join(zip_path, file)
                if os.path.isfile(src):
                    shutil.copy(src, os.path.join(raw_extract_folder, file))
        else:
            # Nếu là tệp ZIP lưu trữ ảnh
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_extract_folder)
            
        # Thu thập toàn bộ tệp tin ảnh từ thư mục thô thô ban đầu
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        raw_images = []
        for root, _, files in os.walk(raw_extract_folder):
            for file in files:
                if file.lower().endswith(image_extensions) and not file.startswith('._'):
                    raw_images.append(os.path.join(root, file))
                    
        if not raw_images:
            raise ValueError("Không tìm thấy ảnh hợp lệ nào trong dữ liệu tải lên.")
            
        # Thuật toán sắp xếp tự nhiên (Natural Sort) tránh lỗi nhảy thứ tự số ví dụ: "2" đứng trước "10"
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
            
        raw_images.sort(key=natural_sort_key)
        
        # Chuẩn hóa tên ảnh về chuỗi số tăng dần liên tục (001, 002...) để tránh các ký tự tiếng Việt, khoảng trống gây lỗi dịch
        # Tiến hành phân mảnh (slice) dọc nếu ảnh siêu dài (chiều cao > 10000px) tránh crash OpenCV/PaddleOCR
        # Thực hiện Stitching và Smart Slicing (Khâu dọc và cắt ảnh thông minh tại khoảng trắng)
        images = self.stitch_and_smart_slice(raw_images, input_folder)
        
        self.log(f"Đã khâu và cắt thông minh thành {len(images)} trang ảnh. Bắt đầu OCR...", 10.0)
        
        # BƯỚC 2: Nhận diện chữ (OCR)
        self.log("BƯỚC 2: Đang tải mô hình nhận diện chữ (PaddleOCR)...", 15.0)
        
        if self.src_lang.lower() == "auto":
            self.log("Đang dò ngôn ngữ tự động từ ảnh đầu tiên...", 15.5)
            sample_img = images[0]
            detected_lang = self.detect_image_language(sample_img)
            self.src_lang = detected_lang
            self.log(f"Đã nhận diện ngôn ngữ: {detected_lang}", 16.0)
            
        from paddleocr import PaddleOCR
        lang_map = {"en": "en", "japan": "japan", "korean": "korean", "ch": "ch"}
        ocr_lang = lang_map.get(self.src_lang, "en")
        ocr_model = PaddleOCR(use_angle_cls=True, lang=ocr_lang, enable_mkldnn=False, det_limit_side_len=3000, det_limit_type='max', drop_score=0.3)
        
        ocr_data = {}  # Cấu trúc: {đường_dẫn_ảnh: [ {id, text, box_points, bbox} ]}
        total_images = len(images)
        
        for idx, img_path in enumerate(images):
            img_name = os.path.basename(img_path)
            self.log(f"Đang quét OCR ảnh {idx+1}/{total_images}: {img_name}...", 15.0 + (idx / total_images) * 25.0)
            
            try:
                # Gọi mô hình PaddleOCR
                try:
                    # PaddleOCR 3.x+ không nhận đối số `cls` ở phương thức dự đoán
                    raw_result = ocr_model.ocr(img_path)
                except TypeError:
                    # Dự phòng cho PaddleOCR 2.x
                    raw_result = ocr_model.ocr(img_path, cls=True)
                
                # Chuẩn hóa định dạng kết quả giữa PaddleOCR 3.x+ và 2.x
                result = raw_result
                if raw_result and isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict) and 'rec_texts' in raw_result[0]:
                    normalized = []
                    for page_data in raw_result:
                        page_items = []
                        rec_texts = page_data.get('rec_texts', [])
                        rec_scores = page_data.get('rec_scores', [])
                        rec_polys = page_data.get('rec_polys', [])
                        
                        for i in range(len(rec_texts)):
                            box = rec_polys[i] if i < len(rec_polys) else []
                            if hasattr(box, 'tolist'):
                                box = box.tolist()
                            
                            text = rec_texts[i]
                            score = rec_scores[i] if i < len(rec_scores) else 1.0
                            page_items.append([box, (text, score)])
                        normalized.append(page_items)
                    result = normalized
                
                img_ocr_items = []
                if result and len(result) > 0 and result[0] is not None:
                    for bubble_idx, detection in enumerate(result[0]):
                        box = detection[0]  # Tọa độ đa giác 4 điểm: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
                        text = detection[1][0]  # Văn bản gốc (tiếng Anh/Trung...)
                        conf = detection[1][1]  # Độ tin cậy của OCR
                        
                        # Tính toán tọa độ hộp bao thẳng cạnh (axis-aligned bounding box)
                        xs = [pt[0] for pt in box]
                        ys = [pt[1] for pt in box]
                        x0, y0, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        
                        bubble_id = f"{img_name.split('.')[0]}-O{bubble_idx}"
                        img_ocr_items.append({
                            "id": bubble_id,
                            "original_text": text,
                            "box_points": box,  # Các điểm đa giác phục vụ cho bước xóa chữ
                            "bbox": [x0, y0, x2, y2],  # Hộp chữ nhật phục vụ bước vẽ chữ
                            "confidence": float(conf)
                        })
                
                grouped_items = self.group_ocr_boxes(img_ocr_items, self.src_lang)
                ocr_data[img_path] = grouped_items
                
            except Exception as e:
                self.log(f"Lỗi khi quét OCR ảnh {img_name}: {str(e)}", 15.0 + (idx / total_images) * 25.0)
                ocr_data[img_path] = []
                
        # Trích xuất dữ liệu OCR thô dạng danh sách để hiển thị trên giao diện và tải về
        ocr_results_list = []
        for img_path in images:
            for item in ocr_data.get(img_path, []):
                ocr_results_list.append({
                    "id": item["id"],
                    "text": item["original_text"]
                })
        self.log("Đã hoàn thành quét OCR toàn bộ các trang.", 45.0, event_type="ocr_completed", data=ocr_results_list)
        
        # Bước 3: Dịch thuật ngữ cảnh thông qua API Gemini (Context-Aware Translation)
        self.log("BƯỚC 3: Dịch thuật gộp qua API Gemini Studio...", 45.0)
        
        # Phẳng hóa danh sách các mục thoại để chuẩn bị gộp dịch
        all_ocr_items = []
        for img_path in images:
            all_ocr_items.extend(ocr_data[img_path])
            
        translated_texts = {} # Chứa kết quả ánh xạ {id: văn_bản_đã_dịch}
        
        if self.custom_translation_map:
            self.log("HỆ THỐNG: Sử dụng bản dịch JSON tùy chỉnh do người dùng cung cấp.", 48.0)
            # Điền các bản dịch từ custom map
            for item in all_ocr_items:
                item_id = item["id"]
                if item_id in self.custom_translation_map:
                    translated_texts[item_id] = self.custom_translation_map[item_id]
            
            # Gửi sự kiện cập nhật bản dịch về client
            self.log("Đã áp dụng bản dịch tùy chỉnh.", 70.0, event_type="translation_completed", data=translated_texts)
            
        elif all_ocr_items:
            if not self.api_key:
                self.log("HỆ THỐNG CẢNH BÁO: Không có API Key và không có Bản dịch tùy chỉnh. Bỏ qua bước dịch thuật.", 70.0)
                self.log("Bỏ qua dịch thuật.", 70.0, event_type="translation_completed", data={})
            else:
                # Gom các ảnh thành từng cụm trang dựa trên kích thước `batch_size_pages`
                page_clusters = []
                current_cluster = []
                
                for i, img_path in enumerate(images):
                    current_cluster.append(img_path)
                    if len(current_cluster) >= self.batch_size_pages or i == len(images) - 1:
                        page_clusters.append(current_cluster)
                        current_cluster = []
                
                total_clusters = len(page_clusters)
                self.log(f"Tổng cộng có {len(all_ocr_items)} ô thoại. Chia làm {total_clusters} nhóm trang để gửi Gemini...", 48.0)
                
                for c_idx, cluster in enumerate(page_clusters):
                    self.log(f"Đang gửi nhóm trang {c_idx+1}/{total_clusters} lên Gemini...", 48.0 + (c_idx / total_clusters) * 22.0)
                    
                    # Thu thập tất cả các ô thoại trong cụm trang hiện tại
                    cluster_items = []
                    for img_path in cluster:
                        for item in ocr_data[img_path]:
                            cluster_items.append({
                                "id": item["id"],
                                "text": item["original_text"]
                            })
                    
                    if not cluster_items:
                        continue  # Không có chữ nào cần dịch trong cụm này
                        
                    # Gửi cụm thoại lên Gemini API để xử lý dịch gộp ngữ cảnh
                    translated_batch = self.translate_batch(cluster_items)
                    translated_texts.update(translated_batch)
                
                # Gửi sự kiện cập nhật bản dịch về client
                self.log("Đã nhận bản dịch từ Gemini.", 70.0, event_type="translation_completed", data=translated_texts)
        else:
            self.log("Không phát hiện văn bản nào cần dịch.", 70.0)
            self.log("Không có văn bản dịch.", 70.0, event_type="translation_completed", data={})
            
        # Bước 4: Xử lý đồ họa & Ráp chữ tự động (Inpainting & Typesetting)
        self.log("BƯỚC 4: Xóa chữ cũ (Inpainting) & Ráp chữ Việt mới (Typesetting)...", 70.0)
        
        # Xác định đường dẫn font chữ hỗ trợ tiếng Việt
        font_path = os.path.join("fonts", "Nunito-Bold.ttf")
        if not os.path.exists(font_path):
            # Đường dẫn dự phòng nếu thư mục fonts ở ngoài thư mục hiện tại
            font_path = "../fonts/Nunito-Bold.ttf"
            if not os.path.exists(font_path):
                font_path = "Arial"  # Dùng Arial có sẵn trong hệ thống làm phương án dự phòng
                
        for idx, img_path in enumerate(images):
            img_name = os.path.basename(img_path)
            self.log(f"Đang xử lý đồ họa ảnh {idx+1}/{total_images}: {img_name}...", 70.0 + (idx / total_images) * 20.0)
            
            try:
                # Đọc ảnh phục vụ xử lý inpainting bằng OpenCV
                cv2_img = cv2.imread(img_path)
                if cv2_img is None:
                    self.log(f"Lỗi: Không đọc được ảnh OpenCV {img_name}", 70.0 + (idx / total_images) * 20.0)
                    shutil.copy(img_path, os.path.join(output_folder, img_name))
                    continue
                    
                items = ocr_data[img_path]
                
                if items:
                    # 1. Định vị và Xóa nền Lai (Hybrid Inpainting)
                    sfx_mask = np.zeros(cv2_img.shape[:2], dtype=np.uint8)
                    has_sfx = False
                    typeset_info = {}
                    
                    yolo_boxes = []
                    if self.use_yolo:
                        yolo_boxes = self.detect_bubbles_yolo(cv2_img)
                    
                    for item in items:
                        t_text = translated_texts.get(item["id"])
                        if not t_text or t_text == item["original_text"]:
                            continue
                            
                        # Chạy thuật toán tìm contour bong bóng và hình chữ nhật nội tiếp lớn nhất (nếu bật YOLOv8 / Bubble Detection)
                        if self.use_yolo:
                            matched_yolo_box = None
                            ocr_x0, ocr_y0, ocr_x2, ocr_y2 = item["bbox"]
                            ocr_cx = (ocr_x0 + ocr_x2) / 2
                            ocr_cy = (ocr_y0 + ocr_y2) / 2
                            for ybox in yolo_boxes:
                                yx1, yy1, yx2, yy2 = ybox
                                if yx1 <= ocr_cx <= yx2 and yy1 <= ocr_cy <= yy2:
                                    matched_yolo_box = ybox
                                    break
                            
                            if matched_yolo_box:
                                is_bubble = True
                                best_rect = [matched_yolo_box[0]+2, matched_yolo_box[1]+2, matched_yolo_box[2]-2, matched_yolo_box[3]-2]
                                bg_color = (255, 255, 255)
                            else:
                                is_bubble, bubble_cnt, best_rect, bg_color = self.find_bubble_contour_and_rect(cv2_img, item["bbox"])
                        else:
                            is_bubble = True
                            best_rect = item["bbox"]
                            bg_color = (255, 255, 255)
                        
                        if is_bubble:
                            # Nhánh 1: Tô đè màu đơn sắc của bong bóng thoại (Local Color Padding)
                            sub_mask = np.zeros(cv2_img.shape[:2], dtype=np.uint8)
                            for poly in item["box_points"]:
                                pts = np.array(poly, dtype=np.int32)
                                cv2.fillPoly(sub_mask, [pts], 255)
                            # Giãn nở mask chữ thêm 4 pixel để nuốt sạch nét viền chữ cũ
                            sub_mask = cv2.dilate(sub_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
                            cv2_img[sub_mask == 255] = bg_color
                            
                            typeset_info[item["id"]] = {
                                "bbox": best_rect,
                                "is_sfx": False
                            }
                        else:
                            # Nhánh 2: Chữ tự do/SFX lơ lửng -> Gom vào mask để chạy Inpaint nâng cao
                            for poly in item["box_points"]:
                                pts = np.array(poly, dtype=np.int32)
                                cv2.fillPoly(sfx_mask, [pts], 255)
                            has_sfx = True
                            
                            typeset_info[item["id"]] = {
                                "bbox": item["bbox"], # SFX giữ nguyên box cũ
                                "is_sfx": True
                            }
                            
                    if has_sfx:
                        # Giãn nở mặt nạ SFX thêm 5px để nuốt hết bóng và viền chữ cũ
                        sfx_mask = cv2.dilate(sfx_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
                        
                        # Chạy inpaint cho SFX bằng LaMa-ONNX hoặc Fallback OpenCV
                        lama = None
                        try:
                            # Import động SimpleLama
                            from simple_lama_inpainting import SimpleLama
                            if not hasattr(self, "_lama_instance"):
                                self.log("Đang khởi tạo mô hình AI LaMa-ONNX cho SFX...", 72.0)
                                self._lama_instance = SimpleLama()
                            lama = self._lama_instance
                        except Exception as le:
                            print(f"Cảnh báo: Không dùng được LaMa-ONNX ({le}). Sử dụng OpenCV Inpaint làm dự phòng.")
                            
                        if lama is not None:
                            # LaMa nhận PIL Image và PIL Mask
                            pil_img_temp = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
                            pil_mask_temp = Image.fromarray(sfx_mask)
                            inpainted_pil = lama(pil_img_temp, pil_mask_temp)
                            cv2_img = cv2.cvtColor(np.array(inpainted_pil), cv2.COLOR_RGB2BGR)
                        else:
                            # OpenCV Inpaint
                            cv2_img = cv2.inpaint(cv2_img, sfx_mask, inpaintRadius=8, flags=cv2.INPAINT_TELEA)
                            
                    # Chuyển sang PIL Image để vẽ chữ tiếng Việt mới chất lượng cao
                    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
                else:
                    pil_img = Image.open(img_path).convert("RGB")
                    typeset_info = {}
                    
                # 2. Vẽ chữ dịch mới lên ảnh
                draw = ImageDraw.Draw(pil_img)
                
                for item in items:
                    t_text = translated_texts.get(item["id"])
                    if t_text and t_text != item["original_text"]:
                        # Lấy thông tin tọa độ và định dạng chữ đã tối ưu từ bước trước
                        info = typeset_info.get(item["id"], {"bbox": item["bbox"], "is_sfx": False})
                        self.draw_text_in_box(draw, t_text, info["bbox"], font_path, is_sfx=info["is_sfx"])
                    
                # Làm nét ảnh (Upscaling & Sharpening)
                w_img, h_img = pil_img.size
                if w_img < 1200:
                    ratio = 1200 / w_img
                    new_w = 1200
                    new_h = int(h_img * ratio)
                    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    from PIL import ImageFilter
                    pil_img = pil_img.filter(ImageFilter.SHARPEN)
                    
                # Lưu ảnh kết quả vào thư mục đầu ra
                output_img_path = os.path.join(output_folder, img_name)
                pil_img.save(output_img_path)
                
            except Exception as e:
                self.log(f"Lỗi khi vẽ chữ trên ảnh {img_name}: {str(e)}", 70.0 + (idx / total_images) * 20.0)
                traceback.print_exc()
                # Nếu xử lý thất bại, copy ảnh gốc sang thư mục đầu ra để làm phương án dự phòng
                shutil.copy(img_path, os.path.join(output_folder, img_name))
                
        # Bước 5: Ghép nối ảnh và chuẩn bị đầu ra (Stitch & Archive)
        self.log("BƯỚC 5: Nén file kết quả và chuẩn bị tải về...", 90.0)
        
        # Ghép dọc toàn bộ ảnh kết quả
        self.stitch_output_images(output_folder, temp_dir)
        
        # Nén thư mục output thành tệp ZIP
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
            for root, _, files in os.walk(output_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_folder)
                    zip_out.write(file_path, arcname)
                    
        self.log("HỆ THỐNG: Đã hoàn thành toàn bộ quy trình dịch truyện thành công!", 100.0)
        
    def translate_batch(self, items: list) -> dict:
        """
        Gửi một cụm thoại lên Gemini API và nhận ánh xạ dịch thuật có cấu trúc (ID - Bản dịch).
        """
        # Chuẩn bị dữ liệu JSON gửi đi
        batch_payload = {
            "context": "Đây là danh sách các câu thoại trong truyện tranh cần dịch sang tiếng Việt. Hãy đảm bảo xưng hô đồng bộ, tự nhiên, bám sát mạch truyện.",
            "items": items
        }
        
        # Chuẩn hóa mã ngôn ngữ gốc hiển thị
        lang_map = {
            "ko": "tiếng Hàn",
            "korean": "tiếng Hàn",
            "japan": "tiếng Nhật",
            "jp": "tiếng Nhật",
            "ch": "tiếng Trung",
            "zh": "tiếng Trung",
            "en": "tiếng Anh",
            "english": "tiếng Anh"
        }
        lang_name = lang_map.get(self.src_lang.lower(), self.src_lang)

        prompt = f"""
Bạn là một dịch giả truyện tranh (manga/webtoon) chuyên nghiệp. Hãy dịch các câu thoại từ ngôn ngữ {lang_name} sang tiếng Việt.

HƯỚNG DẪN XƯNG HÔ & PHONG CÁCH:
- Đảm bảo xưng hô đồng bộ, tự nhiên và phù hợp với ngữ cảnh của câu chuyện (ví dụ: Ta - Ngươi, Thiếu chủ - Đại nhân, Tôi - Cậu, Anh - Em, Tỷ tỷ - Muội muội...).
- Tông giọng dịch yêu cầu: {self.tone} (ví dụ: tự nhiên, lịch sự, cổ trang, dễ thương).
- Dựa vào mạch truyện của các câu thoại liên tiếp để suy luận mối quan hệ nhân vật chính xác. BẮT BUỘC TUÂN THỦ TONE DỊCH.
"""

        # Bổ sung chỉ dẫn riêng từ người dùng để cá nhân hóa ngữ nghĩa dịch
        if self.additional_instructions:
            prompt += f"\nYÊU CẦU DỊCH THUẬT & XƯNG HÔ ĐẶC BIỆT TỪ NGƯỜI DÙNG:\n- {self.additional_instructions}\n"

        prompt += f"""
DỮ LIỆU ĐẦU VÀO (định dạng JSON chứa danh sách câu thoại kèm ID):
{json.dumps(batch_payload, ensure_ascii=False, indent=2)}

Hãy dịch toàn bộ danh sách trên và trả về kết quả dưới định dạng JSON khớp chính xác với cấu trúc sau:
{{
  "translations": [
    {{
      "id": "chuỗi ID gốc của câu thoại",
      "translated_text": "nội dung đã dịch sang tiếng Việt"
    }}
  ]
}}
"""
        
        # Cấu hình JSON Schema để ép định dạng trả về từ mô hình AI
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
        
        # Thử liệt kê mô hình khả dụng trước qua API key để tự phát hiện tên model chính xác
        model_names_to_try = []
        try:
            supported_models = list(genai.list_models())
            flash_models = [
                m.name.replace("models/", "") 
                for m in supported_models 
                if "flash" in m.name.lower() and "generatecontent" in "".join(m.supported_generation_methods).lower()
            ]
            if flash_models:
                flash_models.sort(reverse=True)
                model_names_to_try.extend(flash_models)
        except Exception as le:
            print(f"Cảnh báo: Không thể liệt kê danh sách model ({le})")
            
        # Thêm các tên model dự phòng tiêu chuẩn
        for fallback in ["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"]:
            if fallback not in model_names_to_try:
                model_names_to_try.append(fallback)
                
        response_data = None
        last_err = None
        
        for m_name in model_names_to_try:
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
                break
            except Exception as e:
                last_err = e
                print(f"Model {m_name} lỗi hoặc không được hỗ trợ: {str(e)}")
                continue
                
        if not response_data:
            print("Tất cả các model thử nghiệm đều thất bại.")
            if last_err:
                raise last_err
            else:
                raise Exception("Không thể khởi tạo dịch thuật Gemini.")
                
        try:
            # Chuyển kết quả JSON thành từ điển ánh xạ
            result_map = {}
            for item in response_data.get("translations", []):
                result_map[item["id"]] = item["translated_text"]
            return result_map
            
        except Exception as e:
            print(f"Lỗi phân tích kết quả dịch: {str(e)}")
            traceback.print_exc()
            return {}

    def wrap_text(self, text: str, font, max_width: float) -> list:
        """
        Tự động ngắt văn bản thành các dòng nhỏ sử dụng thư viện textwrap.
        """
        import textwrap
        
        # Ước lượng độ rộng trung bình của một ký tự để tính width cho textwrap
        sample = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        try:
            bbox = font.getbbox(sample)
            avg_char_w = (bbox[2] - bbox[0]) / len(sample)
        except Exception:
            avg_char_w = 10.0
            
        if avg_char_w <= 0:
            avg_char_w = 10.0
            
        chars_per_line = int(max_width / avg_char_w)
        if chars_per_line < 1:
            chars_per_line = 1
            
        return textwrap.wrap(text, width=chars_per_line)

    def draw_text_in_box(self, draw: ImageDraw.Draw, text: str, bbox: list, font_path: str, is_sfx: bool = False):
        """
        Tự động chọn kích thước font chữ tối ưu, ngắt dòng và vẽ văn bản cân giữa 
        vào bên trong hộp giới hạn [x0, y0, x2, y2]. Có tạo viền để tăng độ nét.
        """
        x0, y0, x2, y2 = bbox
        box_w = x2 - x0
        box_h = y2 - y0
        
        optimal_font = None
        optimal_lines = []
        optimal_line_heights = []
        
        # Thử nghiệm giảm kích thước font chữ từ 28 xuống 6 (Clamp tối đa 28, tối thiểu 6)
        for font_size in range(28, 5, -2):
            try:
                if font_path == "Arial":
                    font = ImageFont.load_default()
                else:
                    font = ImageFont.truetype(font_path, font_size)
            except Exception:
                font = ImageFont.load_default()
                
            lines = self.wrap_text(text, font, box_w)
            
            line_heights = []
            max_line_w = 0
            for line in lines:
                l_bbox = font.getbbox(line)
                line_w = l_bbox[2] - l_bbox[0]
                line_h = l_bbox[3] - l_bbox[1]
                line_heights.append(line_h)
                if line_w > max_line_w:
                    max_line_w = line_w
                    
            spacing = 4
            total_text_h = sum(line_heights) + spacing * (len(lines) - 1) if line_heights else 0
            
            # Nếu cả chiều rộng và chiều cao khối chữ nằm vừa khít trong ô thoại thì chọn cỡ font này
            if max_line_w <= box_w and total_text_h <= box_h:
                optimal_font = font
                optimal_lines = lines
                optimal_line_heights = line_heights
                break
                
        # Nếu không có kích cỡ nào vừa khít trong khoảng [6, 28], ép cứng kích thước font tối thiểu là 6 (Clamp)
        if optimal_font is None:
            try:
                if font_path == "Arial":
                    optimal_font = ImageFont.load_default()
                else:
                    optimal_font = ImageFont.truetype(font_path, 6)
            except Exception:
                optimal_font = ImageFont.load_default()
            optimal_lines = self.wrap_text(text, optimal_font, box_w)
            optimal_line_heights = []
            for line in optimal_lines:
                l_bbox = optimal_font.getbbox(line)
                optimal_line_heights.append(l_bbox[3] - l_bbox[1])
                
        spacing = 4
        total_text_h = sum(optimal_line_heights) + spacing * (len(optimal_lines) - 1) if optimal_line_heights else 0
        
        # Căn giữa văn bản theo chiều dọc (Vertical Centering)
        current_y = y0 + (box_h - total_text_h) / 2
        
        # Căn giữa văn bản theo chiều ngang cho từng dòng (Horizontal Centering)
        for i, line in enumerate(optimal_lines):
            l_bbox = optimal_font.getbbox(line)
            line_w = l_bbox[2] - l_bbox[0]
            line_h = optimal_line_heights[i]
            
            current_x = x0 + (box_w - line_w) / 2
            
            # Cấu hình màu vẽ và độ dày viền tùy thuộc vào loại chữ (SFX vs Bong bóng)
            if is_sfx:
                # SFX: Vẽ chữ màu trắng với viền đen dày để che đi chữ Trung Quốc cũ lơ lửng
                draw.text(
                    (current_x, current_y), 
                    line, 
                    fill=(255, 255, 255), 
                    font=optimal_font,
                    stroke_width=4,
                    stroke_fill=(0, 0, 0)
                )
            else:
                # Bong bóng thoại: Chữ màu đen, viền trắng mỏng 2px
                draw.text(
                    (current_x, current_y), 
                    line, 
                    fill=(0, 0, 0), 
                    font=optimal_font,
                    stroke_width=2,
                    stroke_fill=(255, 255, 255)
                )
            current_y += line_h + spacing

    def stitch_output_images(self, output_folder: str, temp_dir: str):
        """
        Ghép nối dọc tất cả các ảnh trong thư mục output thành một ảnh dài duy nhất
        (Webtoon format) và lưu lại dưới dạng file JPEG chất lượng cao.
        """
        import cv2
        import numpy as np
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        image_files = []
        for file in sorted(os.listdir(output_folder)):
            if file.lower().endswith(image_extensions) and not file.startswith('._'):
                image_files.append(os.path.join(output_folder, file))
                
        if not image_files:
            return
            
        images = []
        for path in image_files:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
                
        if not images:
            return
            
        # Nối dọc (vstack) tất cả các ảnh lại với nhau
        # Đảm bảo các ảnh cùng chiều rộng trước khi nối
        w_common = images[0].shape[1]
        resized_images = []
        for img in images:
            h, w = img.shape[:2]
            if w != w_common:
                h_new = int(h * (w_common / w))
                img_resized = cv2.resize(img, (w_common, h_new), interpolation=cv2.INTER_LANCZOS4)
                resized_images.append(img_resized)
            else:
                resized_images.append(img)
                
        stitched_img = np.vstack(resized_images)
        stitched_path = os.path.join(temp_dir, "translated_stitched.jpg")
        
        # Lưu chất lượng JPEG cao (95)
    def detect_image_language(self, img_path: str) -> str:
        """
        Dùng model OCR 'ch' để đọc thử 1 ảnh, quét các ký tự để đoán ngôn ngữ.
        """
        try:
            from paddleocr import PaddleOCR
            temp_ocr = PaddleOCR(use_angle_cls=True, lang="ch", enable_mkldnn=False, det_limit_side_len=3000, det_limit_type='max', drop_score=0.3)
            try:
                result = temp_ocr.ocr(img_path)
            except TypeError:
                result = temp_ocr.ocr(img_path, cls=True)
                
            if not result or not result[0]:
                return "en"
                
            full_text = ""
            for line in result[0]:
                full_text += line[1][0]
                
            has_kana = any('\u3040' <= char <= '\u30ff' for char in full_text)
            has_hangul = any('\uac00' <= char <= '\ud7a3' for char in full_text)
            has_hanzi = any('\u4e00' <= char <= '\u9fff' for char in full_text)
            
            if has_kana:
                return "japan"
            elif has_hangul:
                return "korean"
            elif has_hanzi:
                return "ch"
            else:
                return "en"
        except Exception as e:
            self.log(f"Lỗi khi dò ngôn ngữ tự động: {e}. Mặc định dùng 'en'.", 15.5)
            return "en"

    def draw_text_with_custom_size(self, draw, text: str, bbox: list, font_path: str, custom_size=None):
        from PIL import ImageFont
        x0, y0, x2, y2 = bbox
        box_w = x2 - x0
        box_h = y2 - y0

        if custom_size and str(custom_size).isdigit() and int(custom_size) > 0:
            font_size = int(custom_size)
            try:
                font = ImageFont.truetype(font_path, font_size) if font_path != "Arial" else ImageFont.load_default()
            except:
                font = ImageFont.load_default()
                
            lines = self.wrap_text(text, font, box_w)
            line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
            spacing = 4
            total_text_h = sum(line_heights) + spacing * (len(lines) - 1) if line_heights else 0
            current_y = y0 + (box_h - total_text_h) / 2
            
            for i, line in enumerate(lines):
                l_bbox = font.getbbox(line)
                line_w = l_bbox[2] - l_bbox[0]
                current_x = x0 + (box_w - line_w) / 2
                draw.text((current_x, current_y), line, fill=(0, 0, 0), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
                current_y += line_heights[i] + spacing
        else:
            self.draw_text_in_box(draw, text, bbox, font_path, False)

    def rerender_single_image(self, job_dir: str, filename: str, boxes: list):
        import cv2
        import numpy as np
        import os
        from PIL import Image, ImageDraw
        
        input_path = os.path.join(job_dir, "temp", "input", filename)
        output_folder = os.path.join(job_dir, "temp", "output")
        output_path = os.path.join(output_folder, filename)
        
        if not os.path.exists(input_path):
            raise Exception(f"Không tìm thấy ảnh gốc {filename}")
            
        cv2_img = cv2.imread(input_path)
        if cv2_img is None:
            raise Exception(f"Lỗi đọc ảnh {filename}")
            
        # Xóa nền cũ (Tô trắng toàn bộ vùng box do người dùng chỉ định)
        for box in boxes:
            x0, y0, x2, y2 = [int(v) for v in box["bbox"]]
            cv2.rectangle(cv2_img, (x0, y0), (x2, y2), (255, 255, 255), -1)
            
        cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_img_rgb)
        draw = ImageDraw.Draw(pil_img)
        font_path = os.path.join("fonts", "Nunito-Bold.ttf")
        if not os.path.exists(font_path):
            font_path = "Arial"
            
        # Vẽ chữ mới
        for box in boxes:
            text = box.get("text", "")
            if not text: continue
            self.draw_text_with_custom_size(draw, text, box["bbox"], font_path, box.get("font_size"))
            
        cv2_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, cv2_result)
        
        self.stitch_output_images(output_folder, os.path.join(job_dir, "temp"))
        return output_path
