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
    def __init__(self, api_key: str, src_lang: str = "en", tone: str = "tự nhiên", batch_size_pages: int = 10, additional_instructions: str = "", status_callback=None, custom_translation: str = ""):
        self.api_key = api_key
        self.src_lang = src_lang
        self.tone = tone
        self.batch_size_pages = batch_size_pages
        self.additional_instructions = additional_instructions
        self.status_callback = status_callback
        
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
            if lang in ["ch", "japan", "ko"]:
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

    def slice_large_image(self, img_path: str, output_folder: str, prefix: str, max_height: int = 10000) -> list[str]:
        """
        Tự động phân mảnh dọc ảnh siêu dài thành các phần nhỏ để tránh crash OpenCV/PaddleOCR
        và giảm thiểu nguy cơ quá tải bộ nhớ (OOM).
        """
        from PIL import Image
        import numpy as np
        import cv2
        
        img = Image.open(img_path)
        w, h = img.size
        
        if h <= max_height:
            # Không cần phân mảnh, chỉ cần lưu file
            ext = os.path.splitext(img_path)[1].lower()
            new_path = os.path.join(output_folder, f"{prefix}{ext}")
            img.save(new_path)
            return [new_path]
            
        self.log(f"Phát hiện ảnh siêu dài ({w}x{h}). Bắt đầu phân mảnh ảnh dọc...", 7.0)
        
        # Chuyển sang ảnh grayscale để tính toán độ phân tán (variance) hàng loạt nhanh hơn
        img_np = np.array(img.convert("RGB"))
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        slice_paths = []
        y = 0
        part_idx = 1
        
        target_max = max_height
        target_min = int(max_height * 0.75) # ví dụ: 7500px
        
        while y < h:
            if h - y <= target_max:
                # Phân mảnh cuối cùng
                slice_y = h
            else:
                # Tìm điểm cắt tốt nhất trong khoảng [y + target_min, y + target_max]
                search_start = y + target_min
                search_end = min(y + target_max, h)
                
                slice_gray = gray_img[search_start:search_end]
                # Tính phương sai hàng ngang (variance) của từng dòng để tìm khoảng trắng phân tách các khung truyện (panels)
                row_variances = np.var(slice_gray, axis=1)
                
                best_local_y = np.argmin(row_variances)
                slice_y = search_start + best_local_y
                
            # Cắt ảnh
            slice_img = img.crop((0, y, w, slice_y))
            
            ext = os.path.splitext(img_path)[1].lower()
            part_name = f"{prefix}_part{part_idx:02d}{ext}"
            part_path = os.path.join(output_folder, part_name)
            slice_img.save(part_path)
            
            slice_paths.append(part_path)
            self.log(f"Đã cắt phân mảnh {part_idx}: dòng {y} đến {slice_y} (Cao: {slice_y - y}px)", 7.0)
            
            y = slice_y
            part_idx += 1
            
        return slice_paths

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
        images = []
        part_counter = 1
        for raw_img_path in raw_images:
            prefix = f"{part_counter:03d}"
            sliced_paths = self.slice_large_image(raw_img_path, input_folder, prefix)
            images.extend(sliced_paths)
            part_counter += len(sliced_paths)
            
        self.log(f"Đã chuẩn hóa và tìm thấy {len(images)} ảnh truyện tranh. Bắt đầu OCR...", 10.0)
        
        # Bước 2: Nhận diện chữ bằng mô hình OCR (PaddleOCR)
        self.log("BƯỚC 2: Quét OCR nhận diện chữ trên toàn bộ ảnh...", 15.0)
        ocr_model = get_ocr(self.src_lang)
        
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
                    # 1. Xóa văn bản gốc bằng công nghệ Inpainting
                    # Tạo ảnh mặt nạ nhị phân (Binary Mask)
                    mask = np.zeros(cv2_img.shape[:2], dtype=np.uint8)
                    has_changes = False
                    for item in items:
                        t_text = translated_texts.get(item["id"])
                        if t_text and t_text != item["original_text"]:
                            for poly in item["box_points"]:
                                pts = np.array(poly, dtype=np.int32)
                                cv2.fillPoly(mask, [pts], 255)
                            has_changes = True
                            
                    if has_changes:
                        # Giãn nở nhẹ mặt nạ để bao phủ triệt để bóng chữ cũ
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                        mask = cv2.dilate(mask, kernel, iterations=1)
                        
                        # Tiến hành phục hồi ảnh (tẩy chữ)
                        inpainted_cv2 = cv2.inpaint(cv2_img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                        
                        # Chuyển đổi ngược về Pillow Image để vẽ chữ chất lượng cao
                        inpainted_rgb = cv2.cvtColor(inpainted_cv2, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(inpainted_rgb)
                    else:
                        pil_img = Image.open(img_path).convert("RGB")
                else:
                    # Nếu trang không chứa văn bản, chỉ cần mở trực tiếp bằng Pillow
                    pil_img = Image.open(img_path).convert("RGB")
                    
                # 2. Vẽ chữ dịch mới lên ảnh
                draw = ImageDraw.Draw(pil_img)
                
                for item in items:
                    t_text = translated_texts.get(item["id"])
                    if t_text and t_text != item["original_text"]:
                        # Tiến hành ngắt câu và vẽ chữ cân đối vào hộp thoại
                        self.draw_text_in_box(draw, t_text, item["bbox"], font_path)
                    
                # Lưu ảnh kết quả vào thư mục đầu ra
                output_img_path = os.path.join(output_folder, img_name)
                pil_img.save(output_img_path)
                
            except Exception as e:
                self.log(f"Lỗi khi vẽ chữ trên ảnh {img_name}: {str(e)}", 70.0 + (idx / total_images) * 20.0)
                traceback.print_exc()
                # Nếu xử lý thất bại, copy ảnh gốc sang thư mục đầu ra để làm phương án dự phòng
                shutil.copy(img_path, os.path.join(output_folder, img_name))
                
        # Bước 5: Đóng gói và chuẩn bị đầu ra (Archive & Output Ingestion)
        self.log("BƯỚC 5: Nén file kết quả và chuẩn bị tải về...", 90.0)
        
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
        
        prompt = f"""
Bạn là một dịch giả truyện tranh (manga/webtoon) chuyên nghiệp. Hãy dịch các câu thoại từ ngôn ngữ gốc sang tiếng Việt.

HƯỚNG DẪN XƯNG HÔ & PHONG CÁCH:
- Đảm bảo xưng hô đồng bộ, tự nhiên và phù hợp với ngữ cảnh của câu chuyện (ví dụ: Ta - Ngươi, Thiếu chủ - Đại nhân, Tôi - Cậu, Anh - Em, Tỷ tỷ - Muội muội...).
- Tông giọng dịch yêu cầu: {self.tone} (ví dụ: tự nhiên, lịch sự, cổ trang, dễ thương).
- Dựa vào mạch truyện của các câu thoại liên tiếp để suy luận mối quan hệ nhân vật chính xác.
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
        Tự động ngắt văn bản thành các dòng nhỏ không vượt quá chiều rộng max_width.
        """
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
            # Sử dụng getbbox để tương thích tốt với Pillow phiên bản mới (PIL 10+)
            bbox = font.getbbox(test_line)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    # Nếu một từ đơn quá rộng so với khung thoại, ép nó xuống dòng mới đơn độc
                    lines.append(word)
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    def draw_text_in_box(self, draw: ImageDraw.Draw, text: str, bbox: list, font_path: str):
        """
        Tự động chọn kích thước font chữ tối ưu, ngắt dòng và vẽ văn bản cân giữa 
        vào bên trong hộp giới hạn [x0, y0, x2, y2]. Có tạo viền trắng để tăng độ nét.
        """
        x0, y0, x2, y2 = bbox
        box_w = x2 - x0
        box_h = y2 - y0
        
        optimal_font = None
        optimal_lines = []
        optimal_line_heights = []
        
        # Thử nghiệm giảm kích thước font chữ từ 28 xuống 10 để chọn kích cỡ vừa vặn nhất
        for font_size in range(28, 9, -2):
            try:
                # Tải font chữ TrueType
                if font_path == "Arial":
                    font = ImageFont.load_default() # Font dự phòng mặc định
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
                
        # Nếu không có kích cỡ nào vừa khít, ép buộc dùng size chữ tối thiểu là 10
        if optimal_font is None:
            try:
                if font_path == "Arial":
                    optimal_font = ImageFont.load_default()
                else:
                    optimal_font = ImageFont.truetype(font_path, 10)
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
            
            # Vẽ chữ màu đen với viền trắng stroke dày 2px để tăng độ tương phản trên mọi phông nền
            draw.text(
                (current_x, current_y), 
                line, 
                fill=(0, 0, 0), 
                font=optimal_font,
                stroke_width=2,
                stroke_fill=(255, 255, 255)
            )
            current_y += line_h + spacing
