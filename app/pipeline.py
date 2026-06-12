import os
import zipfile
import shutil
import json
import traceback
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai

# Cache OCR models globally to prevent reloading on each request
_ocr_instances = {}

def get_ocr(lang):
    global _ocr_instances
    if lang not in _ocr_instances:
        # Import inside function to allow application to start fast and fail gracefully
        # if paddleocr is not installed yet
        from paddleocr import PaddleOCR
        print(f"Initializing PaddleOCR for language: {lang}...")
        # show_log=False keeps console clean
        _ocr_instances[lang] = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    return _ocr_instances[lang]


class MangaPipeline:
    def __init__(self, api_key: str, src_lang: str = "en", tone: str = "tự nhiên", batch_size_pages: int = 10, status_callback=None):
        self.api_key = api_key
        self.src_lang = src_lang
        self.tone = tone
        self.batch_size_pages = batch_size_pages
        self.status_callback = status_callback
        
        # Configure Gemini API
        if api_key:
            genai.configure(api_key=api_key)
            
    def log(self, message: str, percent: float):
        if self.status_callback:
            self.status_callback(message, percent)
        else:
            print(f"[{percent:.1f}%] {message}")

    def run_pipeline(self, zip_path: str, output_zip_path: str, temp_dir: str):
        """
        Executes the 5-step Manga Translation Automation Pipeline.
        """
        # Step 1: Extract Input Ingestion
        self.log("BƯỚC 1: Giải nén tệp tin ảnh đầu vào...", 5.0)
        input_folder = os.path.join(temp_dir, "input")
        output_folder = os.path.join(temp_dir, "output")
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(input_folder)
            
        # Get all valid image files sorted
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        images = []
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith(image_extensions) and not file.startswith('._'):
                    images.append(os.path.join(root, file))
                    
        # Sort natural order
        images.sort()
        
        if not images:
            raise ValueError("Không tìm thấy ảnh nào trong tệp ZIP tải lên.")
            
        self.log(f"Đã tìm thấy {len(images)} ảnh truyện tranh. Bắt đầu OCR...", 10.0)
        
        # Step 2: OCR with PaddleOCR
        self.log("BƯỚC 2: Quét OCR nhận diện chữ trên toàn bộ ảnh...", 15.0)
        ocr_model = get_ocr(self.src_lang)
        
        ocr_data = {}  # {image_path: [ {id, text, box_points, bbox} ]}
        total_images = len(images)
        
        for idx, img_path in enumerate(images):
            img_name = os.path.basename(img_path)
            self.log(f"Đang quét OCR ảnh {idx+1}/{total_images}: {img_name}...", 15.0 + (idx / total_images) * 25.0)
            
            try:
                # PaddleOCR result
                # result is a list of lists: [ [[box, (text, confidence)], ...] ]
                result = ocr_model.ocr(img_path, cls=True)
                
                img_ocr_items = []
                if result and len(result) > 0 and result[0] is not None:
                    for bubble_idx, detection in enumerate(result[0]):
                        box = detection[0]  # [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
                        text = detection[1][0]  # English/Chinese text
                        conf = detection[1][1]  # Confidence
                        
                        # Calculate axis-aligned bounding box
                        xs = [pt[0] for pt in box]
                        ys = [pt[1] for pt in box]
                        x0, y0, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        
                        bubble_id = f"{img_name.split('.')[0]}-O{bubble_idx}"
                        img_ocr_items.append({
                            "id": bubble_id,
                            "original_text": text,
                            "box_points": box,  # polygon points for inpainting
                            "bbox": [x0, y0, x2, y2],  # rect for text rendering
                            "confidence": float(conf)
                        })
                
                ocr_data[img_path] = img_ocr_items
                
            except Exception as e:
                self.log(f"Lỗi khi quét OCR ảnh {img_name}: {str(e)}", 15.0 + (idx / total_images) * 25.0)
                ocr_data[img_path] = []
                
        # Step 3: Context-Aware Translation with Gemini
        self.log("BƯỚC 3: Dịch thuật gộp qua API Gemini Studio...", 45.0)
        
        # Flat list of items for batching
        all_ocr_items = []
        for img_path in images:
            all_ocr_items.extend(ocr_data[img_path])
            
        translated_texts = {} # {id: translated_text}
        
        if all_ocr_items:
            # Group images into page clusters of size `batch_size_pages`
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
                
                # Extract text elements in this cluster
                cluster_items = []
                for img_path in cluster:
                    for item in ocr_data[img_path]:
                        cluster_items.append({
                            "id": item["id"],
                            "text": item["original_text"]
                        })
                
                if not cluster_items:
                    continue  # No text to translate in this cluster
                    
                # Call Gemini for this cluster
                translated_batch = self.translate_batch(cluster_items)
                translated_texts.update(translated_batch)
        else:
            self.log("Không phát hiện văn bản nào cần dịch.", 70.0)
            
        # Step 4: Typesetting & Graphics Processing (Inpainting & Rendering)
        self.log("BƯỚC 4: Xóa chữ cũ (Inpainting) & Ráp chữ Việt mới (Typesetting)...", 70.0)
        
        font_path = os.path.join("fonts", "Nunito-Bold.ttf")
        if not os.path.exists(font_path):
            # Fallback path if fonts folder is outside
            font_path = "../fonts/Nunito-Bold.ttf"
            if not os.path.exists(font_path):
                font_path = "Arial"  # rely on PIL to search system paths
                
        for idx, img_path in enumerate(images):
            img_name = os.path.basename(img_path)
            self.log(f"Đang xử lý đồ họa ảnh {idx+1}/{total_images}: {img_name}...", 70.0 + (idx / total_images) * 20.0)
            
            try:
                # Load image for OpenCV inpainting
                cv2_img = cv2.imread(img_path)
                if cv2_img is None:
                    self.log(f"Lỗi: Không đọc được ảnh OpenCV {img_name}", 70.0 + (idx / total_images) * 20.0)
                    shutil.copy(img_path, os.path.join(output_folder, img_name))
                    continue
                    
                items = ocr_data[img_path]
                
                if items:
                    # 1. Erase text (Inpainting)
                    # Create mask
                    mask = np.zeros(cv2_img.shape[:2], dtype=np.uint8)
                    for item in items:
                        pts = np.array(item["box_points"], dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                        
                    # Dilate mask slightly to clean text border details
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    
                    # Apply inpainting
                    inpainted_cv2 = cv2.inpaint(cv2_img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                    
                    # Convert to PIL Image for high-quality text rendering
                    inpainted_rgb = cv2.cvtColor(inpainted_cv2, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(inpainted_rgb)
                else:
                    # If no text elements, just load image directly into PIL
                    pil_img = Image.open(img_path).convert("RGB")
                    
                # 2. Render Text
                draw = ImageDraw.Draw(pil_img)
                
                for item in items:
                    t_text = translated_texts.get(item["id"], item["original_text"])
                    # Wrap and render text within bbox
                    self.draw_text_in_box(draw, t_text, item["bbox"], font_path)
                    
                # Save to output folder
                output_img_path = os.path.join(output_folder, img_name)
                pil_img.save(output_img_path)
                
            except Exception as e:
                self.log(f"Lỗi khi vẽ chữ trên ảnh {img_name}: {str(e)}", 70.0 + (idx / total_images) * 20.0)
                traceback.print_exc()
                # Copy original as fallback
                shutil.copy(img_path, os.path.join(output_folder, img_name))
                
        # Step 5: Archive & Output Ingestion
        self.log("BƯỚC 5: Nén file kết quả và chuẩn bị tải về...", 90.0)
        
        # Zip output folder
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
            for root, _, files in os.walk(output_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_folder)
                    zip_out.write(file_path, arcname)
                    
        self.log("HỆ THỐNG: Đã hoàn thành toàn bộ quy trình dịch truyện thành công!", 100.0)
        
    def translate_batch(self, items: list) -> dict:
        """
        Sends a batch of text blocks to Gemini and returns structured ID-translation mappings.
        """
        # Format the request JSON
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
        
        # Configure model schema for strict safety
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
        
        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": schema
                }
            )
            
            response = model.generate_content(prompt)
            response_data = json.loads(response.text)
            
            # Map items
            result_map = {}
            for item in response_data.get("translations", []):
                result_map[item["id"]] = item["translated_text"]
            return result_map
            
        except Exception as e:
            print(f"Gemini API Error: {str(e)}")
            traceback.print_exc()
            # In case of API failure, return empty map so system falls back to original text
            return {}

    def wrap_text(self, text: str, font, max_width: float) -> list:
        """
        Splits text into lines that do not exceed max_width.
        """
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
            # Use font.getbbox for modern PIL compatibility
            bbox = font.getbbox(test_line)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    # Single word is wider than box width, force it
                    lines.append(word)
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    def draw_text_in_box(self, draw: ImageDraw.Draw, text: str, bbox: list, font_path: str):
        """
        Scales and draws wrapped text centered inside a bounding box [x0, y0, x2, y2].
        Applies a thin stroke outline for clarity.
        """
        x0, y0, x2, y2 = bbox
        box_w = x2 - x0
        box_h = y2 - y0
        
        optimal_font = None
        optimal_lines = []
        optimal_line_heights = []
        
        # Try to find a font size from 28 down to 10 that makes the text fit
        for font_size in range(28, 9, -2):
            try:
                # Load TTF font
                if font_path == "Arial":
                    font = ImageFont.load_default() # Fallback
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
            
            # If wrapped text fits within box boundaries, use this font size
            if max_line_w <= box_w and total_text_h <= box_h:
                optimal_font = font
                optimal_lines = lines
                optimal_line_heights = line_heights
                break
                
        # If still none fit, force standard size 10 font
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
        
        # Center vertically
        current_y = y0 + (box_h - total_text_h) / 2
        
        # Draw each line centered horizontally
        for i, line in enumerate(optimal_lines):
            l_bbox = optimal_font.getbbox(line)
            line_w = l_bbox[2] - l_bbox[0]
            line_h = optimal_line_heights[i]
            
            current_x = x0 + (box_w - line_w) / 2
            
            # Draw text with 2px white border (stroke) and black text for legibility
            draw.text(
                (current_x, current_y), 
                line, 
                fill=(0, 0, 0), 
                font=optimal_font,
                stroke_width=2,
                stroke_fill=(255, 255, 255)
            )
            current_y += line_h + spacing
