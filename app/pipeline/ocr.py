import gc

_ocr_cache = {}

def get_ocr_model(
    lang: str, 
    det_db_unclip_ratio: float = 1.6, 
    det_db_box_thresh: float = 0.6
):
    """Khởi tạo (hoặc lấy từ cache) đối tượng PaddleOCR cho ngôn ngữ chỉ định."""
    global _ocr_cache
    if lang not in _ocr_cache:
        from paddleocr import PaddleOCR
        _ocr_cache[lang] = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            enable_mkldnn=False,
            det_limit_side_len=3000,
            det_limit_type='max',
            drop_score=0.3,
            cpu_threads=2,
            det_db_unclip_ratio=det_db_unclip_ratio,
            det_db_box_thresh=det_db_box_thresh
        )
    return _ocr_cache[lang]

def detect_image_language(img_path: str, log_fn=None) -> str:
    """
    Dùng model OCR 'ch' đọc thử 1 ảnh để nhận dạng tập ký tự và suy ra ngôn ngữ.
    """
    try:
        ocr = get_ocr_model("ch")
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
        if log_fn:
            log_fn(f"Lỗi khi dò ngôn ngữ tự động: {e}. Mặc định dùng 'en'.", 15.5)
        else:
            print(f"Lỗi khi dò ngôn ngữ tự động: {e}. Mặc định dùng 'en'.")
        return "en"

def run_ocr_on_image(ocr_model, img_path: str) -> list:
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
