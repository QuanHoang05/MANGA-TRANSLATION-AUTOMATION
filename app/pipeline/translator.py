import json
import google.generativeai as genai

def parse_translation_json(json_str: str) -> dict:
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

def translate_batch(
    items: list, 
    src_lang: str, 
    tgt_lang: str, 
    tone: str, 
    additional_instructions: str = ""
) -> dict:
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

    src_name = lang_display.get(src_lang.lower(), src_lang)
    tgt_name = tgt_display.get(tgt_lang.lower(), tgt_lang)

    prompt = f"""Bạn là một dịch giả truyện tranh (manga/webtoon) chuyên nghiệp. Hãy dịch các câu thoại từ {src_name} sang {tgt_name}.

HƯỚNG DẪN XƯNG HÔ & PHONG CÁCH:
- Đảm bảo xưng hô đồng bộ, tự nhiên và phù hợp với ngữ cảnh câu chuyện.
- Tông giọng dịch yêu cầu: {tone}.
- Dựa vào mạch truyện của các câu thoại liên tiếp để suy luận mối quan hệ nhân vật chính xác.
"""
    if additional_instructions:
        prompt += f"\nYÊU CẦU ĐẶC BIỆT TỪ NGƯỜI DÙNG:\n- {additional_instructions}\n"

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
