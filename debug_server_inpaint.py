import cv2
import os
import json
from PIL import Image, ImageDraw
import numpy as np

def main():
    from app.pipeline import MangaPipeline
    p = MangaPipeline(None)
    p.src_lang = 'ch'
    job_id = '967628cf-f656-4004-aa34-ace974e4926e'
    temp_dir = f'/workspace/data/jobs/{job_id}/temp'
    input_folder = os.path.join(temp_dir, 'input')
    output_folder = os.path.join(temp_dir, 'output_debug')
    os.makedirs(output_folder, exist_ok=True)
    
    img_name = '001_slice.png'
    img_path = os.path.join(input_folder, img_name)
    
    from app.pipeline import get_ocr
    ocr = get_ocr('ch')
    raw_res = ocr.ocr(img_path)
    page_items = []
    for i in range(len(raw_res[0]['rec_texts'])):
        box = raw_res[0]['rec_polys'][i]
        if hasattr(box, 'tolist'): box = box.tolist()
        text = raw_res[0]['rec_texts'][i]
        conf = raw_res[0]['rec_scores'][i]
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        page_items.append({'id': f'001_slice-O{i}', 'original_text': text, 'box_points': box, 'bbox': [min(xs), min(ys), max(xs), max(ys)], 'confidence': conf})
        
    grouped = p.group_ocr_boxes(page_items[1:], 'ch')
    nm_box = page_items[0]
    grouped.insert(0, {
        'id': '001_slice-B0',
        'original_text': nm_box['original_text'],
        'box_points': [nm_box['box_points']],
        'bbox': nm_box['bbox'],
        'confidence': nm_box['confidence']
    })
    
    for idx, item in enumerate(grouped):
        item['id'] = f'001_slice-B{idx}'
        item['polygon'] = [[item['bbox'][0], item['bbox'][1]], [item['bbox'][2], item['bbox'][1]], [item['bbox'][2], item['bbox'][3]], [item['bbox'][0], item['bbox'][3]]]
        item['is_bubble'] = True
        
    translated_texts = {
        "001_slice-B0": "Cốt truyện",
        "001_slice-B1": "Câu chuyện về hoàng tử và kẻ ăn mày có vẻ ngoài giống hệt nhau gặp gỡ nhau",
        "001_slice-B2": "Họ giấu mọi người và hoán đổi quần áo cũng như cuộc sống cho nhau"
    }
    
    cv2_img = cv2.imread(img_path)
    typeset_info = {}
    
    for item in grouped:
        t_text = translated_texts.get(item["id"])
        if not t_text or t_text == item["original_text"]:
            continue
        
        is_bubble = item.get("is_bubble", True)
        polygon = item.get("polygon", [])
        box_points = item.get("box_points", [])
        
        print(f"Erasing text for ID: {item['id']}, text: {item['original_text']}, box_points count: {len(box_points)}")
        p.erase_text_from_bubble(cv2_img, polygon, box_points, is_bubble)
        
        best_rect = p.get_inscribed_rect_from_polygon(cv2_img, polygon)
        typeset_info[item["id"]] = {
            "bbox": best_rect,
            "is_sfx": not is_bubble
        }
        
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_path = "/workspace/fonts/Nunito-Bold.ttf"
    
    for item in grouped:
        t_text = translated_texts.get(item["id"])
        if t_text and t_text != item["original_text"]:
            info = typeset_info.get(item["id"], {"bbox": item["bbox"], "is_sfx": False})
            print(f"Drawing text for ID: {item['id']}, bbox: {info['bbox']}")
            p.draw_text_in_box(draw, t_text, info["bbox"], font_path, is_sfx=info["is_sfx"])
            
    pil_img.save(os.path.join(output_folder, img_name))
    print("Saved output_debug/001_slice.png successfully")

if __name__ == '__main__':
    main()
