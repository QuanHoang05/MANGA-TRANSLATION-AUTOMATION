import cv2
import numpy as np
import os
from PIL import Image

def get_line_dims(font, line):
    try:
        bb = font.getbbox(line)
        return bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        return len(line) * 8, 16

def erase_text_from_bubble_debug(cv2_img, all_pts):
    H_img, W_img = cv2_img.shape[:2]
    stroke_mask = np.zeros((H_img, W_img), dtype=np.uint8)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    print(f"Total polygons to erase: {len(all_pts)}")
    for idx, pts in enumerate(all_pts):
        pts = np.array(pts, dtype=np.int32)
        poly_mask = np.zeros((H_img, W_img), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 255)
        
        region_pixels = gray[poly_mask == 255]
        if len(region_pixels) == 0:
            print(f"Poly {idx}: region_pixels is empty")
            continue
        bg_val = np.median(region_pixels)
        
        if bg_val > 127:
            thresh = min(180, bg_val - 20)
            text_pixels = (gray < thresh) & (poly_mask == 255)
            print(f"Poly {idx} (Light BG): bg_val={bg_val}, thresh={thresh}, text_pixels={np.sum(text_pixels)}")
        else:
            thresh = max(80, bg_val + 20)
            text_pixels = (gray > thresh) & (poly_mask == 255)
            print(f"Poly {idx} (Dark BG): bg_val={bg_val}, thresh={thresh}, text_pixels={np.sum(text_pixels)}")
            
        stroke_mask[text_pixels] = 255
        
    print(f"Initial stroke_mask sum: {np.sum(stroke_mask)}")
    
    xs_all = []
    ys_all = []
    for pts in all_pts:
        pts = np.array(pts, dtype=np.int32)
        xs_all.extend(pts[:, 0])
        ys_all.extend(pts[:, 1])
        
    if xs_all and ys_all:
        w_box = max(xs_all) - min(xs_all)
        h_box = max(ys_all) - min(ys_all)
        k_size = max(5, int(min(w_box, h_box) * 0.15))
        if k_size % 2 == 0:
            k_size += 1
    else:
        k_size = 5
        
    print(f"Dilating with kernel size: {k_size}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    stroke_mask_dilated = cv2.dilate(stroke_mask, kernel, iterations=1)
    print(f"Dilated stroke_mask sum: {np.sum(stroke_mask_dilated)}")
    
    cv2.imwrite("debug_stroke_mask.png", stroke_mask_dilated)
    
    # Run inpaint
    ys, xs = np.where(stroke_mask_dilated > 0)
    if len(ys) > 0 and len(xs) > 0:
        x0, y0, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        pad = 15
        xmin = max(0, x0 - pad)
        xmax = min(W_img, x2 + pad)
        ymin = max(0, y0 - pad)
        ymax = min(H_img, y2 + pad)
        
        crop = cv2_img[ymin:ymax, xmin:xmax].copy()
        mask_crop = stroke_mask_dilated[ymin:ymax, xmin:xmax].copy()
        inpainted = cv2.inpaint(crop, mask_crop, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        cv2_img[ymin:ymax, xmin:xmax] = inpainted
        print("Inpainted crop area successfully saved.")
    else:
        print("No stroke pixels found to inpaint.")
    
    cv2.imwrite("debug_inpainted.png", cv2_img)

def main():
    from app.pipeline import get_ocr
    ocr = get_ocr('ch')
    img_path = '/workspace/data/jobs/967628cf-f656-4004-aa34-ace974e4926e/temp/input/001_slice.png'
    raw_res = ocr.ocr(img_path)
    
    print("Detections:")
    all_pts_to_erase = []
    for i in range(len(raw_res[0]['rec_texts'])):
        text = raw_res[0]['rec_texts'][i]
        box = raw_res[0]['rec_polys'][i]
        if hasattr(box, 'tolist'): box = box.tolist()
        print(f"  {text}: {box}")
        if text != 'nm': # ignore the first one
            all_pts_to_erase.append(box)
            
    cv2_img = cv2.imread(img_path)
    erase_text_from_bubble_debug(cv2_img, all_pts_to_erase)

if __name__ == '__main__':
    main()
