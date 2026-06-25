import math
import numpy as np
import cv2

def group_ocr_boxes(ocr_items: list, lang: str, cv2_img=None) -> list:
    """
    Gom nhóm các dòng chữ OCR thuộc cùng một bong bóng thoại bằng Union-Find.
    Nếu có cv2_img, chỉ merge các box nằm trong cùng một bong bóng thoại (bubble contour).
    Nếu không phát hiện bong bóng (như chữ SFX), sử dụng khoảng cách tâm để merge.
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

    is_cjk = lang.lower() in ("ch", "chinese", "jp", "japan", "japanese", "ko", "korean")

    # Bước 1: Dò tìm bong bóng thoại cho từng box trên ảnh
    box_bubble_idx = [None] * n
    if cv2_img is not None:
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        H_img, W_img = gray.shape[:2]
        bubble_masks = []  # Danh sách các (mask, bounding_box_of_mask)

        for i in range(n):
            x0, y0, x2, y2 = map(int, ocr_items[i]["bbox"])
            cx, cy = int((x0 + x2) / 2), int((y0 + y2) / 2)
            cx = max(0, min(W_img - 1, cx))
            cy = max(0, min(H_img - 1, cy))

            # Kiểm tra nhanh xem tâm của box i có nằm trong bất kỳ bubble mask nào đã tìm thấy không
            found_idx = None
            for idx, (mask, (bx0, by0, bx2, by2)) in enumerate(bubble_masks):
                if bx0 <= cx <= bx2 and by0 <= cy <= by2:
                    if mask[cy, cx] == 255:
                        found_idx = idx
                        break

            if found_idx is not None:
                box_bubble_idx[i] = found_idx
            else:
                # Nếu chưa nằm trong mask nào, chạy floodFill tìm bubble từ box i (chạy CỤC BỘ để tránh tràn màu)
                seed_x, seed_y = cx, cy
                box_w, box_h = x2 - x0, y2 - y0

                if gray[seed_y, seed_x] < 80:
                    # Chọn điểm sáng nhất lân cận nếu điểm trung tâm trúng nét chữ đen
                    ny0, ny1 = max(0, seed_y - 15), min(H_img, seed_y + 15)
                    nx0, nx1 = max(0, seed_x - 15), min(W_img, seed_x + 15)
                    sub = gray[ny0:ny1, nx0:nx1]
                    if sub.size > 0:
                        _, _, _, max_loc = cv2.minMaxLoc(sub)
                        seed_x = nx0 + max_loc[0]
                        seed_y = ny0 + max_loc[1]

                # Định nghĩa vùng biên cục bộ (crop) để giới hạn floodfill chống loang toàn trang
                pad_w = int(box_w * 1.5) + 120
                pad_h = int(box_h * 1.5) + 120
                xmin = max(0, seed_x - pad_w)
                xmax = min(W_img, seed_x + pad_w)
                ymin = max(0, seed_y - pad_h)
                ymax = min(H_img, seed_y + pad_h)

                crop = gray[ymin:ymax, xmin:xmax].copy()
                crop_h, crop_w = crop.shape[:2]

                seed_x_crop = seed_x - xmin
                seed_y_crop = seed_y - ymin

                ff_mask = np.zeros((crop_h + 2, crop_w + 2), dtype=np.uint8)
                cv2.floodFill(
                    image=crop,
                    mask=ff_mask,
                    seedPoint=(seed_x_crop, seed_y_crop),
                    newVal=255,
                    loDiff=10,
                    upDiff=10,
                    flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
                )
                local_mask = ff_mask[1:-1, 1:-1]
                bubble_pixels = int(np.sum(local_mask == 255))

                if bubble_pixels >= box_w * box_h * 0.35:
                    ys, xs = np.where(local_mask == 255)
                    if xs.size > 0 and ys.size > 0:
                        # Ánh xạ ngược tọa độ cục bộ sang toàn cục
                        global_bubble_mask = np.zeros((H_img, W_img), dtype=np.uint8)
                        global_bubble_mask[ymin:ymax, xmin:xmax] = local_mask
                        
                        bbox_mask = (xs.min() + xmin, ys.min() + ymin, xs.max() + xmin, ys.max() + ymin)
                        bubble_masks.append((global_bubble_mask, bbox_mask))
                        box_bubble_idx[i] = len(bubble_masks) - 1

    # Bước 2: Duyệt qua các cặp để quyết định merge
    for i in range(n):
        for j in range(i + 1, n):
            idx_A = box_bubble_idx[i]
            idx_B = box_bubble_idx[j]

            x0A, y0A, x2A, y2A = ocr_items[i]["bbox"]
            x0B, y0B, x2B, y2B = ocr_items[j]["bbox"]

            cxA, cyA = (x0A + x2A) / 2.0, (y0A + y2A) / 2.0
            cxB, cyB = (x0B + x2B) / 2.0, (y0B + y2B) / 2.0

            dist = math.sqrt((cxA - cxB)**2 + (cyA - cyB)**2)
            h_avg = ((y2A - y0A) + (y2B - y0B)) / 2.0

            # TH1: Cả hai đều thuộc cùng một bong bóng thoại đã quét
            if idx_A is not None and idx_B is not None and idx_A == idx_B:
                # Giới hạn khoảng cách tối đa để tránh gộp các bubble đôi/sát nhau có đường/kênh nối
                if dist < max(h_avg * 2.0, 60):
                    union(i, j)
            else:
                # TH2: Là chữ ngoài bong bóng hoặc không có cv2_img -> Merge
                w_avg = ((x2A - x0A) + (x2B - x0B)) / 2.0
                
                # Kiểm tra xem có phải chữ dọc tiếng Nhật/Trung không (h > w)
                is_vert_A = (y2A - y0A) > (x2A - x0A)
                is_vert_B = (y2B - y0B) > (x2B - x0B)

                if is_cjk and (is_vert_A or is_vert_B):
                    # Chữ viết dọc: xếp song song cạnh nhau theo phương ngang
                    gap_x = max(x0A, x0B) - min(x2A, x2B)
                    overlap_y = min(y2A, y2B) - max(y0A, y0B)
                    h_max = max(y2A - y0A, y2B - y0B)

                    # Gom các cột dọc khi khoảng cách ngang nhỏ và có sự trùng lặp chiều dọc
                    if gap_x < w_avg * 1.8 and (overlap_y > -20 or abs(cyA - cyB) < h_max * 0.5):
                        if dist < w_avg * 3.5:
                            union(i, j)
                else:
                    # Chữ viết ngang: sử dụng logic khoảng cách dọc và độ phủ ngang ban đầu
                    overlap_x = min(x2A, x2B) - max(x0A, x0B)
                    if dist < h_avg * 2.2 and overlap_x > -(w_avg * 0.4):
                        union(i, j)

    groups: dict = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(ocr_items[i])

    merged_items = []

    for g_idx, group in enumerate(groups.values()):
        # Xác định nhóm chữ này là viết dọc hay viết ngang
        is_group_vertical = False
        if is_cjk and len(group) > 0:
            vertical_count = sum(1 for item in group if (item["bbox"][3] - item["bbox"][1]) > (item["bbox"][2] - item["bbox"][0]))
            if vertical_count / len(group) > 0.5:
                is_group_vertical = True
            elif len(group) == 1:
                w = group[0]["bbox"][2] - group[0]["bbox"][0]
                h = group[0]["bbox"][3] - group[0]["bbox"][1]
                if h > w * 1.2:
                    is_group_vertical = True

        # Sắp xếp thứ tự đọc trong nhóm
        if is_cjk and is_group_vertical:
            # Chữ viết dọc (Manga Nhật): Đọc từ PHẢI qua TRÁI (giảm dần X2), rồi TRÊN xuống DƯỚI (tăng dần Y0)
            group.sort(key=lambda item: (-item["bbox"][2], item["bbox"][1]))
        else:
            # Chữ viết ngang: Đọc từ TRÊN xuống DƯỚI (Y0), rồi TRÁI qua PHẢI (X0)
            group.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))

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
