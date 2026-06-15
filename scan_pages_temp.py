import os
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_textline_orientation=True, 
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    lang='ch', 
    device='cpu',
    enable_mkldnn=False
)

print("Starting scan...")
for i in range(1, 100):
    p = f"/workspace/DownloadTruyen/luutruyen/{i:03d}.jpg"
    if not os.path.exists(p):
        continue
    res = ocr.ocr(p)
    if res and res[0]:
        for line in res[0]:
            text = line[1][0]
            if '不是吧' in text or '魔头' in text or '前世' in text or '姬天道' in text:
                print(f"Match found in page {i:03d}: {text}", flush=True)
print("Scan completed.")
