import requests
import re
import os

chunks = [
    "C3El_qzG.js",
    "C-n5trzf.js",
    "Dang-ZOx.js",
    "CNMzG-O2.js",
    "DI2tV9j7.js",
    "CP81-TBh.js",
    "DbcQa6ds.js",
    "Co4UvDXP.js",
    "DK7M7gbV.js",
    "RfDLajVv.js",
    "Dri--Ywq.js"
]

headers = {"User-Agent": "Mozilla/5.0"}

for chunk in chunks:
    chunk_path = f"DownloadTruyen/{chunk}"
    if not os.path.exists(chunk_path):
        url = f"https://acfan.pro/_nuxt/{chunk}"
        resp = requests.get(url, headers=headers)
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
            
    with open(chunk_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Search for keywords
    keywords = ["Uint8Array", "ArrayBuffer", "charCodeAt", "createObjectURL", "responseType", "XOR", "decrypt"]
    found = []
    for kw in keywords:
        if kw in content:
            found.append(kw)
            
    if found:
        print(f"[FOUND IN {chunk}]: {found}")
        # Print a snippet of where Uint8Array or createObjectURL is used
        matches = re.finditer(r'(.{0,70}(?:Uint8Array|createObjectURL|charCodeAt).{0,70})', content)
        for i, m in enumerate(matches):
            print(f"    - {m.group(1).strip()}")
            if i > 5:
                print("    - ... truncated")
                break
