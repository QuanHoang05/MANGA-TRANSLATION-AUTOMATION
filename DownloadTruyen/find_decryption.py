import os
import re

chunk_path = "DownloadTruyen/DK7M7gbV.js"
if not os.path.exists(chunk_path):
    # Download it if not exists
    import requests
    url = f"https://acfan.pro/_nuxt/DK7M7gbV.js"
    print(f"Downloading {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    with open(chunk_path, "w", encoding="utf-8") as f:
        f.write(resp.text)

with open(chunk_path, "r", encoding="utf-8") as f:
    js_content = f.read()

print(f"JS content length: {len(js_content)}")

# Look for occurrences of word 'Uint8Array', 'ArrayBuffer', charCodeAt, decrypt, XOR, or '^'
print("\nSearching for binary/array buffer usage:")
matches = re.finditer(r'(.{0,100}(?:ArrayBuffer|Uint8Array|charCodeAt|Uint8ClampedArray|Blob|URL\.createObjectURL|decrypt|responseType).{0,100})', js_content)
for i, m in enumerate(matches):
    print(f"[{i:d}] {m.group(1).strip()}")
    if i > 50:
        print("... truncated")
        break
