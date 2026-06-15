import requests
import re

chunks = [
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

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

for chunk in chunks:
    url = f"https://acfan.pro/_nuxt/{chunk}"
    print(f"Downloading {url}...")
    try:
        resp = requests.get(url, headers=headers)
        content = resp.text
        # Search for paths
        paths = re.findall(r'"(/api/[^"]+)"', content)
        # Also look for single quotes
        paths.extend(re.findall(r"'(/api/[^']+)'", content))
        # Look for templates like `/api/...`
        paths.extend(re.findall(r'`(/api/[^`]+)`', content))
        
        # Look for occurrences of comics or chapter
        interesting = [p for p in paths if "comics" in p or "chapter" in p or "detail" in p]
        if interesting:
            print(f"  [FOUND IN {chunk}]:")
            for p in sorted(list(set(interesting))):
                print(f"    {p}")
    except Exception as e:
        print(f"  Error {chunk}: {e}")
