import requests
import re
import json

url = "https://acfan.pro/comics/read/34843?chapter=1"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print(f"Fetching {url}...")
resp = requests.get(url, headers=headers)
html = resp.text
print(f"Status: {resp.status_code}, Length: {len(html)}")

# Ghi ra file tam de check
with open("DownloadTruyen/temp_page.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Searching for Nuxt state or image lists in script tags...")
# Tim script chua NUXT_DATA hoac NUXT
for m in re.finditer(r'<script[^>]*>(.*?)</script>', html, re.DOTALL):
    content = m.group(1)
    if "__NUXT__" in content or "NUXT_DATA" in content or "images" in content or "chapter" in content:
        print(f"\n--- Found interesting script (length: {len(content)}): ---")
        print(content[:1000] + "...")
