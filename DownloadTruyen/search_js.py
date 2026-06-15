import requests
import re

url = "https://acfan.pro/_nuxt/C3El_qzG.js"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print(f"Downloading {url}...")
resp = requests.get(url, headers=headers)
js_content = resp.text
print(f"Downloaded JS length: {len(js_content)}")

# Search for potential api paths in the JS
# Typical patterns: "/api/..." or specific words like "chapter" or "comics"
paths = re.findall(r'"(/[^"]+)"', js_content)
api_paths = [p for p in paths if "api" in p or "chapter" in p or "comics" in p]

print("\nPossible API paths found in JS:")
for p in sorted(list(set(api_paths))):
    print(f"  {p}")
