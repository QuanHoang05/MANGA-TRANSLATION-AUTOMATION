import json

with open("DownloadTruyen/temp_page.html", "r", encoding="utf-8") as f:
    html = f.read()

import re
match = re.search(r'<script type="application/json"[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
if not match:
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
    content = ""
    for s in scripts:
        if "少爷的替身" in s:
            content = s
            break
else:
    content = match.group(1)

data = json.loads(content.strip())

print("All strings in Nuxt Data:")
for i, item in enumerate(data):
    if isinstance(item, str):
        # Print strings, especially if they look like paths or codes
        if "/" in item or "api" in item or "chapter" in item or len(item) > 10:
            print(f"[{i:3d}] {item}")
