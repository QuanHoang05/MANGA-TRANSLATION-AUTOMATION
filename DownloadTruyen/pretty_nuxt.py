import json
import re

with open("DownloadTruyen/temp_page.html", "r", encoding="utf-8") as f:
    html = f.read()

# Find the script block
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

# Pretty print the array to a text file
with open("DownloadTruyen/pretty_nuxt.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Pretty printed Nuxt data to DownloadTruyen/pretty_nuxt.json")
