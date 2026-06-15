import re

with open("DownloadTruyen/temp_page.html", "r", encoding="utf-8") as f:
    html = f.read()

scripts = re.findall(r'<script[^>]+src="([^"]+)"', html)
print("Scripts loaded by HTML:")
for s in scripts:
    print(f"  {s}")
