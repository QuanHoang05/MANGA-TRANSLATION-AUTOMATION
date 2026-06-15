import re

with open("DownloadTruyen/temp_page.html", "r", encoding="utf-8") as f:
    html = f.read()

# Find modulepreload links
preloads = re.findall(r'href="(/_nuxt/[^"]+\.js)"', html)
print("Preload chunks found:")
for p in preloads:
    print(f"  {p}")
