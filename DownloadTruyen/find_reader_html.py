import re

with open("DownloadTruyen/temp_page.html", "r", encoding="utf-8") as f:
    html = f.read()

# Search for divs with classes containing reader, comic, read, or detail
divs = re.findall(r'<div[^>]+class="([^"]*)"[^>]*>', html)
reader_divs = [d for d in divs if "reader" in d or "comic" in d or "read" in d]

print("Reader related divs in HTML:")
for d in set(reader_divs):
    print(f"  {d}")

# Also look for any img tags in the HTML
imgs = re.findall(r'<img[^>]+src="([^"]+)"', html)
print("\nImages found in HTML:")
for img in imgs:
    print(f"  {img}")
