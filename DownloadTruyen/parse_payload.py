import re
import json

with open("DownloadTruyen/temp_page.html", "r", encoding="utf-8") as f:
    html = f.read()

# Find the script block containing the big JSON list
# In Nuxt 3, it is typically in a <script type="application/json" id="__NUXT_DATA__">... or similar
match = re.search(r'<script type="application/json"[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
if not match:
    # If not found with type, search for a large JSON block starting with [
    match = re.search(r'<script[^>]*>(.*?)</script>', html, re.DOTALL) # let's search all scripts
    # Find the one that starts with [[ or containing the big list
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
    content = ""
    for s in scripts:
        if "少爷的替身" in s:
            content = s
            break
else:
    content = match.group(1)

print(f"Content length: {len(content)}")

# If the content is JSON, load it
try:
    # Nuxt 3 payload is valid JSON
    data = json.loads(content.strip())
    print("Successfully parsed Nuxt payload as JSON!")
    print(f"Number of elements in array: {len(data)}")
    
    # Print all elements that look like domains or image paths
    urls = []
    domains = []
    images = []
    
    for item in data:
        if isinstance(item, str):
            if item.startswith("http"):
                domains.append(item)
            elif item.endswith((".jpg", ".jpeg", ".png", ".webp")):
                images.append(item)
                
    print("\n--- DOMAINS FOUND ---")
    for d in domains:
        print(f"  {d}")
        
    print(f"\n--- IMAGES FOUND ({len(images)}) ---")
    for img in images[:30]:
        print(f"  {img}")
    if len(images) > 30:
        print(f"  ... and {len(images) - 30} more")
        
except Exception as e:
    print(f"Failed to parse as JSON: {e}")
    # Just search using regex
    print("\n--- Regex Search for .jpg/png/webp ---")
    jpgs = re.findall(r'"([^"]+\.(?:jpg|png|webp))"', content)
    for j in set(jpgs):
        print(f"  {j}")
