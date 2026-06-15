import re

with open("DownloadTruyen/C3El_qzG.js", "r", encoding="utf-8") as f:
    content = f.read()

# Search for "Fetch XOR failed"
matches = re.finditer(r'Fetch XOR failed', content)
for i, m in enumerate(matches):
    start = max(0, m.start() - 500)
    end = min(len(content), m.end() + 500)
    snippet = content[start:end].replace("\n", " ")
    print(f"[{i:d}] ... {snippet} ...\n")
