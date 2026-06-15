import re

with open("DownloadTruyen/C3El_qzG.js", "r", encoding="utf-8") as f:
    content = f.read()

# Search for createObjectURL and print around it
matches = re.finditer(r'createObjectURL', content)
for i, m in enumerate(matches):
    start = max(0, m.start() - 250)
    end = min(len(content), m.end() + 250)
    snippet = content[start:end].replace("\n", " ")
    print(f"[{i:d}] ... {snippet} ...\n")
