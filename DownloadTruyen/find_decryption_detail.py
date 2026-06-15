import re

with open("DownloadTruyen/C3El_qzG.js", "r", encoding="utf-8") as f:
    content = f.read()

# Find occurrences of charCodeAt, ^, Uint8Array or decrypt
# Let's search for functions that do charCodeAt and look for XOR operations
print("Searching for charCodeAt occurrences:")
matches = re.finditer(r'(\w+\.charCodeAt\([^)]*\))', content)
for i, m in enumerate(matches):
    start = max(0, m.start() - 150)
    end = min(len(content), m.end() + 150)
    snippet = content[start:end].replace("\n", " ")
    print(f"[{i:d}] ... {snippet} ...\n")
    if i > 20:
        print("... truncated")
        break
