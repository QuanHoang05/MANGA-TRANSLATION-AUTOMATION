import re
with open('app/static/js/app.js', 'r', encoding='utf-8') as f:
    c = f.read()

c = re.sub(r'statusBadge\.textContent = "TH\?T B\?I";', 'statusBadge.textContent = "THẤT BẠI";', c)
c = re.sub(r'statusBadge\.textContent = "TH.+T B.+I";', 'statusBadge.textContent = "THẤT BẠI";', c)
c = re.sub(r'statusBadge\.textContent = "HOA\?N THA\?NH";', 'statusBadge.textContent = "HOÀN THÀNH";', c)
c = re.sub(r'statusBadge\.textContent = ".*ANG X.* LA\?";', 'statusBadge.textContent = "ĐANG XỬ LÝ";', c)

with open('app/static/js/app.js', 'w', encoding='utf-8') as f:
    f.write(c)
print("done")
