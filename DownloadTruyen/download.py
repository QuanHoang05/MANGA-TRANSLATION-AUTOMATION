import os
import sys
import re
import json
import requests
from urllib.parse import urlparse, parse_qs

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def get_chapter_info(url):
    """
    Truy cập trang đọc truyện, phân tích dữ liệu Nuxt để lấy chapterId,
    sau đó gọi API để lấy danh sách ảnh truyện.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Origin": "https://acfan.pro",
        "Referer": "https://acfan.pro/"
    }
    
    # 1. Lấy thông tin chapter từ query parameter trong URL
    parsed_url = urlparse(url)
    queries = parse_qs(parsed_url.query)
    target_chapter_str = queries.get("chapter", ["1"])[0]
    try:
        target_chapter = int(target_chapter_str)
    except ValueError:
        target_chapter = 1
        
    print(f"[*] Đang tải trang đọc truyện: {url}")
    print(f"[*] Chương mục tiêu: {target_chapter}")
    
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Không thể tải trang truyện. Status code: {resp.status_code}")
        
    html = resp.text
    
    # 2. Tìm thẻ script chứa dữ liệu __NUXT_DATA__
    match = re.search(r'<script type="application/json"[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if not match:
        # Tìm dự phòng trong các thẻ script thường
        scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
        content = ""
        for s in scripts:
            if "chapterList" in s or "chapterId" in s:
                content = s
                break
    else:
        content = match.group(1)
        
    if not content:
        raise Exception("Không tìm thấy dữ liệu cấu trúc Nuxt trong HTML.")
        
    try:
        data = json.loads(content.strip())
    except Exception as je:
        raise Exception(f"Lỗi phân tích cú pháp JSON Nuxt: {je}")
        
    # 3. Phân tích dữ liệu Nuxt để tìm chapterId tương ứng với target_chapter
    chapter_id = None
    for item in data:
        if isinstance(item, dict) and "chapterId" in item and "chapterNum" in item:
            cid_idx = item["chapterId"]
            cnum_idx = item["chapterNum"]
            
            # Giải mã chỉ số tham chiếu trong mảng data
            if isinstance(cid_idx, int) and 0 <= cid_idx < len(data):
                cid_val = data[cid_idx]
            else:
                continue
                
            if isinstance(cnum_idx, int) and 0 <= cnum_idx < len(data):
                cnum_val = data[cnum_idx]
            else:
                continue
                
            if cnum_val == target_chapter:
                chapter_id = cid_val
                break
                
    if not chapter_id:
        # Tìm phương án dự phòng (lấy chapter đầu tiên tìm thấy)
        print("[WARN] Không tìm thấy chapter khớp chính xác số chương. Thử lấy chapterId đầu tiên có trong dữ liệu...")
        for item in data:
            if isinstance(item, dict) and "chapterId" in item:
                cid_idx = item["chapterId"]
                if isinstance(cid_idx, int) and 0 <= cid_idx < len(data):
                    chapter_id = data[cid_idx]
                    print(f"[OK] Đã chọn chapterId dự phòng: {chapter_id}")
                    break
                    
    if not chapter_id:
        raise Exception("Không tìm thấy chapterId hợp lệ trong dữ liệu Nuxt.")
        
    print(f"[OK] Đã tìm thấy chapterId: {chapter_id}")
    
    # 4. Gọi API của acfan.pro để lấy danh sách ảnh
    api_url = f"https://acfan.pro/api/comics/base/public/chapterInfo?chapterId={chapter_id}"
    print(f"[*] Đang gọi API lấy danh sách ảnh: {api_url}")
    api_resp = requests.get(api_url, headers=headers)
    
    if api_resp.status_code != 200:
        raise Exception(f"API trả về mã lỗi: {api_resp.status_code}")
        
    api_data = api_resp.json()
    if api_data.get("code") != 200:
        raise Exception(f"API báo lỗi: {api_data.get('msg')}")
        
    data_res = api_data.get("data", {})
    domain = data_res.get("domain", "")
    img_list = data_res.get("imgList", [])
    
    if not img_list:
        raise Exception("Danh sách ảnh trả về từ API trống.")
        
    return domain, img_list

def decrypt_content(content):
    """
    Giải mã nội dung ảnh sử dụng XOR với key '2020-zq3-888' cho 100 byte đầu tiên.
    """
    key = b"2020-zq3-888"
    data = bytearray(content)
    for i in range(min(100, len(data))):
        data[i] ^= key[i % len(key)]
    return bytes(data)

def is_image_valid(filepath):
    """
    Kiểm tra xem file ảnh có định dạng hợp lệ hay chưa (đã được giải mã).
    """
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return False
        with open(filepath, "rb") as f:
            head = f.read(4)
            # JPEG: \xff\xd8, WEBP: RIFF, PNG: \x89PNG, GIF: GIF8
            if head.startswith(b"\xff\xd8") or head.startswith(b"RIFF") or head.startswith(b"\x89PNG") or head.startswith(b"GIF8"):
                return True
    except Exception:
        pass
    return False

def download_images(domain, img_list, output_dir):
    """
    Tải danh sách ảnh về thư mục kết quả và giải mã XOR.
    """
    os.makedirs(output_dir, exist_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://acfan.pro/"
    }
    
    total = len(img_list)
    print(f"\n[*] Bắt đầu tải {total} ảnh về thư mục: {os.path.abspath(output_dir)}")
    
    # Sử dụng Session để giữ kết nối HTTP/Keep-Alive giúp tăng tốc độ tải
    session = requests.Session()
    session.headers.update(headers)
    
    for idx, img_path in enumerate(img_list):
        img_url = domain + img_path
        ext = os.path.splitext(img_path)[1]
        if not ext:
            ext = ".jpg"
            
        filename = f"{idx+1:03d}{ext}"
        filepath = os.path.join(output_dir, filename)
        
        # Bỏ qua nếu ảnh đã tải xong trước đó và hợp lệ
        if is_image_valid(filepath):
            print(f"  [{idx+1}/{total}] Đã tồn tại & hợp lệ: {filename} (Bỏ qua)")
            continue
            
        print(f"  [{idx+1}/{total}] Đang tải & giải mã: {filename}...", end="", flush=True)
        try:
            r = session.get(img_url, timeout=20)
            if r.status_code == 200:
                decrypted = decrypt_content(r.content)
                with open(filepath, "wb") as f:
                    f.write(decrypted)
                print(" OK")
            else:
                print(f" LỖI (Status: {r.status_code})")
        except Exception as e:
            print(f" LỖI ({e})")
            
    print("\n[OK] Đã hoàn thành tải toàn bộ chương truyện!")

def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "https://acfan.pro/comics/read/34843?chapter=1"
    output_dir = os.path.join("DownloadTruyen", "luutruyen")
    
    try:
        domain, img_list = get_chapter_info(url)
        download_images(domain, img_list, output_dir)
    except Exception as e:
        print(f"\n[LỖI NGHIÊM TRỌNG]: {e}")

if __name__ == "__main__":
    main()
