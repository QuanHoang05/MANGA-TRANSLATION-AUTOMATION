import requests
import json

base_url = "https://acpc.wamd.fun/api/comics/base/public/chapterInfo"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Origin": "https://acfan.pro",
    "Referer": "https://acfan.pro/"
}

# Try GET with id
url1 = f"{base_url}?id=683972"
print(f"Testing GET: {url1}")
try:
    resp = requests.get(url1, headers=headers)
    print(f"Status: {resp.status_code}")
    print(resp.text[:500])
except Exception as e:
    print(f"Error: {e}")

# Try GET with chapterId
url2 = f"{base_url}?chapterId=683972"
print(f"\nTesting GET: {url2}")
try:
    resp = requests.get(url2, headers=headers)
    print(f"Status: {resp.status_code}")
    print(resp.text[:500])
except Exception as e:
    print(f"Error: {e}")

# Try POST with json body
print(f"\nTesting POST with JSON id=683972")
try:
    resp = requests.post(base_url, headers=headers, json={"id": 683972})
    print(f"Status: {resp.status_code}")
    print(resp.text[:500])
except Exception as e:
    print(f"Error: {e}")

print(f"\nTesting POST with JSON chapterId=683972")
try:
    resp = requests.post(base_url, headers=headers, json={"chapterId": 683972})
    print(f"Status: {resp.status_code}")
    print(resp.text[:500])
except Exception as e:
    print(f"Error: {e}")
