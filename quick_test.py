"""Upload test images and poll results"""
import requests, json, os, time, sys

def read_api_key():
    with open('data/tests/linkAPIggStudio', 'r') as f:
        return f.read().strip()

def upload_images(api_key, image_names):
    files = []
    for img in image_names:
        p = os.path.join('data', 'tests', img)
        files.append(('files', (img, open(p,'rb'), 'image/jpeg')))
    
    data = {
        'api_key': api_key,
        'src_lang': 'ch',
        'tone': 'tu nhien',
        'batch_size_pages': str(len(image_names)),
    }
    
    resp = requests.post('http://localhost:8000/api/upload', files=files, data=data, timeout=30)
    for _, (name, f, _) in files:
        f.close()
    return resp.json()['job_id']

def poll_job(job_id, timeout=300):
    start = time.time()
    resp = requests.get(
        f'http://localhost:8000/api/stream-progress?job_id={job_id}',
        stream=True, timeout=timeout
    )
    for line in resp.iter_lines():
        if time.time() - start > timeout:
            print('TIMEOUT!')
            break
        if line:
            txt = line.decode('utf-8')
            if txt.startswith('data:'):
                try:
                    data = json.loads(txt[5:])
                    pct = data.get('progress', 0)
                    log = data.get('log', '')
                    status = data.get('status', '')
                    print(f'  [{pct:5.1f}%] {log}')
                    if status in ('completed', 'failed'):
                        return data
                except Exception as pe:
                    print(f'  Parse error: {pe}')
    return None

def main():
    api_key = read_api_key()
    
    # Test voi ca Page 1, 2, 3, 4, 5 de bao phu day du cac loai bubble
    imgs = [f'{i}.jpg' for i in range(2, 31)]
    print(f'[*] Upload {len(imgs)} anh...')
    job_id = upload_images(api_key, imgs)
    print(f'[*] Job ID: {job_id}')
    print('[*] Polling...')
    result = poll_job(job_id)
    print(f'[*] Job ID de copy anh: {job_id}')

if __name__ == '__main__':
    main()
