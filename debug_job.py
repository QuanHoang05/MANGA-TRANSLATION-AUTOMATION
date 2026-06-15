import requests, json, sys

job_id = sys.argv[1] if len(sys.argv) > 1 else '4c3e94a5-966d-4bcf-93fc-62ff0f3dea48'
resp = requests.get(f'http://localhost:8000/api/stream-progress?job_id={job_id}', stream=True, timeout=10)
for line in resp.iter_lines():
    if line:
        txt = line.decode('utf-8')
        if txt.startswith('data:'):
            data = json.loads(txt[5:])
            ocr = data.get('ocr_results', [])
            trans = data.get('translated_results', {})
            if ocr:
                print('=== OCR ITEMS ===')
                for item in ocr:
                    iid = item.get('id', '')
                    itxt = item.get('text', '')
                    print(f'  [{iid}] {itxt}')
                print()
                print('=== TRANSLATIONS ===')
                for k,v in (trans or {}).items():
                    print(f'  [{k}] => {v}')
            break
