import requests
import json
import time
import sys

job_id = sys.argv[1] if len(sys.argv) > 1 else 'b3a0751b-6016-4e6d-bee3-b137eb4ff143'
print(f'Theo doi job: {job_id}')

try:
    resp = requests.get(
        f'http://localhost:8000/api/stream-progress?job_id={job_id}',
        stream=True, timeout=300
    )
    for line in resp.iter_lines():
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
                        print(f'\n=== HOAN THANH: {status} ===')
                        ocr = data.get('ocr_results')
                        trans = data.get('translated_results')
                        if ocr:
                            print(f'OCR: {len(ocr)} items')
                            for item in ocr[:3]:
                                print(f'  [{item["id"]}] {item["text"][:50]}')
                        if trans:
                            print(f'DICH: {len(trans)} items')
                            for k, v in list(trans.items())[:3]:
                                print(f'  [{k}] => {v[:50]}')
                        break
                except Exception as pe:
                    print(f'  Parse error: {pe} | raw: {txt[:100]}')
except Exception as e:
    print(f'Loi ket noi: {e}')
