#!/usr/bin/env python3
"""Quick one-shot test for a set of OpenRouter models using current OPENROUTER_API_KEY.
"""
import os
import requests
from time import sleep

set_vars = False
try:
    from dotenv import load_dotenv
    load_dotenv()
    set_vars = True
except Exception:
    pass

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_BASE = os.getenv('OPENROUTER_API_BASE', 'https://openrouter.ai/api/v1')

models = [
    'deepseek/deepseek-r1-distill-qwen-32b',
    'meta-llama/llama-3.3-70b-instruct',
    'qwen/qwen2.5-72b-instruct',
    'nousresearch/hermes-4-70b',
    'thedrummer/rocinante-12b',
    'deepseek/deepseek-v3.2',
    'thedrummer/skyfall-36b-v2',
    'deepseek/deepseek-v3.2-speciale',
]

headers = {'Authorization': f'Bearer {OPENROUTER_API_KEY}', 'X-Title': 'ResumeCollectorImaginator', 'HTTP-Referer': 'https://github.com/aggressor-FZX/Resume-Collector'}

prompt = [{"role": "user", "content": "Create a very short JSON object: {\"test\":\"ok\"}. Respond only with JSON."}]

for m in models:
    print('\n--- Testing', m)
    try:
        r = requests.post(f"{OPENROUTER_API_BASE}/chat/completions", headers=headers, json={"model": m, "messages": prompt, "response_format": {"type": "json_object"}}, timeout=30)
        print('Status:', r.status_code)
        try:
            print('JSON:', r.json())
        except Exception:
            print('Text:', r.text[:1000])
    except Exception as e:
        print('Exception:', e)
    sleep(0.3)

print('\nDone')
