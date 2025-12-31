#!/usr/bin/env python3
"""Run a rotating test for paid OpenRouter models for a given duration (default 60s).
Logs per-request diagnostics and prints a final summary.
"""
import time
import os
import requests
from itertools import cycle
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

# Paid models list - keep consistent with generate_imaginator_data_free_tier.PAID_OPENROUTER_MODELS
PAID_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "NousResearch/Hermes-4.3-36B",
    "TheDrummer/Rocinante-12B-v1.1",
    "deepseek-ai/DeepSeek-V3.2-Exp",
    "TheDrummer/Skyfall-36B-v2",
    "deepseek-ai/DeepSeek-V3.2-Speciale",
]

DURATION = int(os.getenv("PAID_TEST_DURATION", "60"))  # seconds
DELAY = float(os.getenv("PAID_TEST_DELAY", "1.0"))   # seconds between requests
LOG_PATH = os.getenv("PAID_TEST_LOG", "logs/paid_openrouter_test.log")

prompt = [{"role": "user", "content": "Create a very short JSON object: {\"test\":\"ok\"}. Respond only with JSON."}]
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://github.com/aggressor-FZX/Resume-Collector",
    "X-Title": "ResumeCollectorImaginator",
}

if not OPENROUTER_API_KEY:
    print("OPENROUTER_API_KEY not set. Aborting test.")
    raise SystemExit(1)

cycler = cycle(PAID_MODELS)
end_time = time.time() + DURATION

stats = {m: {"count": 0, "ok": 0, "errors": []} for m in PAID_MODELS}

with open(LOG_PATH, "w", encoding="utf-8") as logf:
    logf.write(f"Starting paid OpenRouter test for {DURATION}s, delay={DELAY}s\n")
    while time.time() < end_time:
        model = next(cycler)
        t0 = time.time()
        try:
            r = requests.post(
                url=f"{OPENROUTER_API_BASE}/chat/completions",
                headers=headers,
                json={"model": model, "messages": prompt, "response_format": {"type": "json_object"}},
                timeout=30,
            )
            status = r.status_code
            body_text = None
            try:
                body = r.json()
                body_text = str(body)
            except Exception:
                body_text = r.text[:2000]

            stats[model]["count"] += 1
            if 200 <= status < 300:
                stats[model]["ok"] += 1
                logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {model} | {status} | OK | {body_text}\n")
            else:
                stats[model]["errors"].append((status, body_text))
                logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {model} | {status} | ERROR | {body_text[:2000]}\n")
        except Exception as e:
            stats[model]["count"] += 1
            stats[model]["errors"].append((None, str(e)))
            logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {model} | EXC | ERROR | {e}\n")
        # Flush to disk so we can inspect while running
        logf.flush()
        # Respect a small delay to avoid immediate hammering
        elapsed = time.time() - t0
        to_sleep = max(0, DELAY - elapsed)
        time.sleep(to_sleep)

# Print summary
print("--- Paid OpenRouter Test Summary ---")
for m in PAID_MODELS:
    s = stats[m]
    print(f"{m}: attempts={s['count']}, successes={s['ok']}, errors={len(s['errors'])}")
print(f"Detailed log written to {LOG_PATH}")
