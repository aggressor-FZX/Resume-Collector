#!/usr/bin/env python3
"""Debug script: send test requests to OpenRouter for paid, HF-like, and free models and print full response diagnostics."""
import os
import requests
# Avoid load_dotenv here to prevent assertion issues in some environments; rely on environment variables already set in the CI/container.
import os
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

# If the environment does not already have the key, try loading .env for convenience during local debugging
if not OPENROUTER_API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    except Exception:
        pass

# Import paid models from the main script to ensure canonical values
try:
    from generate_imaginator_data_free_tier import PAID_OPENROUTER_MODELS
    models = list(PAID_OPENROUTER_MODELS) + ["nex-agi/deepseek-v3.1-nex-n1:free"]
except Exception:
    models = [
        "deepseek/deepseek-r1-distill-qwen-32b",
        "meta-llama/llama-3.3-70b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "nousresearch/hermes-4-70b",
        "thedrummer/rocinante-12b",
        "deepseek/deepseek-v3.2",
        "thedrummer/skyfall-36b-v2",
        "deepseek/deepseek-v3.2-speciale",
        "nex-agi/deepseek-v3.1-nex-n1:free",
    ]

prompt = [{"role": "user", "content": "Create a very short JSON object: {\"test\":\"ok\"}. Respond only with JSON."}]

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://github.com/aggressor-FZX/Resume-Collector",
    "X-Title": "ResumeCollectorImaginator",
}

for model in models:
    print("\n--- Testing model:", model)
    try:
        r = requests.post(
            url=f"{OPENROUTER_API_BASE}/chat/completions",
            headers=headers,
            json={"model": model, "messages": prompt, "response_format": {"type": "json_object"}},
            timeout=30,
        )
        print("Status:", r.status_code)
        print("Headers:\n", r.headers)
        try:
            body = r.json()
            print("Body JSON:\n", body)
            # Helpful note for 404s that indicate privacy/data policy blocks
            if r.status_code == 404 and isinstance(body, dict):
                err = body.get('error', {})
                msg = err.get('message') or body.get('message')
                if msg and 'No endpoints found matching your data policy' in msg:
                    print('\nNOTE: This model appears to require additional account/data-privacy settings to be enabled in your OpenRouter account. See https://openrouter.ai/settings/privacy for details.')
        except Exception:
            print("Body text:\n", r.text[:2000])
    except Exception as e:
        print("Request failed:", e)
