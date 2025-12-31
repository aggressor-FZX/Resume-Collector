#!/usr/bin/env python3
"""Validate configured paid OpenRouter model IDs against the OpenRouter /models listing.

Usage: python scripts/validate_openrouter_paid_models.py

It will print which configured paid models are valid, and suggest replacements for invalid ones based on fuzzy matching of available model ids/names.
"""
import os
import sys
import requests
import difflib

# Try to import configured models from the main script
try:
    from generate_imaginator_data_free_tier import PAID_OPENROUTER_MODELS, OPENROUTER_API_BASE, OPENROUTER_API_KEY
except Exception as e:
    print('Warning: could not import generate_imaginator_data_free_tier (falling back to simple config).', e)
    PAID_OPENROUTER_MODELS = [
        "NousResearch/Hermes-4.3-36B",
        "TheDrummer/Rocinante-12B-v1.1",
        "deepseek-ai/DeepSeek-V3.2-Exp",
        "TheDrummer/Skyfall-36B-v2",
        "deepseek-ai/DeepSeek-V3.2-Speciale",
    ]
    OPENROUTER_API_BASE = os.getenv('OPENROUTER_API_BASE','https://openrouter.ai/api/v1')
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

if not OPENROUTER_API_KEY:
    print('ERROR: OPENROUTER_API_KEY not set in environment. Aborting.')
    sys.exit(2)

print('OpenRouter base:', OPENROUTER_API_BASE)
print('Configured paid OpenRouter models:')
for m in PAID_OPENROUTER_MODELS:
    print(' -', m)

# Fetch models
resp = requests.get(f"{OPENROUTER_API_BASE}/models", headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, timeout=30)
if resp.status_code != 200:
    print('Failed to fetch /models:', resp.status_code, resp.text[:500])
    sys.exit(1)

data = resp.json()
# The listing may be nested under 'data' or 'models', or be a raw list
models_list = None
if isinstance(data, dict) and 'data' in data:
    # OpenRouter returns {'data': [...]} in some versions
    if isinstance(data['data'], list):
        models_list = data['data']
    elif isinstance(data['data'], dict) and 'models' in data['data']:
        models_list = data['data']['models']
elif isinstance(data, dict) and 'models' in data:
    models_list = data['models']
elif isinstance(data, list):
    models_list = data
else:
    print('Unexpected /models response structure. Dumping top-level keys:', list(data.keys()) if isinstance(data, dict) else type(data))
    sys.exit(1)

available_ids = [m.get('id') or m.get('canonical_slug') or m.get('name') for m in models_list]
available_ids = [a for a in available_ids if a]

print('\nFound', len(available_ids), 'models on OpenRouter. Listing a sample:')
for a in available_ids[:30]:
    print(' -', a)

# Helper to find best match

def best_match(target, candidates):
    target_norm = target.lower()
    # exact
    for c in candidates:
        if c.lower() == target_norm:
            return c, 1.0
    # substring
    for c in candidates:
        if target_norm in c.lower() or c.lower() in target_norm:
            return c, 0.9
    # fuzzy
    match = difflib.get_close_matches(target, candidates, n=1, cutoff=0.6)
    if match:
        return match[0], 0.7
    # no match
    return None, 0.0

print('\nValidation results:')
replacements = {}
for m in PAID_OPENROUTER_MODELS:
    match, score = best_match(m, available_ids)
    if score >= 0.9:
        print(f"OK: {m} -> {match} (score={score})")
    elif score > 0:
        print(f"CLOSE: {m} -> {match} (score={score})")
        replacements[m] = match
    else:
        print(f"MISSING: {m} -> NO MATCH FOUND")

if replacements:
    print('\nSuggestions: Replace the following configured models with the suggested candidates:')
    for k,v in replacements.items():
        print(f" - {k}  ->  {v}")
    print('\nIf a model is MISSING, it may be unavailable or named differently in your OpenRouter account. Consider removing it from PAID_OPENROUTER_MODELS or checking account entitlements.')

print('\nDone.')
