#!/usr/bin/env python3
"""Annotate ambiguous samples using meta-llama/llama-3.2-3b-instruct via OpenRouter.

Usage: python scripts/annotate_with_openrouter.py --input data/qc_samples/short_samples.jsonl --out data/annotated_samples.jsonl --low data/low_confidence.jsonl --limit 10

Behavior:
- Loads jsonl input of examples (expects objects with 'content' or 'messages')
- For each sample sends a chat completion request to OpenRouter model 'meta-llama/llama-3.2-3b-instruct'
- Tries OPENROUTER_API_KEY then OPENROUTER_API_KEY_2 if first fails
- Expects model to return a JSON object; falls back to heuristic parsing when necessary
- Saves annotated results and a separate low-confidence file
"""
import os
import time
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import requests

load_dotenv(find_dotenv())

# Use environment variables (Codespaces Secrets recommended) and do NOT hardcode keys.
OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_KEY_2 = os.getenv('OPENROUTER_API_KEY_2')
API_URL = 'https://openrouter.ai/api/v1/chat/completions'
# Model name is case-sensitive on OpenRouter
MODEL = 'meta-llama/llama-3.2-3b-instruct'

# Build headers for a request; additional headers may be added for identification
def make_headers(key: str, referer: str | None = None, x_title: str | None = None) -> dict:
    h = {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json'
    }
    if referer:
        h['HTTP-Referer'] = referer
    if x_title:
        h['X-Title'] = x_title
    return h

# Default safety / rate limits
DEFAULT_RATE_LIMIT = 1.0  # requests per second
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5
DEFAULT_PAUSE = 1.0  # seconds between calls by default

# Conservative token estimation factor (chars -> tokens)
TOKENS_PER_CHAR = 1.0 / 4.0

PROMPT_SYSTEM = (
    "You are a world-class tech resume writer.\n"
    "Rewrite the user resume bullet into an achievement-oriented, professional bullet. "
    "Remove marketplace fluff (e.g., 'Level 2 Seller', 'Top Rated Seller', '1k+'), "
    "quantify conservatively when possible, and be concise.\n"
    "Return a JSON object only, with fields: 'rewrite' (string), 'confidence' (0.0-1.0), 'note' (brief string explaining uncertainty if any).\n"
    "If you are unsure or cannot produce a high-quality rewrite, set confidence < 0.6 and explain why in 'note'."
)


def call_openrouter(messages, api_keys, referer=None, x_title=None, max_retries=DEFAULT_RETRIES, max_tokens=256, temperature=0.0, rate_limit=DEFAULT_RATE_LIMIT, backoff_factor=DEFAULT_BACKOFF_FACTOR, dry_run=False):
    """Call OpenRouter chat endpoint using provided list of api_keys (tries keys + retries with backoff).

    Returns (response_json, error_string, meta) where meta contains estimated input tokens.
    If dry_run=True the function returns a simulated response and does not perform network calls.
    """
    payload = {
        'model': MODEL,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature
    }

    # estimate input tokens for cost tracking
    total_input_text = ''.join(m.get('content','') for m in messages)
    est_input_tokens = max(1, int(len(total_input_text) * TOKENS_PER_CHAR))

    if dry_run:
        # Simulate a successful JSON response structure
        simulated = {'choices': [{'message': {'content': json.dumps({'rewrite':'[simulated rewrite]','confidence':0.9,'note':'dry_run'})}}]}
        return simulated, None, {'est_input_tokens': est_input_tokens}

    last_err = None
    for key in api_keys:
        if not key:
            continue
        # per-key retry loop
        for attempt in range(1, max_retries + 1):
            try:
                headers = make_headers(key, referer=referer, x_title=x_title)
                resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    return resp.json(), None, {'est_input_tokens': est_input_tokens}
                else:
                    last_err = f'HTTP {resp.status_code}: {resp.text}'
            except requests.exceptions.RequestException as e:
                last_err = str(e)
            # exponential backoff sleep
            sleep_time = backoff_factor * (2 ** (attempt - 1))
            time.sleep(sleep_time)
        # if key exhausted tries, move to next key
    return None, last_err, {'est_input_tokens': est_input_tokens}


def prepare_messages(sample_text: str):
    return [
        {'role': 'system', 'content': PROMPT_SYSTEM},
        {'role': 'user', 'content': sample_text}
    ]


def parse_model_output(text: str):
    # try to parse JSON out of model output
    text = (text or '').strip()
    # sometimes model puts markdown or code fences; extract inner fenced content when present
    if text.startswith('```'):
        # strip outer fences and get inner content
        try:
            parts = text.split('```')
            if len(parts) >= 3:
                # parts[1] is the content between first pair of fences
                text = parts[1].strip()
            else:
                text = parts[-1].strip()
        except Exception:
            pass
    try:
        obj = json.loads(text)
        return obj, None
    except Exception:
        # attempt to find json substring
        import re
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj, None
            except Exception as e:
                return None, f'json_extract_failed: {e}'
        else:
            return None, 'no_json_found'


def select_samples(input_path: Path, limit: int = None):
    arr = []
    with input_path.open('r', encoding='utf-8') as fh:
        for i, ln in enumerate(fh):
            if limit and i >= limit:
                break
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            # supports both {'content':...} and {'messages':...}
            if 'content' in obj and obj['content']:
                text = obj['content']
            elif 'messages' in obj:
                # try to pull user message
                msgs = obj['messages']
                text = ''
                for m in msgs:
                    if m.get('role') == 'user':
                        text = m.get('content','')
                        break
            else:
                text = ''
            if not text:
                continue
            arr.append({'orig': obj, 'text': text})
    return arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/qc_samples/short_samples.jsonl')
    parser.add_argument('--out', type=str, default='data/annotated_samples.jsonl')
    parser.add_argument('--low', type=str, default='data/low_confidence.jsonl')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--pause', type=float, default=DEFAULT_PAUSE)
    parser.add_argument('--budget', type=float, default=4.0, help='USD budget cap; stop when cumulative cost meets or exceeds this')
    parser.add_argument('--input-price-per-million', type=float, default=0.02, help='USD per 1M input tokens')
    parser.add_argument('--output-price-per-million', type=float, default=0.02, help='USD per 1M output tokens')
    parser.add_argument('--dry-run', action='store_true', help='Simulate calls without performing network requests')
    args = parser.parse_args()

    api_keys = [OPENROUTER_KEY, OPENROUTER_KEY_2]
    if not any(api_keys):
        print('No OpenRouter keys found in env (OPENROUTER_API_KEY / OPENROUTER_API_KEY_2). Aborting.')
        raise SystemExit(1)

    input_path = Path(args.input)
    out_path = Path(args.out)
    low_path = Path(args.low)

    samples = select_samples(input_path, limit=args.limit)
    print('Selected', len(samples), 'samples from', input_path)

    # budget tracking
    budget_usd = float(args.budget)
    input_price_per_token = float(args.input_price_per_million if False else args.input_price_per_million if False else args.input_price_per_million) if False else (args.input_price_per_million / 1_000_000.0)
    output_price_per_token = (args.output_price_per_million / 1_000_000.0)

    def est_tokens_from_text(s: str) -> int:
        # conservative estimate: 1 token ~= 4 characters
        return max(1, int(len(s) / 4))

    cumulative_cost = 0.0
    cumulative_input_tokens = 0
    cumulative_output_tokens = 0

    with out_path.open('w', encoding='utf-8') as fout, low_path.open('w', encoding='utf-8') as flow:
        for i, s in enumerate(samples, start=1):
            raw_text = s['text']
            messages = prepare_messages(raw_text)

            # estimate input tokens from system+user messages
            system_text = PROMPT_SYSTEM
            user_text = raw_text
            est_input_tokens = est_tokens_from_text(system_text) + est_tokens_from_text(user_text)
            est_input_cost = est_input_tokens * input_price_per_token

            # if running this call would exceed budget, stop before calling
            if cumulative_cost + est_input_cost >= budget_usd:
                print(f'Stopping before call #{i}: budget {budget_usd} USD would be exceeded by estimated input cost {est_input_cost:.6f} USD (cumulative {cumulative_cost:.6f} USD)')
                break

            # perform the call (supports dry-run)
            resp_json, err, meta = call_openrouter(messages, api_keys, referer=os.getenv('CODESPACE_NAME'), x_title='annotate_with_openrouter', dry_run=args.dry_run)
            # if the call layer provided an input tokens estimate, prefer it
            meta_est_input = meta.get('est_input_tokens') if isinstance(meta, dict) else None
            if meta_est_input:
                est_input_tokens = meta_est_input
                est_input_cost = est_input_tokens * input_price_per_token

            if err:
                print(f'[{i}] call error: {err} (will mark low confidence)')
                ann = {'input': raw_text, 'rewrite': None, 'confidence': 0.0, 'note': f'call_error:{err}', 'est_input_tokens': est_input_tokens, 'est_input_cost': est_input_cost}
                flow.write(json.dumps(ann, ensure_ascii=False) + '\n')
                fout.write(json.dumps(ann, ensure_ascii=False) + '\n')
                cumulative_cost += est_input_cost
                cumulative_input_tokens += est_input_tokens
                print(f'  cumulative cost: ${cumulative_cost:.6f} (budget ${budget_usd:.2f})')
                time.sleep(args.pause)
                if cumulative_cost >= budget_usd:
                    print('Budget exhausted after failed call. Stopping.')
                    break
                continue

            try:
                # parse model text from response
                choices = resp_json.get('choices') or []
                if not choices:
                    raise Exception('no_choices')
                content = choices[0].get('message', {}).get('content', '') if isinstance(choices[0].get('message', {}), dict) else choices[0].get('message')

                # estimate output tokens from response content
                est_output_tokens = est_tokens_from_text(content)
                est_output_cost = est_output_tokens * output_price_per_token

                # check budget with both input+output estimates
                if cumulative_cost + est_input_cost + est_output_cost > budget_usd:
                    print(f'Stopping after receiving response #{i}: executing would exceed budget (${budget_usd:.2f}).')
                    # mark low confidence and stop
                    ann = {'input': raw_text, 'rewrite': None, 'confidence': 0.0, 'note': 'would_exceed_budget_after_output', 'est_input_tokens': est_input_tokens, 'est_output_tokens': est_output_tokens, 'est_total_cost': est_input_cost + est_output_cost}
                    flow.write(json.dumps(ann, ensure_ascii=False) + '\n')
                    fout.write(json.dumps(ann, ensure_ascii=False) + '\n')
                    break

                parsed, perr = parse_model_output(content)
                if parsed is None:
                    # fallback: record raw content with lower confidence
                    ann = {'input': raw_text, 'rewrite': content, 'confidence': 0.5, 'note': f'parse_err:{perr}', 'est_input_tokens': est_input_tokens, 'est_output_tokens': est_output_tokens, 'est_call_cost': est_input_cost + est_output_cost}
                    fout.write(json.dumps(ann, ensure_ascii=False) + '\n')
                    if ann['confidence'] < 0.6:
                        flow.write(json.dumps(ann, ensure_ascii=False) + '\n')
                else:
                    # validate fields
                    rewrite = parsed.get('rewrite')
                    confidence = parsed.get('confidence')
                    note = parsed.get('note')
                    if confidence is None:
                        confidence = 1.0 if len(rewrite or '') > 50 else 0.6
                    ann = {'input': raw_text, 'rewrite': rewrite, 'confidence': float(confidence), 'note': note, 'est_input_tokens': est_input_tokens, 'est_output_tokens': est_output_tokens, 'est_call_cost': est_input_cost + est_output_cost}
                    fout.write(json.dumps(ann, ensure_ascii=False) + '\n')
                    if ann['confidence'] < 0.6:
                        flow.write(json.dumps(ann, ensure_ascii=False) + '\n')

                # update cumulative counters
                cumulative_input_tokens += est_input_tokens
                cumulative_output_tokens += est_output_tokens
                cumulative_cost += (est_input_cost + est_output_cost)

                print(f'[{i}] input_toks={est_input_tokens}, output_toks={est_output_tokens}, call_cost=${(est_input_cost+est_output_cost):.6f}, cumulative_cost=${cumulative_cost:.6f}')

                if cumulative_cost >= budget_usd:
                    print(f'Budget ${budget_usd:.2f} reached after call #{i}; stopping further requests.')
                    break

            except Exception as e:
                ann = {'input': raw_text, 'rewrite': None, 'confidence': 0.0, 'note': f'process_error:{e}'}
                fout.write(json.dumps(ann, ensure_ascii=False) + '\n')
                flow.write(json.dumps(ann, ensure_ascii=False) + '\n')
                cumulative_cost += est_input_cost
                cumulative_input_tokens += est_input_tokens
                print(f'[{i}] processing error, cumulative_cost=${cumulative_cost:.6f}')
                if cumulative_cost >= budget_usd:
                    print('Budget exhausted after processing error. Stopping.')
                    break

            time.sleep(args.pause)

    print('Done. Annotated output:', out_path, 'Low confidence:', low_path)
    print(f'Final cumulative cost=${cumulative_cost:.6f}, input_tokens={cumulative_input_tokens}, output_tokens={cumulative_output_tokens}')
