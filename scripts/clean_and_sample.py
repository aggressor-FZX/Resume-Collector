#!/usr/bin/env python3
"""Run cleaning pass with diagnostics, sample kept/removed records, and optionally annotate samples with OpenRouter (dry-run default).

Usage:
  python3 scripts/clean_and_sample.py --input <pairs.jsonl> --keep 10 --sample 20 --annotate --dry-run
"""
import argparse
from pathlib import Path
import json
import html
import re
from collections import Counter
import random
from typing import List, Tuple

import sys
# ensure project root is on sys.path so imports like 'scripts.annotate_with_openrouter' work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# reuse annotator if available
try:
    from scripts.annotate_with_openrouter import call_openrouter, prepare_messages, parse_model_output
    HAS_ANNOTATOR = True
except Exception as e:
    print('annotator import failed:', repr(e))
    HAS_ANNOTATOR = False

RE_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
RE_PHONE = re.compile(r"\+?\d[\d \-()]{6,}\d")
RE_CLICK = re.compile(r"click to apply", flags=re.I)


def norm_text(s: str) -> str:
    if s is None:
        return ''
    s = html.unescape(s)
    s = s.replace('\r',' ').replace('\n',' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def redact_pii(s: str) -> Tuple[str,bool]:
    changed = False
    new = RE_EMAIL.sub('[REDACTED_EMAIL]', s)
    if new != s:
        changed = True
    new2 = RE_PHONE.sub('[REDACTED_PHONE]', new)
    if new2 != new:
        changed = True
    return new2, changed


def run_clean(input_path: Path, n_keep_per_output: int = 10) -> Tuple[Path, dict, List[dict]]:
    out_path = Path(str(input_path).replace('.jsonl', '__cleaned.jsonl'))
    report_path = Path(str(input_path).replace('.jsonl', '__clean_report.json'))
    removed_path = Path(str(input_path).replace('.jsonl', '__removed_diagnostics.jsonl'))

    output_counts = Counter()
    with input_path.open('r', encoding='utf-8') as fh:
        for ln in fh:
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            out = norm_text(obj.get('output',''))
            output_counts[out] += 1

    kept_per_output = Counter()
    seen_pairs = set()

    kept = 0
    removed_by_cap = 0
    redacted_total = 0
    removed_exact_dup = 0
    removed_other = 0

    removed_samples = []

    with input_path.open('r', encoding='utf-8') as fh, out_path.open('w', encoding='utf-8') as fo, removed_path.open('w', encoding='utf-8') as fr:
        for ln in fh:
            try:
                obj = json.loads(ln)
            except Exception:
                removed_other += 1
                continue
            inp = norm_text(obj.get('input',''))
            out = norm_text(obj.get('output',''))

            inp_r, changed_inp = redact_pii(inp)
            out_r, changed_out = redact_pii(out)
            if changed_inp or changed_out:
                redacted_total += 1
                inp, out = inp_r, out_r

            inp = RE_CLICK.sub('', inp).strip()
            out = out.strip()

            pair_key = (inp, out)
            if pair_key in seen_pairs:
                removed_exact_dup += 1
                removed_samples.append({'reason':'exact_dup', 'input':inp, 'output':out})
                continue

            if output_counts[out] > n_keep_per_output:
                if kept_per_output[out] >= n_keep_per_output:
                    removed_by_cap += 1
                    removed_samples.append({'reason':'cap_exceeded', 'input':inp, 'output':out})
                    continue
                kept_per_output[out] += 1

            seen_pairs.add(pair_key)
            kept += 1
            fo.write(json.dumps({'input': inp, 'output': out}, ensure_ascii=False) + '\n')

    report = {
        'input_file': str(input_path),
        'output_file': str(out_path),
        'total_input_records': sum(output_counts.values()),
        'unique_outputs': len(output_counts),
        'kept': kept,
        'removed_exact_dup': removed_exact_dup,
        'removed_by_cap': removed_by_cap,
        'redacted_count': redacted_total,
        'removed_other': removed_other,
        'high_freq_outputs': sum(1 for v in output_counts.values() if v > n_keep_per_output),
    }
    with report_path.open('w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)

    return out_path, report, removed_samples


def sample_records(cleaned_path: Path, removed_samples: List[dict], sample_n: int = 20):
    # sample kept
    kept_sample = []
    total = 0
    for ln in cleaned_path.open('r', encoding='utf-8'):
        total += 1
        if len(kept_sample) < sample_n:
            kept_sample.append(json.loads(ln))
        else:
            # reservoir sampling
            r = random.randint(0, total-1)
            if r < sample_n:
                kept_sample[r] = json.loads(ln)
    removed_sample = random.sample(removed_samples, min(len(removed_samples), sample_n)) if removed_samples else []
    return kept_sample, removed_sample


def annotate_samples(samples: List[dict], api_keys: list, dry_run: bool = True, out_prefix: str = 'annotated'):
    annotations = []
    if not HAS_ANNOTATOR:
        for s in samples:
            annotations.append({'input': s.get('input'), 'simulated': '[annotator_missing]'})
        return annotations

    annotated = []
    for s in samples:
        messages = prepare_messages(s.get('input',''))
        resp, err, meta = call_openrouter(messages, api_keys, dry_run=dry_run)
        ann = {'input': s.get('input'), 'err': err, 'meta': meta}
        if resp and isinstance(resp, dict):
            # include usage if present
            ann['usage'] = resp.get('usage')
            # extract content and try to parse JSON out of it
            content = None
            try:
                choices = resp.get('choices') or []
                if choices:
                    content = choices[0].get('message', {}).get('content', '') if isinstance(choices[0].get('message', {}), dict) else choices[0].get('message')
                ann['raw_content'] = content
                parsed, perr = parse_model_output(content)
                if parsed:
                    ann.update(parsed)
                else:
                    ann['parse_error'] = perr
            except Exception as e:
                ann['parse_exception'] = str(e)
        else:
            ann['note'] = 'no_response'
        annotated.append(ann)
    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='training-data/formatted/muhammadadiltalay__fiverr-data-gigs__pairs.jsonl')
    parser.add_argument('--keep', type=int, default=10)
    parser.add_argument('--sample', type=int, default=20)
    parser.add_argument('--annotate', action='store_true', help='Run LLM annotation on samples (dry-run by default)')
    parser.add_argument('--dry-run', action='store_true', help='If --annotate, run annotator in dry-run mode')
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print('Input not found:', inp)
        raise SystemExit(1)

    cleaned_path, report, removed_samples = run_clean(inp, n_keep_per_output=args.keep)
    print('Clean report:', json.dumps(report, indent=2))

    kept_sample, removed_sample = sample_records(cleaned_path, removed_samples, sample_n=args.sample)
    kept_out = Path(str(cleaned_path).replace('.jsonl','__kept_sample.jsonl'))
    removed_out = Path(str(cleaned_path).replace('.jsonl','__removed_sample.jsonl'))

    with kept_out.open('w', encoding='utf-8') as fh:
        for s in kept_sample:
            fh.write(json.dumps(s, ensure_ascii=False) + '\n')
    with removed_out.open('w', encoding='utf-8') as fh:
        for s in removed_sample:
            fh.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f'Wrote kept sample -> {kept_out}  removed sample -> {removed_out} (removed_samples_count={len(removed_samples)})')

    if args.annotate:
        api_keys = [None]
        # try to pick environment keys if present
        import os
        k = os.getenv('OPENROUTER_API_KEY')
        k2 = os.getenv('OPENROUTER_API_KEY_2')
        api_keys = [k, k2]
        # If --dry-run is not explicitly set, default to False (live LLM)
        dry_run = args.dry_run
        print('Annotating kept samples (dry_run=%s) ...' % dry_run)
        ann_kept = annotate_samples(kept_sample, api_keys, dry_run=dry_run)
        kept_annotated_path = Path(str(cleaned_path).replace('.jsonl','__kept_sample_annotated.jsonl'))
        kept_annotated_llm_path = Path(str(cleaned_path).replace('.jsonl','__kept_sample_annotated_llm.jsonl'))
        kept_annotated_path.write_text('\n'.join(json.dumps(a, ensure_ascii=False) for a in ann_kept))
        # write explicit LLm parsed file as well
        kept_annotated_llm_path.write_text('\n'.join(json.dumps(a, ensure_ascii=False) for a in ann_kept))
        print('Annotated kept samples written ->', kept_annotated_llm_path)

        print('Annotating removed samples (dry_run=%s) ...' % dry_run)
        ann_removed = annotate_samples(removed_sample, api_keys, dry_run=dry_run)
        removed_annotated_path = Path(str(cleaned_path).replace('.jsonl','__removed_sample_annotated.jsonl'))
        removed_annotated_llm_path = Path(str(cleaned_path).replace('.jsonl','__removed_sample_annotated_llm.jsonl'))
        removed_annotated_path.write_text('\n'.join(json.dumps(a, ensure_ascii=False) for a in ann_removed))
        removed_annotated_llm_path.write_text('\n'.join(json.dumps(a, ensure_ascii=False) for a in ann_removed))
        print('Annotated removed samples written ->', removed_annotated_llm_path)

    print('Done.')

if __name__ == '__main__':
    main()
