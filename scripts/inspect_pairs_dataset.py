#!/usr/bin/env python3
"""Quick inspector for pairs-formatted JSONL files (input/output pairs).

Outputs baseline metrics: total records, JSON errors, missing fields, avg lengths,
common output duplicates, fraction outputs identical to inputs, and templates.
"""
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import re
from difflib import SequenceMatcher

F = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('training-data/formatted/asaniczka__upwork-job-postings-dataset-2024-50k-records__pairs.jsonl')


RE_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
RE_PHONE = re.compile(r"\+?\d[\d \-()]{6,}\d")


def norm_text(s: str) -> str:
    s = s or ''
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def token_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def main():
    total = 0
    json_errors = 0
    missing_in = 0
    missing_out = 0
    outputs_short = 0
    outputs_identical = 0
    outputs_with_email = 0
    outputs_with_phone = 0
    langs_non_english = 0

    out_counter = Counter()
    short_examples = []
    templated_counter = Counter()
    sim_scores = []

    for ln in F.open('r', encoding='utf-8'):
        total += 1
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            json_errors += 1
            continue
        inp = norm_text(obj.get('input',''))
        out = norm_text(obj.get('output',''))
        if not inp:
            missing_in += 1
        if not out:
            missing_out += 1
        out_counter[out] += 1
        if len(out) < 30:
            outputs_short += 1
            if len(short_examples) < 10:
                short_examples.append((inp, out))
        if out.lower() == inp.lower() or token_similarity(inp.lower(), out.lower()) > 0.9:
            outputs_identical += 1
        if RE_EMAIL.search(inp) or RE_EMAIL.search(out):
            outputs_with_email += 1
        if RE_PHONE.search(inp) or RE_PHONE.search(out):
            outputs_with_phone += 1

        # detect templated or repeating patterns by n-gram presence
        # capture simple common starts
        start = out[:60]
        templated_counter[start] += 1

        sim_scores.append(token_similarity(inp.lower(), out.lower()))

    unique_outputs = sum(1 for k in out_counter if out_counter[k] == 1)
    most_common_outs = out_counter.most_common(20)

    print(f'File: {F}\nTotal lines: {total}\nJSON errors: {json_errors}\nMissing input: {missing_in}  Missing output: {missing_out}\n')
    print(f'Outputs short (<30 chars): {outputs_short}\nOutputs identical/near-identical to input: {outputs_identical}\nOutputs with emails: {outputs_with_email}  with phone-like: {outputs_with_phone}\n')
    print('Top 20 most common outputs (count, sample):')
    for k,v in most_common_outs:
        print(f'{v}: {k[:200]!r}')
    avg_sim = sum(sim_scores)/len(sim_scores) if sim_scores else 0.0
    print(f'Avg input/output similarity (0-1): {avg_sim:.4f}\nUnique outputs: {unique_outputs} (of {len(out_counter)})')
    print('\nExamples of short outputs:')
    for i,(inp,out) in enumerate(short_examples):
        print(f'[{i+1}] IN: {inp[:200]!r}\n    OUT: {out!r}\n')

if __name__ == '__main__':
    main()
