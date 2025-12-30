#!/usr/bin/env python3
"""Clean pairs JSONL for fine-tuning.
- Unescape HTML entities
- Normalize whitespace
- Redact emails and phone numbers
- Remove exact duplicate (input,output) pairs
- Cap occurrences of the same output to at most N_KEEP_PER_OUTPUT
- Optionally remove templated outputs (heuristic)
- Write cleaned file and a JSON report
"""
import json
import sys
from pathlib import Path
import html
import re
from collections import Counter

IN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('training-data/formatted/asaniczka__upwork-job-postings-dataset-2024-50k-records__pairs.jsonl')
OUT = Path(str(IN).replace('.jsonl','__cleaned.jsonl'))
REPORT = Path(str(IN).replace('.jsonl','__clean_report.json'))

RE_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
RE_PHONE = re.compile(r"\+?\d[\d \-()]{6,}\d")
RE_CLICK = re.compile(r"click to apply", flags=re.I)

N_KEEP_PER_OUTPUT = 10
# TEMPLATED_PHRASES retained for reference but not used to delete by default
TEMPLATED_PHRASES = [
    'improved productivity and accuracy for clients',
    'improved reporting speed',
    'automated excel workflows',
]


def norm_text(s: str) -> str:
    if s is None:
        return ''
    s = html.unescape(s)
    s = s.replace('\r',' ').replace('\n',' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def redact_pii(s: str) -> str:
    s = RE_EMAIL.sub('[REDACTED_EMAIL]', s)
    s = RE_PHONE.sub('[REDACTED_PHONE]', s)
    return s


seen_pairs = set()
output_counts = Counter()
kept = 0
removed_exact_dup = 0
removed_template = 0
redacted_count = 0
removed_by_cap = 0
removed_other = 0

# first pass: count outputs for capping
print('Counting outputs...')
with IN.open('r', encoding='utf-8') as fh:
    for ln in fh:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        out = norm_text(obj.get('output',''))
        output_counts[out] += 1

high_freq = {o:c for o,c in output_counts.items() if c > N_KEEP_PER_OUTPUT}
print(f'Found {len(high_freq)} outputs with > {N_KEEP_PER_OUTPUT} occurrences')

# We'll keep up to N_KEEP_PER_OUTPUT per high-freq output
kept_per_output = Counter()

with IN.open('r',encoding='utf-8') as fh, OUT.open('w',encoding='utf-8') as fo:
    for ln in fh:
        try:
            obj = json.loads(ln)
        except Exception:
            removed_other += 1
            continue
        inp = norm_text(obj.get('input',''))
        out = norm_text(obj.get('output',''))

        # redact pii
        new_inp = redact_pii(inp)
        new_out = redact_pii(out)
        if new_inp != inp or new_out != out:
            redacted_count += 1
            inp, out = new_inp, new_out

        # remove click-to-apply leftovers in input
        inp = RE_CLICK.sub('', inp).strip()

        pair_key = (inp, out)
        if pair_key in seen_pairs:
            removed_exact_dup += 1
            continue

        # cap frequent outputs (keep up to N_KEEP_PER_OUTPUT examples per identical output)
        if output_counts[out] > N_KEEP_PER_OUTPUT:
            if kept_per_output[out] >= N_KEEP_PER_OUTPUT:
                removed_by_cap += 1
                continue
            kept_per_output[out] += 1

        # accept
        seen_pairs.add(pair_key)
        kept += 1
        fo.write(json.dumps({'input': inp, 'output': out}, ensure_ascii=False) + '\n')


report = {
    'input_file': str(IN),
    'output_file': str(OUT),
    'total_input_records': sum(output_counts.values()),
    'unique_outputs': len(output_counts),
    'kept': kept,
    'removed_exact_dup': removed_exact_dup,
    'removed_template': removed_template,
    'removed_by_cap': removed_by_cap,
    'redacted_count': redacted_count,
    'removed_other': removed_other,
    'high_freq_outputs': len(high_freq),
}

with REPORT.open('w',encoding='utf-8') as fh:
    json.dump(report, fh, indent=2)

print('Done. Report written to', REPORT)
print(report)
