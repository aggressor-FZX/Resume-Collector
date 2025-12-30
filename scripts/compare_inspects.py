#!/usr/bin/env python3
"""Compare two JSONL pairs files and print side-by-side metrics.
Usage: python scripts/compare_inspects.py orig.jsonl cleaned.jsonl
"""
import sys, json, re
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

RE_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
RE_PHONE = re.compile(r"\+?\d[\d \-()]{6,}\d")


def norm(s):
    return (s or '').strip()


def inspect(filep):
    total = 0
    out_counter = Counter()
    emails = 0
    phones = 0
    sim_scores = []
    for ln in Path(filep).open('r',encoding='utf-8'):
        total += 1
        obj = json.loads(ln)
        inp = norm(obj.get('input',''))
        out = norm(obj.get('output',''))
        out_counter[out] += 1
        if RE_EMAIL.search(inp) or RE_EMAIL.search(out):
            emails += 1
        if RE_PHONE.search(inp) or RE_PHONE.search(out):
            phones += 1
        sim_scores.append(SequenceMatcher(None, inp.lower(), out.lower()).ratio())
    top = out_counter.most_common(10)
    return {
        'total': total,
        'unique_outputs': len(out_counter),
        'top1_count': top[0][1] if top else 0,
        'top10': top,
        'emails': emails,
        'phones': phones,
        'avg_sim': sum(sim_scores)/len(sim_scores) if sim_scores else 0,
    }


def pretty(d):
    return json.dumps(d, indent=2)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: compare_inspects.py before.jsonl after.jsonl')
        raise SystemExit(1)
    a = inspect(sys.argv[1])
    b = inspect(sys.argv[2])
    print('Before:\n', pretty(a))
    print('\nAfter:\n', pretty(b))
    print('\nChanges:')
    print(f"Total reduced: {a['total']} -> {b['total']} ({a['total']-b['total']} removed)")
    print(f"Top1 count reduced: {a['top1_count']} -> {b['top1_count']}")
    print(f"Avg similarity: {a['avg_sim']:.4f} -> {b['avg_sim']:.4f}")
    print(f"Unique outputs: {a['unique_outputs']} -> {b['unique_outputs']}")
    print(f"Phone-like tokens: {a['phones']} -> {b['phones']}")
    print(f"Email-like tokens: {a['emails']} -> {b['emails']}")
