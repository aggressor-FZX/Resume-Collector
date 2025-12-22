#!/usr/bin/env python3
"""Run deduplication and QA metrics on combined dataset.

- Loads data/anonymized_combined_resume_dataset.json
- Performs exact and fuzzy dedup (blocking by length and prefix)
- Computes QA metrics: lengths, completeness, PII counts, source distribution
- Writes filtered dataset to data/filtered_resume_dataset.json and parquet
- Writes a JSON report to data/qc_report.json and a short text summary to data/qc_summary.txt
"""
from pathlib import Path
import json
import re
import statistics
import hashlib
from collections import Counter, defaultdict
import math
import pandas as pd

DATA_IN = Path('data/anonymized_combined_resume_dataset.json')
OUT_JSON = Path('data/filtered_resume_dataset.json')
OUT_PARQUET = Path('data/filtered_resume_dataset_filtered.parquet')
QC_JSON = Path('data/qc_report.json')
QC_SUMMARY = Path('data/qc_summary.txt')
SAMPLES_DIR = Path('data/qc_samples')
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
URL_RE = re.compile(r'https?://\S+')
PHONE_RE = re.compile(r'(?:\+?\d{1,3})?[\s\-\(]*\d{2,4}[\)\s\-]*\d{2,4}[\s\-]*\d{2,4}')
LONGNUM_RE = re.compile(r'\b\d{6,}\b')

MIN_LENGTH = 200  # PRD threshold
MIN_SKILLS = 5


def normalize_text(t: str) -> str:
    if not t:
        return ''
    t = t.strip().lower()
    # collapse whitespace
    t = re.sub(r'\s+', ' ', t)
    return t


def token_set(s: str):
    # simple token set for jaccard
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    toks = [w for w in s.split() if len(w) > 2]
    return set(toks)


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    uni = a | b
    return len(inter) / len(uni)


import ijson

def load_records_stream(path: Path):
    """Stream records from a JSON array file using ijson to avoid loading whole file."""
    with path.open('rb') as f:
        for obj in ijson.items(f, 'item'):
            yield obj


def save_json(records, path: Path):
    with path.open('w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def run_dedup(records):
    total_before = len(records)
    # exact dedupe on normalized text
    seen_exact = set()
    unique = []
    for r in records:
        text = r.get('content') or r.get('resume_text') or ''
        norm = normalize_text(text)
        h = hashlib.sha1(norm.encode('utf-8')).hexdigest()
        if h in seen_exact:
            continue
        seen_exact.add(h)
        r['_norm'] = norm
        r['_hash'] = h
        unique.append(r)

    # fuzzy dedup within blocks
    # block by length bin and prefix
    buckets = defaultdict(list)
    for i, r in enumerate(unique):
        ln = len(r['_norm'])
        bin_id = ln // 100
        prefix = (r['_norm'][:50]) if r['_norm'] else ''
        key = f'{bin_id}__{prefix[:20]}'
        buckets[key].append(i)

    to_drop = set()
    for key, idxs in buckets.items():
        if len(idxs) <= 1:
            continue
        # compare within bucket
        token_sets = [token_set(unique[i]['_norm']) for i in idxs]
        for a_pos in range(len(idxs)):
            i = idxs[a_pos]
            if i in to_drop:
                continue
            for b_pos in range(a_pos + 1, len(idxs)):
                j = idxs[b_pos]
                if j in to_drop:
                    continue
                score = jaccard(token_sets[a_pos], token_sets[b_pos])
                if score >= 0.9:
                    # mark the latter as duplicate
                    to_drop.add(j)

    filtered = [r for idx, r in enumerate(unique) if idx not in to_drop]
    # cleanup helper keys
    for r in filtered:
        r.pop('_norm', None)
        r.pop('_hash', None)
    return filtered, total_before, len(filtered), len(to_drop)


def compute_qc(records):
    n = len(records)
    lengths = [len((r.get('content') or '').strip()) for r in records]
    stats = {}
    stats['total'] = n
    stats['min_len'] = min(lengths) if lengths else 0
    stats['max_len'] = max(lengths) if lengths else 0
    stats['mean_len'] = statistics.mean(lengths) if lengths else 0
    stats['median_len'] = statistics.median(lengths) if lengths else 0
    stats['p25_len'] = statistics.quantiles(lengths, n=4)[0] if lengths else 0
    stats['p75_len'] = statistics.quantiles(lengths, n=4)[2] if lengths else 0

    stats['len_ge_200'] = sum(1 for l in lengths if l >= MIN_LENGTH)

    # skills and experience completeness
    stats['have_skills'] = sum(1 for r in records if r.get('skills'))
    stats['have_experience'] = sum(1 for r in records if r.get('experience'))
    stats['have_years'] = sum(1 for r in records if r.get('total_experience_years'))

    # source distribution
    srcs = Counter(r.get('source', 'unknown') for r in records)
    stats['source_distribution'] = dict(srcs.most_common())

    # PII detection
    email_count = 0
    url_count = 0
    phone_count = 0
    longnum_count = 0
    for r in records:
        t = r.get('content') or ''
        if EMAIL_RE.search(t):
            email_count += 1
        if URL_RE.search(t):
            url_count += 1
        if PHONE_RE.search(t):
            phone_count += 1
        if LONGNUM_RE.search(t):
            longnum_count += 1
    stats['pii_emails'] = email_count
    stats['pii_urls'] = url_count
    stats['pii_phones'] = phone_count
    stats['pii_long_numbers'] = longnum_count

    # sample per length quartile
    samples = {}
    if records:
        sorted_recs = sorted(records, key=lambda r: len((r.get('content') or '').strip()))
        def sample_from_range(start_frac, end_frac, k=5):
            start = int(start_frac * len(sorted_recs))
            end = int(end_frac * len(sorted_recs))
            if start >= end:
                return []
            sub = sorted_recs[start:end]
            step = max(1, len(sub) // k)
            return [s for s in sub[::step][:k]]
        samples['short'] = sample_from_range(0.0, 0.25)
        samples['mid'] = sample_from_range(0.25, 0.75)
        samples['long'] = sample_from_range(0.75, 1.0)
    stats['samples_counts'] = {k: len(v) for k, v in samples.items()}
    return stats, samples


if __name__ == '__main__':
    print('Streaming records from', DATA_IN)
    # perform exact dedupe while streaming to keep memory low
    seen_exact = set()
    unique = []
    total = 0
    for rec in load_records_stream(DATA_IN):
        total += 1
        text = rec.get('content') or rec.get('resume_text') or ''
        norm = normalize_text(text)
        h = hashlib.sha1(norm.encode('utf-8')).hexdigest()
        if h in seen_exact:
            continue
        seen_exact.add(h)
        rec['_norm'] = norm
        rec['_hash'] = h
        unique.append(rec)
        if total % 50000 == 0:
            print(f'  streamed {total} records, unique so far {len(unique)}')

    print('Stream complete: total loaded:', total)
    print('Unique after exact dedupe:', len(unique))

    filtered, before, after, dropped = run_dedup(unique)
    print(f'Dedup done: before={before}, after={after}, dropped={dropped}')

    qc_stats, qc_samples = compute_qc(filtered)

    report = {
        'dedup_before': before,
        'dedup_after': after,
        'dedup_removed': dropped,
        'qc': qc_stats
    }

    print('Writing filtered outputs...')
    save_json(filtered, OUT_JSON)
    df = pd.DataFrame(filtered)
    df.to_parquet(OUT_PARQUET, index=False)
    save_json(report, QC_JSON)

    # write human readable summary
    with QC_SUMMARY.open('w', encoding='utf-8') as fh:
        fh.write('QA Summary\n')
        fh.write('==========\n')
        fh.write(f"Total before dedup: {before}\n")
        fh.write(f"Total after dedup: {after}\n")
        fh.write(f"Duplicates removed: {dropped}\n")
        fh.write('\n')
        q = qc_stats
        fh.write(f"Content length - min: {q['min_len']}, median: {q['median_len']}, mean: {q['mean_len']}, max: {q['max_len']}\n")
        fh.write(f"Records >= {MIN_LENGTH} chars: {q['len_ge_200']} ({q['len_ge_200']/after:.2%} of filtered)\n")
        fh.write(f"Have skills: {q['have_skills']}\n")
        fh.write(f"Have experience entries: {q['have_experience']}\n")
        fh.write(f"PII counts - emails: {q['pii_emails']}, urls: {q['pii_urls']}, phones: {q['pii_phones']}, longnums: {q['pii_long_numbers']}\n")
        fh.write('\nSource distribution:\n')
        for s, c in q['source_distribution'].items():
            fh.write(f" - {s}: {c}\n")

    # save sample files for inspection
    for name, lst in qc_samples.items():
        outp = SAMPLES_DIR / f'{name}_samples.jsonl'
        with outp.open('w', encoding='utf-8') as fh:
            for rec in lst:
                fh.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print('QC report written to', QC_JSON, 'summary to', QC_SUMMARY)
    print('Filtered dataset saved to', OUT_JSON)
    print('Done')
