#!/usr/bin/env python3
"""Heuristically generate assistant rewrites for message-formatted training data.

- Reads JSONL files in training-data/formatted
- For each record with messages, extracts original user bullet, removes marketplace fluff
- Generates an assistant rewrite using templates and detected skill tokens
- Writes two outputs per source:
  - <source>__rewritten.jsonl (messages with assistant filled)
  - <source>__pairs.jsonl (input/output pair format: {"input":..., "output":...})

Note: rewrites are synthetic heuristic outputs intended to teach style; they may contain approximate quantifiers phrased cautiously (e.g., "up to 50%") when numeric metrics are not available.
"""
from pathlib import Path
import json
import re

IN_DIR = Path('training-data/formatted')
OUT_DIR = IN_DIR

FLUFF_PATTERNS = [
    r"level\s*\d+\s*seller", r"top\s*rated\s*seller", r"clients\s*\d+", r"\d+k\+",
    r"clients?\s*\d+", r"clients?\s*\d+", r"\b\d+\b"  # trailing numbers often not useful
]
FLUFF_RE = re.compile('|'.join(FLUFF_PATTERNS), flags=re.I)

# simple skill keywords to detect
SKILL_KEYWORDS = [
    'excel','vba','macros','google sheets','google sheets','google','python','web scraping','scraping','data mining',
    'data extraction','automation','selenium','beautifulsoup','scrapy','pivot','dashboard','api','csv','xml','json'
]

# templates
EXCEL_TEMPLATE = "Automated Excel workflows using {tools} to create dashboards and custom macros, improving reporting speed and reducing manual effort for clients."
WEB_TEMPLATE = "Built robust web-scraping and data-extraction pipelines using {tools}, enabling reliable collection of structured datasets for analysis."
GENERIC_TEMPLATE = "Delivered {tools}-based improvements by transforming ad-hoc tasks into repeatable, automated processes that improved productivity and accuracy for clients."


def detect_tools(text: str):
    t = text.lower()
    found = []
    for kw in SKILL_KEYWORDS:
        if kw in t:
            found.append(kw)
    # dedupe & format
    found = list(dict.fromkeys(found))
    if not found:
        # fallback to nouns from text - naive
        tokens = re.findall(r"[a-zA-Z]{3,}", t)
        found = tokens[:3]
    return ', '.join(found)


def clean_user_bullet(bullet: str) -> str:
    b = bullet.strip().strip('"')
    # remove marketplace fluff
    b = FLUFF_RE.sub('', b)
    b = re.sub(r'\s+', ' ', b).strip()
    return b


def generate_assistant(bullet: str):
    cleaned = clean_user_bullet(bullet)
    tools = detect_tools(cleaned)
    t = cleaned.lower()
    # choose template by keyword
    if any(k in t for k in ('excel','vba','macros','pivot','dashboard')):
        rewrite = EXCEL_TEMPLATE.format(tools=tools or 'Excel and VBA')
    elif any(k in t for k in ('web scraping','scraping','selenium','beautifulsoup','scrapy','data extraction','data mining')):
        rewrite = WEB_TEMPLATE.format(tools=tools or 'web scraping tools')
    else:
        rewrite = GENERIC_TEMPLATE.format(tools=tools or 'relevant tools')
    # add a cautious quantified sentence if plausible
    if 'automate' in t or 'automated' in t or 'automation' in t:
        rewrite += ' Typically achieved measurable time savings (e.g., up to 30-60%) through automation.'
    return rewrite


def process_file(path: Path):
    out_rewritten = OUT_DIR / (path.stem + '__rewritten.jsonl')
    out_pairs = OUT_DIR / (path.stem + '__pairs.jsonl')
    written = 0
    with path.open('r', encoding='utf-8') as fh, out_rewritten.open('w', encoding='utf-8') as fout_r, out_pairs.open('w', encoding='utf-8') as fout_p:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # try to recover if partial
                continue
            msgs = obj.get('messages') or []
            # find user content quoted text
            user_msg = ''
            for m in msgs:
                if m.get('role') == 'user':
                    user_msg = m.get('content','')
                    break
            # extract quoted line if present
            quoted = ''
            q = re.search(r'"([^"]{10,300})"', user_msg)
            if q:
                quoted = q.group(1)
            else:
                # fallback - strip prefix 'Improve this resume bullet:'
                quoted = re.sub(r'(?i)Improve this resume bullet:\s*', '', user_msg).strip()
            assistant = generate_assistant(quoted)
            # update messages
            new_msgs = []
            for m in msgs:
                if m.get('role') == 'system':
                    # normalize system message to not include dataset context
                    new_msgs.append({'role':'system','content':'You are a world-class tech resume writer.'})
                elif m.get('role') == 'user':
                    new_msgs.append(m)
                elif m.get('role') == 'assistant':
                    new_msgs.append({'role':'assistant','content':assistant})
            # write rewritten
            fout_r.write(json.dumps({'messages': new_msgs}, ensure_ascii=False) + '\n')
            # write pair
            inp = re.sub(r'(?i)Improve this resume bullet:\s*', '', user_msg).strip()
            pair = {'input': inp, 'output': assistant}
            fout_p.write(json.dumps(pair, ensure_ascii=False) + '\n')
            written += 1
    return written


if __name__ == '__main__':
    files = sorted(IN_DIR.glob('*.jsonl'))
    total = 0
    for f in files:
        print('Processing', f.name)
        n = process_file(f)
        print('  wrote', n, 'rewritten examples for', f.name)
        total += n
    print('Done. Total rewritten:', total)
