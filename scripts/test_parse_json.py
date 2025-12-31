#!/usr/bin/env python3
"""Quick tests for parse_json_from_response."""
import sys
from generate_imaginator_data_free_tier import parse_json_from_response

samples = [
    '{"job_ad_text":"foo","extracted_skills_json":{},"domain_insights_json":{}}',
    '```json\n{"job_ad_text":"foo","extracted_skills_json":{},"domain_insights_json":{}}\n```',
    '[{"type":"function","name":"JSON Create","parameters":{"json":"{\\"job_ad_text\\":\\"foo\\",\\"extracted_skills_json\\":{},\\"domain_insights_json\\":{}}"}}]',
    'Some text before {"job_ad_text":"foo","extracted_skills_json":{},"domain_insights_json":{}} some after',
    'No JSON here, just text',
]

for s in samples:
    obj, err = parse_json_from_response(s)
    print('Sample:', s[:80].replace('\n',' '))
    print('Result:', 'OK' if obj else 'ERR', err if err else '')
    print('-'*40)

print('Done')
