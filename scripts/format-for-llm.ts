#!/usr/bin/env ts-node

import fs from 'fs';
import path from 'path';
import { dedupeBullets, anonymizeText } from './utils/text-utils';

function usage() {
  console.error('Usage: ts-node scripts/format-for-llm.ts --input <in.json> --output <out.jsonl> [--format <openai|anthropic|llama>]');
  process.exit(1);
}

const args = process.argv.slice(2);
const inputIdx = args.indexOf('--input');
const outputIdx = args.indexOf('--output');
const formatIdx = args.indexOf('--format');

if (inputIdx === -1 || outputIdx === -1) usage();

const inputFile = args[inputIdx + 1];
const outputFile = args[outputIdx + 1];
const format = (formatIdx !== -1 && args[formatIdx + 1]) ? args[formatIdx + 1] : 'openai';

if (!inputFile || !outputFile) usage();

if (!fs.existsSync(inputFile)) {
  console.error(`Input file not found: ${inputFile}`);
  process.exit(1);
}

const raw = JSON.parse(fs.readFileSync(inputFile, 'utf8'));
const resumes: any[] = raw.resumes || raw.items || [];
const outDir = path.dirname(outputFile);
if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

const out = fs.createWriteStream(outputFile, { flags: 'w' });
let count = 0;

for (const r of resumes) {
  const original = (r.original_bullet || r.original || r.text || r.raw || '').toString().trim();
  if (!original) continue;

  // anonymize and dedupe
  const anonymized = anonymizeText(original);
  const improved = (r.improved_bullet || r.improved || r.rewrite || '').toString().trim();
  const anonymized_improved = anonymizeText(improved);
  const context = (r.context || `${r.role || ''} ${r.company || ''}`.trim()) || 'General tech role';

  // dedupe: we may dedupe across a resume's bullets
  const bullets = dedupeBullets([anonymized]);
  for (const b of bullets) {
    const messages: any[] = [];
    if (format === 'openai') {
      messages.push({ role: 'system', content: `You are a world-class tech resume writer. Transform weak, passive bullet points into powerful, metric-driven achievements. Context: ${context}` });
      messages.push({ role: 'user', content: `Improve this resume bullet:\n"${b}"` });
      messages.push({ role: 'assistant', content: anonymized_improved || '' });
    } else if (format === 'anthropic') {
      messages.push({ role: 'system', content: `You are a world-class tech resume writer. Context: ${context}` });
      messages.push({ role: 'user', content: `Rewrite: ${b}` });
      messages.push({ role: 'assistant', content: anonymized_improved || '' });
    } else {
      messages.push({ role: 'system', content: `You are an expert tech resume writer. Context: ${context}` });
      messages.push({ role: 'user', content: b });
      messages.push({ role: 'assistant', content: anonymized_improved || '' });
    }

    const example = { messages };
    out.write(JSON.stringify(example) + '\n');
    count += 1;
  }
}

out.end(() => {
  console.log(`✅ Formatted ${count} examples to ${outputFile} in ${format} format`);
});
