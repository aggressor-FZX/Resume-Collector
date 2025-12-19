#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

function usage() {
  console.error('Usage: node scripts/format-for-llm.js --input <in.json> --output <out.jsonl> [--format <openai|anthropic|llama>]');
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
const resumes = raw.resumes || raw.items || [];
const outDir = path.dirname(outputFile);
if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

const out = fs.createWriteStream(outputFile, { flags: 'w' });
let count = 0;

function safeString(s) {
  if (s === undefined || s === null) return '';
  return String(s).trim();
}

for (const r of resumes) {
  const original = safeString(r.original_bullet || r.original || r.text || r.raw);
  const improved = safeString(r.improved_bullet || r.improved || r.rewrite);
  const context = safeString(r.context || `${r.role || ''} ${r.company || ''}`.trim()) || 'General tech role';

  if (!original) continue; // skip bad records

  const messages = [];

  if (format === 'openai') {
    messages.push({
      role: 'system',
      content: `You are a world-class tech resume writer. Transform weak, passive bullet points into powerful, metric-driven achievements. Context: ${context}`
    });
    messages.push({ role: 'user', content: `Improve this resume bullet:\n"${original}"` });
    messages.push({ role: 'assistant', content: improved || '' });
  } else if (format === 'anthropic') {
    // simple anthropic-style pair
    messages.push({ role: 'system', content: `You are a world-class tech resume writer. Context: ${context}` });
    messages.push({ role: 'user', content: `Rewrite: ${original}` });
    messages.push({ role: 'assistant', content: improved || '' });
  } else {
    // generic format
    messages.push({ role: 'system', content: `You are an expert tech resume writer. Context: ${context}` });
    messages.push({ role: 'user', content: original });
    messages.push({ role: 'assistant', content: improved || '' });
  }

  const example = { messages };
  out.write(JSON.stringify(example) + '\n');
  count += 1;
}

out.end(() => {
  console.log(`✅ Formatted ${count} examples to ${outputFile} in ${format} format`);
});
