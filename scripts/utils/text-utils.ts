import _ from 'lodash';

// Basic anonymization: removes emails, phone numbers, and simple person names patterns
export function anonymizeText(s: string): string {
  let out = s;
  // remove emails
  out = out.replace(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g, '[EMAIL]');
  // remove phones like 123-456-7890 or (123) 456-7890 or 1234567890
  out = out.replace(/(\+\d{1,2}\s?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}/g, '[PHONE]');
  // simple name redaction heuristics: "John Doe" or "J. Doe" -> [NAME]
  out = out.replace(/\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b/g, '[NAME]');
  out = out.replace(/\b([A-Z]\.\s*[A-Z][a-z]+)\b/g, '[NAME]');
  return out;
}

export function dedupeBullets(bullets: string[]): string[] {
  // normalize whitespace and lower-case
  const norm = bullets.map((b: string) => b.replace(/\s+/g, ' ').trim());
  // dedupe with lodash uniqBy using simple lower-case key
  return _.uniqBy(norm, (b: string) => b.toLowerCase());
}
