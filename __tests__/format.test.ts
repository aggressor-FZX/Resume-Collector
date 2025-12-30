import { anonymizeText, dedupeBullets } from '../scripts/utils/text-utils';

describe('anonymizeText', () => {
  it('replaces emails and phones and names', () => {
    const input = 'Responsible for contacting john.doe@example.com and (123) 456-7890. Worked with John Doe on infra.';
    const out = anonymizeText(input);
    expect(out).toContain('[EMAIL]');
    expect(out).toContain('[PHONE]');
    expect(out).toContain('[NAME]');
  });

  it('does not mangle normal words', () => {
    const input = 'Implemented caching and monitoring on servers';
    const out = anonymizeText(input);
    expect(out).toBe(input);
  });
});

describe('dedupeBullets', () => {
  it('removes duplicates case-insensitively', () => {
    const bullets = ['Fix bugs', 'fix bugs', 'Improved latency'];
    const out = dedupeBullets(bullets);
    expect(out).toHaveLength(2);
    expect(out.map(b => b.toLowerCase())).toContain('fix bugs');
  });
});
