from pathlib import Path

src = Path(__file__).resolve().parent.parent / 'data' / 'inventario-de-medicamentos-marzo-2024.csv'

print('Scanning', src)
lines = src.read_text(encoding='latin-1').splitlines()

def count_fields(line):
    in_q = False
    cnt_commas = 0
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"':
            # handle escaped quotes ""
            if i+1 < len(line) and line[i+1] == '"':
                i += 2
                continue
            in_q = not in_q
        elif ch == ',' and not in_q:
            cnt_commas += 1
        i += 1
    return cnt_commas + 1

counts = [count_fields(l) for l in lines]
from collections import Counter
c = Counter(counts)
print('Field counts distribution (top 10):')
for k,v in c.most_common(10):
    print(k, v)

# expected = most common field count
expected = c.most_common(1)[0][0]
print('Expected fields per row:', expected)

# find malformed lines
malformed = [(i+1, counts[i], lines[i]) for i in range(len(lines)) if counts[i] != expected]
print('Malformed lines found:', len(malformed))

# print up to first 20 malformed lines
for idx, cnt, line in malformed[:20]:
    print('\nLine', idx, 'fields=', cnt)
    print(line[:400])

# write a small report file
out = Path(__file__).resolve().parent.parent / 'data' / 'cleaned' / 'marzo_malformed_report.txt'
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', encoding='utf-8') as f:
    f.write('Expected fields per row: %s\n' % expected)
    f.write('Total lines: %d\n' % len(lines))
    f.write('Malformed lines: %d\n' % len(malformed))
    f.write('\nFirst 200 malformed lines (line_no | fields | content truncated):\n')
    for idx, cnt, line in malformed[:200]:
        line_trunc = line[:400].replace('\n', ' ')
        f.write("%d | %d | %s\n" % (idx, cnt, line_trunc))

print('\nWrote report to', out)
