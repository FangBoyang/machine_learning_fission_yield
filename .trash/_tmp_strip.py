import re, sys, io, os

def strip(path, limit=6000):
    raw = open(path, 'rb').read()
    for enc in ('utf-8', 'gbk', 'latin-1'):
        try:
            s = raw.decode(enc)
            break
        except Exception:
            continue
    s = re.sub(r'(?is)<(script|style|svg|noscript)[^>]*>.*?</\1>', ' ', s)
    # arxiv abstract blocks
    s = re.sub(r'(?is)<blockquote[^>]*class="abstract[^"]*"[^>]*>', '\n=== ABSTRACT ===\n', s)
    s = re.sub(r'(?is)<h1[^>]*class="title[^"]*"[^>]*>', '\n=== TITLE ===\n', s)
    s = re.sub(r'(?is)<[^>]+>', ' ', s)
    s = re.sub(r'&nbsp;', ' ', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'&#\d+;', '', s)
    s = re.sub(r'&[a-z]+;', ' ', s)
    s = re.sub(r'[ \t\xa0]+', ' ', s)
    s = re.sub(r'\n\s*\n+', '\n', s)
    return s.strip()[:limit]

for p in sys.argv[1:]:
    print('\n' + '=' * 30 + ' ' + os.path.basename(p) + ' ' + '=' * 30)
    sys.stdout.write(strip(p) + '\n')
