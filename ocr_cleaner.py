
# ocr_cleaner.py

def clean(text: str) -> str:
    import re
    from collections import Counter
    text = text.replace(chr(13), "")  # remove CRs safely
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    cleaned, last, run = [], None, 0
    for ln in lines:
        if ln == last:
            run += 1
            if run <= 3:
                cleaned.append(ln)
        else:
            last, run = ln, 1
            cleaned.append(ln)
    freq = Counter(cleaned)
    final, short_count = [], {}
    for ln in cleaned:
        if len(ln.split()) == 1 and len(ln) <= 5 and freq[ln] > 6:
            c = short_count.get(ln, 0) + 1
            short_count[ln] = c
            if c <= 3:
                final.append(ln)
            continue
        final.append(ln)
    return chr(10).join(final)
