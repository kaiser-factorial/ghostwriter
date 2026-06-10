#!/usr/bin/env python3
"""
clean_corpora.py — Parse and normalize the four ghost corpora into entry-level JSONL.

Sources (all US public domain):
  - Samuel Pepys, Diary (1660-1669). PG #4200, Wheatley ed. Footnotes stripped.
  - Vincent van Gogh, Letters of a Post-Impressionist (Ludovici trans., 1912/1913). PG #40393.
  - Katherine Mansfield, Journal (Knopf 1927/1928 printing). Archive.org OCR; editorial
    brackets by J.M. Murry stripped (both for voice purity and conservatism).
  - Mary MacLane, The Story of Mary MacLane (1902) PG #43696, and
    I, Mary MacLane (1917) PG #43556.

Output: data/clean/<persona>.jsonl with one entry per line:
  {"persona": str, "date": str|None, "synthetic_date": bool, "title": str|None, "text": str}

Usage: python scripts/clean_corpora.py [--raw-dir data/raw] [--out-dir data/clean]
"""
import argparse
import json
import random
import re
from pathlib import Path

MONTHS = ("January February March April May June July August September "
          "October November December").split()
MONTH_RE = "|".join(MONTHS)

# ---------------------------------------------------------------- helpers

def gutenberg_body(text: str) -> str:
    """Slice out the Gutenberg header/footer."""
    start = re.search(r"\*\*\* ?START OF (THIS|THE) PROJECT GUTENBERG.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF (THIS|THE) PROJECT GUTENBERG", text)
    s = start.end() if start else 0
    e = end.start() if end else len(text)
    return text[s:e]


def unwrap(paragraph: str) -> str:
    """Join hard-wrapped lines into a single flowing paragraph."""
    return re.sub(r"\s*\n\s*", " ", paragraph).strip()


def paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def clean_ws(s: str) -> str:
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def entry_ok(text: str, min_chars: int = 120) -> bool:
    """Reject fragments too short to carry voice."""
    return len(text) >= min_chars


# ---------------------------------------------------------------- Pepys

def clean_pepys(raw_dir: Path) -> list[dict]:
    text = (raw_dir / "pepys_complete.txt").read_text(encoding="utf-8", errors="replace")
    text = gutenberg_body(text).replace("\r\n", "\n")

    # Strip indented footnote blocks (Wheatley's notes: lines indented >= 4 spaces).
    lines = [ln for ln in text.split("\n") if not re.match(r"^\s{4,}\S", ln)]
    text = "\n".join(lines)
    # Strip any leftover inline editorial brackets that survived (conservative).
    text = re.sub(r"\[[^\[\]]{0,400}?\]", "", text, flags=re.S)
    # Drop "ETEXT EDITOR'S BOOKMARKS" blocks if present.
    text = re.sub(r"ETEXT EDITOR'S BOOKMARKS.*?(?=\n[A-Z]+ \d{4}|\Z)", "", text, flags=re.S)

    month_hdr = re.compile(rf"^({MONTH_RE.upper()})\s+(\d{{4}})(?:-(\d{{2,4}}))?\s*$")
    day_start = re.compile(r"^(\d{1,2})(st|nd|rd|th)[.\s]")

    entries, cur_month, cur_year, cur = [], None, None, None
    for para in paragraphs(text):
        first = para.lstrip().split("\n", 1)[0]
        mh = month_hdr.match(first.strip())
        if mh:
            cur_month = mh.group(1).capitalize()
            # "1659-1660" style: the latter year is the modern reckoning.
            cur_year = mh.group(3) or mh.group(2)
            if len(cur_year) == 2:
                cur_year = mh.group(2)[:2] + cur_year
            continue
        ds = day_start.match(para.lstrip())
        if ds and cur_month:
            if cur:
                entries.append(cur)
            day = f"{ds.group(1)}{ds.group(2)}"
            body = day_start.sub("", para.lstrip(), count=1)
            cur = {
                "persona": "pepys",
                "date": f"{day} {cur_month} {cur_year}",
                "synthetic_date": False,
                "title": None,
                "text": unwrap(body),
            }
        elif cur:
            cur["text"] += "\n\n" + unwrap(para)
    if cur:
        entries.append(cur)

    out = []
    for e in entries:
        e["text"] = clean_ws(e["text"])
        if entry_ok(e["text"]):
            out.append(e)
    return out


# ---------------------------------------------------------------- Van Gogh

def clean_vangogh(raw_dir: Path) -> list[dict]:
    text = (raw_dir / "vangogh_letters.txt").read_text(encoding="utf-8", errors="replace")
    text = gutenberg_body(text).replace("\r\n", "\n")

    # Drop Ludovici's introductory essay: keep from first letters section onward.
    m = re.search(r"^LETTERS TO HIS BROTHER\s*$", text, flags=re.M)
    if m:
        text = text[m.end():]
    # Remove section headings and illustrations.
    text = re.sub(r"^LETTERS TO E\. BERNARD\s*$", "", text, flags=re.M)
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)

    # Letters are separated by asterisk rules.
    letters = re.split(r"\n\s*\*(?:\s+\*)+\s*\n", text)

    entries = []
    for letter in letters:
        letter = letter.strip()
        if not letter:
            continue
        # Strip salutation and closing flourishes for diary-ification, but keep body.
        letter = re.sub(r"^(MY )?DEAR [A-Z]+,?\s*\n", "", letter)
        body = "\n\n".join(unwrap(p) for p in paragraphs(letter))
        body = clean_ws(body)
        if entry_ok(body, min_chars=200):
            entries.append({
                "persona": "vangogh",
                "date": None,  # synthetic dates assigned below
                "synthetic_date": True,
                "title": None,
                "text": body,
            })

    # The Ludovici letters run roughly 1881 (The Hague) -> 1890 (Auvers), in order.
    # Assign evenly spaced synthetic dates across that span so the ghost keeps
    # Vincent's chronology of mood (early earnestness -> Arles fever -> late calm).
    rng = random.Random(1890)
    n = len(entries)
    for i, e in enumerate(entries):
        year = 1881 + round(i * 9 / max(n - 1, 1))
        month = MONTHS[rng.randrange(12)]
        day = rng.randrange(1, 29)
        e["date"] = f"{day} {month} {year}"
    return entries


# ---------------------------------------------------------------- Mansfield

OCR_DIGIT = str.maketrans({"I": "1", "i": "1", "l": "1", "O": "0", "o": "0",
                           "Q": "9", "q": "9", "p": "9", "g": "9", "S": "5"})

def _ocr_year(line: str):
    """Detect OCR-mangled standalone year headers like 'IQI4' or 'igio' -> 1914/1910."""
    s = line.strip()
    if not (3 <= len(s) <= 6):
        return None
    t = re.sub(r"\s", "", s).translate(OCR_DIGIT)
    if re.fullmatch(r"19[0-2][0-9]", t):
        return int(t)
    return None


def clean_mansfield(raw_dir: Path) -> list[dict]:
    text = (raw_dir / "mansfield_journal_1927.txt").read_text(encoding="utf-8", errors="replace")

    # Trim front matter: start at the first year header (1904) region.
    lines = text.split("\n")
    body_lines, year, started = [], None, False
    page_hdr = re.compile(r"^\s*(Journal\s*(of)?\s*$|Katherine\s*Man.*$|Journal\s+\S{1,8}\s*$)")
    for ln in lines:
        y = _ocr_year(ln)
        if y:
            started = True
            body_lines.append(f"\n@@YEAR {y}@@\n")
            continue
        if not started:
            continue
        if page_hdr.match(ln) and len(ln.strip()) < 30:
            continue
        body_lines.append(ln)
    text = "\n".join(body_lines)

    # Remove Murry's editorial bracket blocks (may span lines).
    text = re.sub(r"\[[^\[\]]{0,2000}?\]", "", text, flags=re.S)
    # Strip OCR page-number artifact lines like " - 102 = " or "102".
    text = re.sub(r"^\s*[-=~*•']*\s*\d{1,3}\s*[-=~*•']*\s*$", "", text, flags=re.M)
    # OCR de-hyphenation across former line breaks: "to- day" -> "to-day" style joins.
    text = re.sub(r"(\w)-\s+(?=\w)", r"\1", text)
    # Collapse runs of spaces (OCR double-spacing).
    text = re.sub(r"[ \t]{2,}", " ", text)

    date_start = re.compile(rf"^({MONTH_RE})\s*(\d{{1,2}})?\s*\.")
    entries, cur, cur_year = [], None, 1904
    fragment_buf = []
    rng = random.Random(1923)

    def flush_fragments():
        nonlocal fragment_buf
        buf, chunk = [], []
        size = 0
        for p in fragment_buf:
            chunk.append(p)
            size += len(p)
            if size > 700:
                buf.append("\n\n".join(chunk)); chunk, size = [], 0
        if chunk:
            buf.append("\n\n".join(chunk))
        for b in buf:
            if entry_ok(b, 250):
                entries.append({
                    "persona": "mansfield",
                    "date": f"{rng.randrange(1,29)} {MONTHS[rng.randrange(12)]} {cur_year}",
                    "synthetic_date": True, "title": None, "text": clean_ws(b),
                })
        fragment_buf = []

    for para in paragraphs(text):
        ym = re.match(r"@@YEAR (\d{4})@@", para.strip())
        if ym:
            if cur: entries.append(cur); cur = None
            flush_fragments()
            cur_year = int(ym.group(1))
            continue
        p = unwrap(para)
        dm = date_start.match(p)
        if dm:
            if cur: entries.append(cur)
            flush_fragments()
            month, day = dm.group(1), dm.group(2)
            date = f"{day} {month} {cur_year}" if day else f"{month} {cur_year}"
            cur = {"persona": "mansfield", "date": date, "synthetic_date": False,
                   "title": None, "text": clean_ws(p[dm.end():])}
        elif cur and len(cur["text"]) < 1200:
            cur["text"] += "\n\n" + clean_ws(p)
        else:
            if cur: entries.append(cur); cur = None
            fragment_buf.append(clean_ws(p))
    if cur: entries.append(cur)
    flush_fragments()

    return [e for e in entries if entry_ok(e["text"])]


# ---------------------------------------------------------------- MacLane

def clean_maclane(raw_dir: Path) -> list[dict]:
    entries = []

    # --- The Story of Mary MacLane (1902): right-aligned date headers.
    text = (raw_dir / "maclane_story.txt").read_text(encoding="utf-8", errors="replace")
    text = gutenberg_body(text).replace("\r\n", "\n")
    text = re.sub(r"\[(Photograph|Illustration)[^\]]*\]", "", text)
    text = text.replace("_", "").replace("=", "")

    date_line = re.compile(rf"^\s{{20,}}({MONTH_RE})\s+(\d{{1,2}})(,\s*(\d{{4}}))?\.?\s*$",
                           flags=re.M)
    cur_year = 1901
    marks = list(date_line.finditer(text))
    for i, m in enumerate(marks):
        if m.group(4):
            cur_year = int(m.group(4))
        date = f"{m.group(2)} {m.group(1)} {cur_year}"
        seg = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        body = "\n\n".join(unwrap(p) for p in paragraphs(seg))
        body = clean_ws(body)
        if entry_ok(body):
            entries.append({"persona": "maclane", "date": date,
                            "synthetic_date": False, "title": None, "text": body})

    # --- I, Mary MacLane (1917): entries headed by right-aligned day-words
    # ("To-day", "To-morrow"), preceded by an italic title line.
    text = (raw_dir / "maclane_i.txt").read_text(encoding="utf-8", errors="replace")
    text = gutenberg_body(text).replace("\r\n", "\n")
    text = re.sub(r"\[(Photograph|Illustration)[^\]]*\]", "", text)

    head = re.compile(r"^\s{30,}(To-day|To-morrow|[A-Z][a-z]+day)\s*$", flags=re.M)
    title_re = re.compile(r"_([^_\n]{3,80})_\s*$")
    marks = list(head.finditer(text))
    rng = random.Random(1917)
    for i, m in enumerate(marks):
        # look back a few lines for the italic title
        back = text[max(0, m.start() - 300): m.start()]
        tmatch = title_re.search(back.strip())
        title = tmatch.group(1).strip() if tmatch else None
        seg = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        seg = seg.replace("_", "")
        # cut the next entry's title off the tail
        seg = re.sub(r"\n[^\n]{3,80}\n\s*$", "\n", seg)
        body = "\n\n".join(unwrap(p) for p in paragraphs(seg))
        body = clean_ws(body)
        if entry_ok(body):
            # The book was written Jan-Mar 1917 in Butte; scatter synthetic dates there.
            date = f"{rng.randrange(1,29)} {MONTHS[rng.randrange(0,3)]} 1917"
            entries.append({"persona": "maclane", "date": date,
                            "synthetic_date": True, "title": title, "text": body})

    return entries


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw", type=Path)
    ap.add_argument("--out-dir", default="data/clean", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cleaners = {"pepys": clean_pepys, "vangogh": clean_vangogh,
                "mansfield": clean_mansfield, "maclane": clean_maclane}
    stats = {}
    for name, fn in cleaners.items():
        entries = fn(args.raw_dir)
        out = args.out_dir / f"{name}.jsonl"
        with out.open("w") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        chars = sum(len(e["text"]) for e in entries)
        stats[name] = (len(entries), chars)
        print(f"[clean] {name:10s} entries={len(entries):5d} chars={chars:9,d} "
              f"avg={chars // max(len(entries),1):5d}")
    total = sum(c for _, c in stats.values())
    print(f"[clean] TOTAL chars={total:,} (~{total // 4:,} tokens)")


if __name__ == "__main__":
    main()
