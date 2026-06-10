# HOW TO RAISE YOUR OWN GHOST

A general recipe for turning any dead writer's diary into a model you can
talk to. This repo is one instantiation; the method generalizes.

## 0. What you need

- A diary, journal, or letter corpus that is **public domain** (see step 1 —
  this kills more candidates than anything technical will)
- A GPU for ~10–60 minutes (RunPod/Lambda/Colab; a 24GB card is plenty for 3B)
- Python, this repo's `requirements.txt`

## 1. Choose a ghost (the legal séance rules)

The único hard constraint: you need the right to use the text. Rules of thumb
for the US:

- **Published before 1930** → public domain, full stop. This is the green zone.
- **Author died before ~1955 but published later** → danger. Copyright runs
  from *publication*, not composition. Anne Frank wrote in the 1940s; the
  diary published 1947+ and is locked for years yet. Anaïs Nin wrote in the
  1930s; published 1966+; locked for decades. Plath: published 1982; locked.
- **Translations are separately copyrighted.** Van Gogh's letters are PD, but
  only *translations published before 1930* (like Ludovici 1912) are usable.
  The standard 1958 English edition is not.
- **Editions matter.** Mansfield's *Journal* (1927) is PD; the 1954
  "definitive edition" added restored text and is not. Use the right scan.
- **Editorial apparatus** (footnotes, introductions, bracketed insertions by
  a modern editor) can carry its own copyright even when the diary text is
  PD. Strip it — which you want to do anyway for voice purity.
- Owning a copy (a purchased ebook, a school PDF) conveys **zero** training
  rights. License to read ≠ license to reproduce.

Good hunting grounds: Project Gutenberg (search "diary", "journal",
"letters"), Archive.org pre-1930 scans, Wikisource. The
[gutendex](https://gutendex.com) API makes Gutenberg searchable
programmatically.

Voice candidates we shortlisted but didn't use, free to a good home:
W.N.P. Barbellion (*Journal of a Disappointed Man*, 1919 — dying naturalist,
mordantly funny), Marie Bashkirtseff (Blind trans. 1890 — ego supernova),
Dorothy Wordsworth (Grasmere journals), Anne Lister (raw diaries PD, but
sourcing PD *transcriptions* takes work), Franz Kafka's diaries in German
(died 1924; German text PD — translation rights are the trap).

## 2. Get the text

```bash
curl -sL "https://www.gutenberg.org/ebooks/<ID>.txt.utf-8" -o data/raw/ghost.txt
# if gutenberg.org rate-limits, use the mirror:
# https://aleph.pglaf.org/4/3/6/9/43696/43696-0.txt   (id digit-split path)
```

For Archive.org scans, the OCR text lives at
`https://archive.org/download/<identifier>/<identifier>_djvu.txt`. Expect OCR
noise; budget cleaning time accordingly (see what `clean_mansfield` in
`scripts/clean_corpora.py` does about mangled year headers and page furniture).

## 3. Write a parser for your ghost

This is the real work, and it is bespoke per book. Every diary encodes entries
differently: Pepys uses month headers + day ordinals, MacLane right-aligns her
dates, Mansfield's journal is half dated entries and half fragments. Your
parser's contract is simple — emit one JSON object per entry:

```json
{"persona": "yourghost", "date": "19 January 1917", "synthetic_date": false,
 "title": null, "text": "..."}
```

Tips learned the hard way:

- **Strip editor voice**: footnotes, bracketed insertions, introductions.
  The ghost should contain only the ghost.
- **No usable dates?** Assign synthetic ones — but *preserve corpus order*
  across the writer's real date range, so their emotional chronology survives
  (see `clean_vangogh`). Flag them `synthetic_date: true`.
- **Fragments** (undated scraps): buffer and chunk them to entry-sized pieces
  rather than discarding — Mansfield's fragments are some of her best voice.
- **Set a minimum entry length** (~120 chars). Two-line entries teach mostly
  formatting.
- Eyeball 20 random parsed entries before moving on. OCR garbage and
  editorial residue hide in the middle of books, not the start.

Then register your cleaner in the `cleaners` dict in `clean_corpora.py`, add
the persona name to `PERSONAS` in `build_dataset.py`, and you're plumbed in.

## 4. Build the dataset

```bash
python scripts/clean_corpora.py
python scripts/build_dataset.py            # see --help for knobs
```

The two knobs that shape the ghost's behavior:

- `--persona-token-prob` (default 0.5): fraction of examples carrying the
  persona token. Higher → stronger single-voice control, weaker blend.
  1.0 = no blended mode at all; 0.0 = one anonymous composite ghost.
- `--prompt-frac` (default 0.15): fraction carrying introspection prompts.
  Keep prompts *generic* (answerable by any entry). Pairing specific prompts
  with random entries teaches prompt-deafness.

If one corpus dwarfs the others, subsample it (see `subsample_pepys` —
stratify by year so you keep the whole life, not one random stretch).

## 5. Smoke test on CPU, then train on GPU

```bash
# laptop / CPU container — verifies the entire code path with a 135M model:
python scripts/train.py --config configs/smoke_test.yaml

# RunPod / real GPU:
pip install -r requirements.txt
python scripts/train.py --config configs/train_qwen3b.yaml
```

The trainer prints every effective hparam at start, loss curves during, and a
post-train inference grid (every persona, blended, seeded, prompted) with all
generation settings — so each run is a self-documenting experiment.

Starting hparams that behave well at this corpus size (~500K tokens):
lr 2e-5, 2 epochs, effective batch 16, cosine schedule, bf16. Watch val loss:
diaries are small data and the memorization knee arrives fast. If persona
voices smear together, raise `--persona-token-prob` and rebuild; if the blend
is boring, lower it.

## 6. Hold the séance

```bash
python scripts/sample.py --model outputs/ghost-qwen3b --date "7 June 2026"
python scripts/sample.py --model outputs/ghost-qwen3b --persona yourghost \
    --date "3 May 1921" --seed-text "I have done a terrible thing."
```

Things worth probing: Does a date in the ghost's living years summon
period-true content? What does the ghost do with a date after its death?
What emerges from the blend — does one voice dominate (rebalance the data),
or does something genuinely composite speak?

Treat the outputs as what they are: a statistical memory of a real person's
self-account, hallucinating new days. It's a memorial form, not a resurrection
— and that's the interesting part.
