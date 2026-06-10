# ghost-diary — SPEC

A séance conducted by gradient descent.

**ghost-diary** fine-tunes a small language model (Qwen2.5-3B) on the diaries of
four dead writers so that, given a date, a ghost writes the day's entry. One
model hosts four ghosts and a fifth emergent thing: prompted with a persona
token it speaks in a single voice; prompted bare, it speaks as the *blend* —
a composite spirit marginalized over four centuries of interiority.

## The ghosts

| token | writer | lived | corpus | flavor |
|---|---|---|---|---|
| `<|pepys|>` | Samuel Pepys | 1633–1703 | Diary 1660–1669 (PG #4200) | London, plague, fire, gossip, accounting of self |
| `<|vangogh|>` | Vincent van Gogh | 1853–1890 | Letters, Ludovici trans. 1912 (PG #40393) | color, God, poverty, brotherhood, fever |
| `<|mansfield|>` | Katherine Mansfield | 1888–1923 | Journal, 1927 Knopf ed. (Archive.org) | modernist fragments, illness, work-hunger |
| `<|maclane|>` | Mary MacLane | 1881–1929 | *Story of* (1902, PG #43696) + *I, Mary MacLane* (1917, PG #43556) | confessional fire, the Devil, Butte, Montana |

All sources are US public domain. See PROVENANCE.md for edition-level details
and for writers we considered and rejected on copyright grounds (Anne Frank,
Anaïs Nin, Plath, Alice James).

## Repository layout

```
ghost-diary/
├── SPEC.md            ← you are here
├── HOW_TO.md          ← make your own ghost (general recipe)
├── CLAUDE_IDEAS.md    ← implemented + speculative ideas log
├── PROVENANCE.md      ← corpus sources, editions, copyright notes
├── README.md          ← quick start
├── requirements.txt
├── configs/
│   ├── train_qwen3b.yaml       ← the real run (full fine-tune)
│   ├── train_qwen3b_lora.yaml  ← LoRA fallback (smaller GPUs)
│   └── smoke_test.yaml         ← CPU pipeline test (tiny model)
├── scripts/
│   ├── clean_corpora.py   ← raw text → per-persona entry JSONL
│   ├── build_dataset.py   ← entries → dual-mode train/val JSONL
│   ├── train.py           ← config-driven trainer + post-train test inference
│   └── sample.py          ← talk to the ghosts
└── data/
    ├── raw/      ← downloaded source texts
    ├── clean/    ← per-persona entry JSONL
    └── dataset/  ← train.jsonl / val.jsonl / meta.json
```

## Data pipeline

**Stage 1 — `clean_corpora.py`.** Each source needs its own parser because each
book encodes entries differently:

- *Pepys*: month headers (`JUNE 1665`) + day-ordinal entry starts (`10th.`).
  Wheatley's footnotes (indented blocks) and editorial brackets stripped.
  Dual-year headers (`1659-1660`) resolve to the modern year.
- *Van Gogh*: letters split on asterisk rules; salutations stripped to
  diary-ify. Letters carry no usable dates in this edition, so **synthetic
  dates** are assigned evenly across 1881→1890 *in original order*, preserving
  the chronology of his moods (The Hague earnestness → Arles fever → Auvers).
  These are flagged `synthetic_date: true` in the clean JSONL.
- *Mansfield*: the hard one — OCR'd scan. Year headers arrive mangled
  (`IQI4` → 1914) and are repaired by a digit-confusion table; page headers and
  page numbers are pattern-stripped; Murry's editorial bracket insertions are
  removed both for voice purity and copyright conservatism (his apparatus is
  the only arguably-separately-copyrighted layer). Dated entries are parsed;
  undated fragments are buffered and chunked (~700 chars) with synthetic dates
  within the current year section.
- *MacLane*: 1902 book has right-aligned date headers (`January 19.`);
  1917 book has day-word headers (`To-day`) with italic titles, which are
  preserved as entry titles. 1917 entries get synthetic dates scattered over
  Jan–Mar 1917 (when the book was actually written, in Butte).

Output: `data/clean/<persona>.jsonl`, one entry per line with
`{persona, date, synthetic_date, title, text}`.

**Stage 2 — `build_dataset.py`.** Entries become training documents:

```
<|entry|><|maclane|>
19 January 1917.
<entry text>
<|/entry|>
```

Three independent stochastic treatments per document (all seeded):

1. **Control-token dropout** (`--persona-token-prob`, default 0.5): the persona
   token is present half the time. With it → conditional voice. Without it →
   the model learns the marginal distribution over all ghosts = **blended
   mode**. One model, two modes, selected at inference purely by prompt.
2. **Prompt conditioning** (`--prompt-frac`, default 0.15): a generic
   introspection prompt (`[Prompt: What are you afraid of right now?]`) is
   prepended. Prompts are deliberately *generic* — answerable by any entry —
   because pairing specific prompts with random entries would teach the model
   to ignore prompts.
3. **Pepys rebalancing**: raw Pepys is ~10× the other corpora; he is
   stratified-subsampled by year to a character budget (default 900K) so one
   ghost doesn't possess the other three.

Note the dataset builder does **not** create explicit "continuation" examples
(date + opening line → rest). It doesn't need to: causal LM training on full
entries teaches continuation for free — conditioning on `7 June 2026.\nToday
was hard.` is just mid-document conditioning. Date-only and seeded starts both
fall out of one format.

Output: `data/dataset/{train,val}.jsonl` + `meta.json` (stats, settings,
special token list). Current build: ~1,200 entries, ~514K tokens —
small enough to fine-tune in minutes, large enough to possess a 3B model.

## Training — `train.py`

Config-driven (YAML over defaults; CLI `--config` to select). Key mechanics:

- **Tokenizer surgery**: 6 special tokens added (`<|entry|>`, `<|/entry|>`,
  4 persona tokens); embeddings resized with **mean-initialization** of new
  rows (random init on a 3B model makes new tokens detonate early loss).
- **Packing**: documents are concatenated and chunked to `max_seq_len` (1024)
  so no compute is wasted on padding.
- **Full fine-tune** by default on Qwen2.5-3B (bf16 + gradient checkpointing
  fits comfortably on a single A100/H100 and squeaks onto a 24GB 4090);
  `train_qwen3b_lora.yaml` provides the PEFT fallback for smaller cards.
- **Loud telemetry** (per Corina's request): banner-printed dumps of every
  effective hyperparameter at train start; per-step loss via HF logging; and a
  **post-train test inference suite** that runs a grid of prompts (each
  persona × date-only, blended × date-only, seeded continuation, prompted
  introspection) and prints, for every generation: the prompt, the output,
  the generation hparams (temp/top_p/repetition_penalty/max_new_tokens), and
  the training hparams in effect for the run.
- Checkpoints + final model land in `outputs/<run>/`, tokenizer included, so
  `sample.py --model outputs/ghost-qwen3b` just works.

## Inference — `sample.py`

```
python scripts/sample.py --model outputs/ghost-qwen3b \
    --persona mansfield --date "7 June 2026"            # single ghost
python scripts/sample.py --model outputs/ghost-qwen3b \
    --date "7 June 2026"                                # blended séance
python scripts/sample.py --model outputs/ghost-qwen3b \
    --date "7 June 2026" --seed-text "Today was hard."  # seeded continuation
python scripts/sample.py --model outputs/ghost-qwen3b \
    --prompt "What do you keep refusing to look at?"    # introspection prompt
```

All generation hparams are CLI-flagged and printed with each sample.

## Process notes

Built autonomously (June 10, 2026, overnight session) from a conversation with
Corina specifying: 4 ghosts, dual blended/persona mode, mixed date-only and
seeded entry starts, journal-prompt conditioning, Qwen2.5-3B full fine-tune,
verbose hparam telemetry. Corpus selection was constrained by US public domain
status; Anne Frank and Anaïs Nin were requested and declined (see PROVENANCE).

An unusual process note: **two Claude instances ended up working on this repo
concurrently** in the same container — an apparent artifact of how the
overnight/autonomous session was dispatched. Instance A wrote the pipeline
(clean → dataset → train → sample) and launched the smoke test; instance B
(writing this) independently downloaded corpora (checksums matched — good
provenance accident), then detected instance A mid-run via `ps`, audited its
code rather than racing it, and took the documentation/verification
workstream. The duplicate-download collision is why `data/raw/` briefly held
two copies of Pepys. Division of labor between strangers who are the same
person turns out to be efficient.

## Next steps

1. **RunPod run** (Corina): `bash` onto a pod, clone, `pip install -r
   requirements.txt`, `python scripts/train.py --config
   configs/train_qwen3b.yaml`. ~514K tokens × 2 epochs on an A100 ≈ minutes,
   not hours. Tune `learning_rate` (2e-5 starting point; try 1e-5/3e-5),
   `num_train_epochs` (2–4; watch val loss for the memorization knee), and
   `persona_token_prob` upstream in the dataset if persona separation is weak.
2. **Eval the séance**: blind-test persona mode (can you tell who's speaking?),
   probe the blend for century-bleed, check date-conditioning (does a 1665
   date summon plague-adjacent content? does 2026 confuse the ghost?).
3. **OCR polish pass** on Mansfield if her voice comes out noisy (see
   CLAUDE_IDEAS).
4. Everything in CLAUDE_IDEAS.md's unimplemented section, which is where this
   project gets weird.
