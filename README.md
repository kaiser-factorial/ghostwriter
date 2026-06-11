---
title: Ghostwriter API
emoji: 👻
colorFrom: purple
colorTo: gray
sdk: docker
app_file: api.py
pinned: false
---

# Ghostwriter 👻📓

Fine-tune one small model on four dead diarists. Give it a date; a ghost
writes the entry. With a persona token you summon one writer; without it,
a blended composite of all four speaks.

The ghosts: **Samuel Pepys** (1660s London) · **Vincent van Gogh** (1880s
letters) · **Katherine Mansfield** (1910s–20s journal) · **Mary MacLane**
(1902/1917 confessions). All US public domain — see `PROVENANCE.md`.

## Quick start (RunPod or any CUDA box)

```bash
git clone <this repo> && cd Ghostwriter
pip install -r requirements.txt

# data is already built (data/dataset/), but to rebuild from raw:
python scripts/clean_corpora.py
python scripts/build_dataset.py

# train — Qwen2.5-3B full fine-tune, ~minutes on an A100:
python scripts/train.py --config configs/train_qwen3b.yaml
# (smaller GPU? configs/train_qwen3b_lora.yaml)

# hold the séance:
python scripts/sample.py --model outputs/ghost-qwen3b --date "7 June 2026"
python scripts/sample.py --model outputs/ghost-qwen3b --persona maclane \
    --date "7 June 2026" --seed-text "Today was hard."
python scripts/sample.py --model outputs/ghost-qwen3b \
    --prompt "What do you keep refusing to look at?"
```

No GPU handy? `python scripts/train.py --config configs/smoke_test.yaml`
runs the entire pipeline on CPU with a 135M model in a few minutes.

## Docs

- `SPEC.md` — what this is, how it works, next steps
- `HOW_TO.md` — raise your own ghost (general recipe + the copyright séance rules)
- `CLAUDE_IDEAS.md` — implemented extras and the speculative backlog
- `PROVENANCE.md` — sources, editions, and the ghosts we couldn't legally summon

Every training run prints its full hyperparameters, and post-train test
inference prints prompt, output, and all generation settings — runs are
self-documenting by design.
