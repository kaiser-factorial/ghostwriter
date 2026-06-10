#!/usr/bin/env python3
"""
build_dataset.py — Turn cleaned entries into a dual-mode training dataset.

Modes taught simultaneously (control-token dropout):
  * PERSONA mode: a persona token (<|pepys|> etc.) precedes the entry; the model
    learns conditional voice generation.
  * BLENDED mode: no persona token; the model marginalizes over all four ghosts
    and a composite voice emerges.
Each entry is emitted with the persona token present with probability
--persona-token-prob (default 0.5).

Entry document format (one training document per entry):

    <|entry|><|maclane|>
    19 January 1917.
    <text...>
    <|/entry|>

A fraction of examples (--prompt-frac) carry a generic introspection prompt:

    <|entry|><|mansfield|>
    [Prompt: What do you keep refusing to look at?]
    4 March 1920.
    <text...>
    <|/entry|>

Inference recipes (see sample.py):
    blended, fresh:   "<|entry|>\n7 June 2026.\n"
    persona, seeded:  "<|entry|><|vangogh|>\n7 June 2026.\nToday was hard."
    prompted:         "<|entry|>\n[Prompt: <your prompt>]\n7 June 2026.\n"

Balancing: Pepys is ~10x the other corpora; entries are stratified-sampled by
year down to --pepys-char-budget characters so one ghost doesn't possess the
other three.

Usage: python scripts/build_dataset.py [--seed 42] [--val-frac 0.02] ...
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

PERSONAS = ["pepys", "vangogh", "mansfield", "maclane"]
ENTRY_OPEN, ENTRY_CLOSE = "<|entry|>", "<|/entry|>"
PERSONA_TOKENS = {p: f"<|{p}|>" for p in PERSONAS}
SPECIAL_TOKENS = [ENTRY_OPEN, ENTRY_CLOSE, *PERSONA_TOKENS.values()]

# Generic introspection prompts: deliberately answerable by *any* diary entry,
# so prompt-conditioning stays truthful. (Specific prompts paired with random
# entries would teach the model to ignore the prompt.)
GENERIC_PROMPTS = [
    "What is on your mind today?",
    "Write about your day.",
    "What happened today?",
    "How are you feeling, honestly?",
    "What do you keep returning to?",
    "Set down the truth of this day.",
    "What did you notice today that no one else did?",
    "What are you afraid of right now?",
    "What do you want?",
    "Write until something true appears.",
    "What would you say if no one could ever read this?",
    "Describe where you are.",
    "What is the weather inside you?",
    "What are you working on, and how does it go?",
    "Who occupied your thoughts today?",
    "What small thing mattered today?",
    "Confess something.",
    "What do you keep refusing to look at?",
    "Make an account of yourself.",
    "What does this day deserve to have remembered of it?",
]


def normalize_date(date: str | None) -> str:
    if not date:
        return "An unmarked day"
    d = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date)  # 16th -> 16
    return d.strip()


def format_doc(entry: dict, use_persona: bool, prompt: str | None) -> str:
    parts = [ENTRY_OPEN]
    if use_persona:
        parts.append(PERSONA_TOKENS[entry["persona"]])
    parts.append("\n")
    if prompt:
        parts.append(f"[Prompt: {prompt}]\n")
    parts.append(f"{normalize_date(entry['date'])}.\n")
    if entry.get("title"):
        parts.append(f"{entry['title']}\n")
    parts.append(entry["text"].strip())
    parts.append(f"\n{ENTRY_CLOSE}")
    return "".join(parts)


def subsample_pepys(entries: list[dict], char_budget: int, rng: random.Random) -> list[dict]:
    """Stratified by year so the whole 1660s decade survives the cut."""
    by_year = defaultdict(list)
    for e in entries:
        m = re.search(r"(\d{4})", e["date"] or "")
        by_year[m.group(1) if m else "?"].append(e)
    for v in by_year.values():
        rng.shuffle(v)
    picked, used = [], 0
    # round-robin across years until budget exhausted
    pools = list(by_year.values())
    i = 0
    while used < char_budget and any(pools):
        pool = pools[i % len(pools)]
        if pool:
            e = pool.pop()
            picked.append(e)
            used += len(e["text"])
        i += 1
        if all(not p for p in pools):
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="data/clean", type=Path)
    ap.add_argument("--out-dir", default="data/dataset", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--persona-token-prob", type=float, default=0.5)
    ap.add_argument("--prompt-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--pepys-char-budget", type=int, default=900_000,
                    help="Max characters of Pepys to keep (he is 10x the others raw)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    docs_train, docs_val, stats = [], [], {}
    for p in PERSONAS:
        entries = [json.loads(l) for l in (args.clean_dir / f"{p}.jsonl").open()]
        if p == "pepys":
            entries = subsample_pepys(entries, args.pepys_char_budget, rng)
        rng.shuffle(entries)
        n_val = max(2, int(len(entries) * args.val_frac))
        val, train = entries[:n_val], entries[n_val:]
        for split_entries, sink in ((train, docs_train), (val, docs_val)):
            for e in split_entries:
                use_persona = rng.random() < args.persona_token_prob
                prompt = rng.choice(GENERIC_PROMPTS) if rng.random() < args.prompt_frac else None
                sink.append({
                    "text": format_doc(e, use_persona, prompt),
                    "persona": e["persona"],
                    "has_persona_token": use_persona,
                    "has_prompt": prompt is not None,
                })
        chars = sum(len(e["text"]) for e in entries)
        stats[p] = {"entries": len(entries), "chars": chars,
                    "train": len(train), "val": len(val)}
        print(f"[dataset] {p:10s} kept={len(entries):4d} chars={chars:8,d} "
              f"(train {len(train)} / val {len(val)})")

    rng.shuffle(docs_train)
    for name, docs in (("train", docs_train), ("val", docs_val)):
        with (args.out_dir / f"{name}.jsonl").open("w") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    n_tok = sum(s["chars"] for s in stats.values()) // 4
    meta = {"personas": PERSONAS, "special_tokens": SPECIAL_TOKENS,
            "persona_token_prob": args.persona_token_prob,
            "prompt_frac": args.prompt_frac, "seed": args.seed,
            "approx_tokens": n_tok, "stats": stats,
            "generic_prompts": GENERIC_PROMPTS}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[dataset] train={len(docs_train)} val={len(docs_val)} docs, "
          f"~{n_tok:,} tokens -> {args.out_dir}")


if __name__ == "__main__":
    main()
