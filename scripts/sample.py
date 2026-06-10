#!/usr/bin/env python3
"""
sample.py — Summon the ghost.

Modes:
  blended   : no persona token; the composite voice answers.
  persona   : --persona pepys|vangogh|mansfield|maclane
  seeded    : add --seed-text "Today was hard." after the date.
  prompted  : add --prompt "What are you afraid of?"

Examples:
  python scripts/sample.py --model outputs/ghost-qwen3b/final --date "7 June 2026"
  python scripts/sample.py --model ... --persona maclane --date "13 January 2026" \
      --seed-text "The devil came again."
  python scripts/sample.py --model ... --prompt "What do you want?" -n 3
  python scripts/sample.py --model ... --interactive

Every generation prints the full set of sampling hparams used.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PERSONAS = ["pepys", "vangogh", "mansfield", "maclane"]


def build_prompt(args) -> str:
    s = "<|entry|>"
    if args.persona:
        s += f"<|{args.persona}|>"
    s += "\n"
    if args.prompt:
        s += f"[Prompt: {args.prompt}]\n"
    s += f"{args.date}.\n"
    if args.seed_text:
        s += args.seed_text
    return s


def generate(model, tokenizer, prompt, args):
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature, top_p=args.top_p,
                      repetition_penalty=args.repetition_penalty,
                      do_sample=True, pad_token_id=tokenizer.eos_token_id)
    eot = tokenizer.convert_tokens_to_ids("<|/entry|>")
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, eos_token_id=[eot, tokenizer.eos_token_id],
                             **gen_kwargs)
    text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=False)
    text = text.replace("<|/entry|>", "").replace(tokenizer.eos_token, "").rstrip()
    print(f"\n┌─ PROMPT {'─' * 60}\n{prompt}")
    print(f"├─ GHOST {'─' * 61}\n{text}")
    print(f"└─ hparams: temp={args.temperature} top_p={args.top_p} "
          f"rep_pen={args.repetition_penalty} max_new={args.max_new_tokens}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--persona", choices=PERSONAS, default=None)
    ap.add_argument("--date", default="7 June 2026")
    ap.add_argument("--seed-text", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("-n", "--num-samples", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {args.model} (dtype={dtype}, device={device})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    if args.interactive:
        print("Séance open. Commands: /persona <name|off>, /date <d>, /temp <t>, /quit")
        print("Anything else is used as seed text after the date (empty = fresh).")
        while True:
            try:
                line = input("👻 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line == "/quit":
                break
            elif line.startswith("/persona"):
                v = line.split(maxsplit=1)[1] if " " in line else "off"
                args.persona = None if v == "off" else v
                print(f"[persona = {args.persona}]")
            elif line.startswith("/date"):
                args.date = line.split(maxsplit=1)[1]
            elif line.startswith("/temp"):
                args.temperature = float(line.split(maxsplit=1)[1])
            else:
                args.seed_text = line or None
                generate(model, tokenizer, build_prompt(args), args)
    else:
        for _ in range(args.num_samples):
            generate(model, tokenizer, build_prompt(args), args)


if __name__ == "__main__":
    main()
