#!/usr/bin/env python3
"""
train.py — Fine-tune a causal LM into the ghost diary.

Default target: full fine-tune of Qwen/Qwen2.5-3B on a single GPU
(A100/H100/4090-class; bf16 + gradient checkpointing). A LoRA fallback is one
flag away (--lora) if VRAM is tight.

Everything is driven by a YAML config (see configs/) and any key can be
overridden on the CLI:  python scripts/train.py --config configs/train_qwen3b.yaml \
                            --override learning_rate=1e-5 num_train_epochs=3

The script is deliberately LOUD: it prints the full resolved hyperparameter
set at start, again at save time, and stamps every post-training test
generation with both the training hparams and the generation hparams used.
"""

import subprocess
import sys

print("[init] Checking for required training libraries...")
try:
    import torch
    import yaml
    import datasets
    import transformers
    import peft
    import accelerate
except ImportError:
    print("[init] Missing libraries detected. Installing via pip now...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "transformers", "datasets", "pyyaml", "peft", "accelerate"
    ])
    print("[init] Libraries installed successfully!\n")
import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, set_seed)

BANNER = "=" * 78


# ------------------------------------------------------------ config

DEFAULTS = dict(
    model_name="Qwen/Qwen2.5-3B",
    dataset_dir="data/dataset",
    output_dir="outputs/ghost-qwen3b-base",
    seed=42,
    max_seq_len=2048,
    # optimization
    learning_rate=2e-5,
    embed_lr_multiplier=10.0,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    bf16=True,
    gradient_checkpointing=True,
    # eval/logging
    logging_steps=10,
    eval_steps=50,
    save_total_limit=2,
    # lora fallback
    lora=False,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    # post-train test inference
    run_test_inference=True,
    test_max_new_tokens=200,
    test_temperature=0.92,
    test_top_p=0.95,
    test_repetition_penalty=1.05,
    save=False,
)


def load_config() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--override", nargs="*", default=[],
                    help="key=value pairs overriding config")
    ap.add_argument("--lora", action="store_true", help="shortcut for lora=true")
    ap.add_argument("--save", action="store_true", help="Save the model weights at the end")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    
    # Auto-detect OPBDH environment variables
    if "OPBDH_MODEL_ID" in os.environ:
        cfg["model_name"] = os.environ["OPBDH_MODEL_ID"]
    if "OPBDH_RESULTS_DIR" in os.environ:
        cfg["output_dir"] = os.path.join(os.environ["OPBDH_RESULTS_DIR"], "ghost-qwen3b")

    if args.save:
        cfg["save"] = True

    if args.config:
        cfg.update(yaml.safe_load(Path(args.config).read_text()) or {})
    for kv in args.override:
        k, v = kv.split("=", 1)
        try:
            v = yaml.safe_load(v)  # parses numbers/bools/lists
        except Exception:
            pass
        cfg[k] = v
    if args.lora:
        cfg["lora"] = True
    return cfg


def shout(title: str, kv: dict):
    print(f"\n{BANNER}\n>>> {title}\n{BANNER}")
    for k, v in kv.items():
        print(f"    {k:32s} = {v}")
    print(BANNER, flush=True)


# ------------------------------------------------------------ data

def load_and_pack(cfg, tokenizer):
    """Tokenize each entry-document, append EOS, pack into max_seq_len blocks."""
    def docs(split):
        path = Path(cfg["dataset_dir"]) / f"{split}.jsonl"
        return [json.loads(l)["text"] for l in path.open()]

    def pack(texts):
        ids = []
        for t in texts:
            ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
            ids.append(tokenizer.eos_token_id)
        L = cfg["max_seq_len"]
        n_blocks = len(ids) // L
        blocks = [ids[i * L:(i + 1) * L] for i in range(n_blocks)]
        return Dataset.from_dict({"input_ids": blocks,
                                  "labels": [b[:] for b in blocks]})

    train, val = pack(docs("train")), pack(docs("val"))
    print(f"[data] packed: train={len(train)} blocks, val={len(val)} blocks "
          f"of {cfg['max_seq_len']} tokens "
          f"(~{len(train) * cfg['max_seq_len']:,} train tokens/epoch)")
    return train, val


# ------------------------------------------------------------ model

def build_model(cfg, tokenizer, special_tokens):
    dtype = torch.bfloat16 if cfg["bf16"] and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], dtype=dtype,
        attn_implementation="sdpa",
    )
    # Resize for the ghost tokens. NOTE: Qwen pads its embedding matrix beyond
    # the vocab (151936 rows vs 151665 tokens), so new token ids may land in
    # pre-existing padded rows — we therefore mean-init the ghost rows
    # explicitly by id rather than relying on matrix growth.
    old_n = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > old_n:
        model.resize_token_embeddings(len(tokenizer))
    ghost_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        vocab_mean = emb[:old_n].mean(dim=0)
        for tid in ghost_ids:
            emb[tid] = vocab_mean + torch.randn_like(vocab_mean) * 0.02
        out = model.get_output_embeddings()
        if out is not None and out.weight.data_ptr() != emb.data_ptr():
            for tid in ghost_ids:
                out.weight[tid] = emb[tid]
    print(f"[model] embedding rows={emb.shape[0]}; mean-initialized "
          f"{len(ghost_ids)} ghost token rows (ids {ghost_ids})")

    if cfg["lora"]:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                          lora_dropout=cfg["lora_dropout"],
                          target_modules=cfg["lora_target_modules"],
                          modules_to_save=["embed_tokens", "lm_head"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
    return model


# ------------------------------------------------------------ test inference

TEST_PROMPTS = [
    # --- CHUNK 1: Wonderful / Statement ---
    ("Wonderful / Blended",     "<|entry|>\n7 June.\nToday was a wonderful day. "),
    ("Wonderful / Pepys",       "<|entry|><|pepys|>\n7 June.\nToday was a wonderful day. "),
    ("Wonderful / Van Gogh",    "<|entry|><|vangogh|>\n7 June.\nToday was a wonderful day. "),
    ("Wonderful / Mansfield",   "<|entry|><|mansfield|>\n7 June.\nToday was a wonderful day. "),
    ("Wonderful / MacLane",     "<|entry|><|maclane|>\n7 June.\nToday was a wonderful day. "),

    # --- CHUNK 2: Interesting / Statement ---
    ("Interesting / Blended",     "<|entry|>\n14 October.\nToday was quite interesting. "),
    ("Interesting / Pepys",       "<|entry|><|pepys|>\n14 October.\nToday was quite interesting. "),
    ("Interesting / Van Gogh",    "<|entry|><|vangogh|>\n14 October.\nToday was quite interesting. "),
    ("Interesting / Mansfield",   "<|entry|><|mansfield|>\n14 October.\nToday was quite interesting. "),
    ("Interesting / MacLane",     "<|entry|><|maclane|>\n14 October.\nToday was quite interesting. "),

    # --- CHUNK 3: Disaster / Statement ---
    ("Disaster / Blended",     "<|entry|>\n14 October.\nToday was an absolute disaster. "),
    ("Disaster / Pepys",       "<|entry|><|pepys|>\n14 October.\nToday was an absolute disaster. "),
    ("Disaster / Van Gogh",    "<|entry|><|vangogh|>\n14 October.\nToday was an absolute disaster. "),
    ("Disaster / Mansfield",   "<|entry|><|mansfield|>\n14 October.\nToday was an absolute disaster. "),
    ("Disaster / MacLane",     "<|entry|><|maclane|>\n14 October.\nToday was an absolute disaster. "),
]


def test_inference(cfg, model, tokenizer, train_summary: dict):
    gen_kwargs = dict(
        max_new_tokens=cfg["test_max_new_tokens"],
        temperature=cfg["test_temperature"],
        top_p=cfg["test_top_p"],
        repetition_penalty=cfg["test_repetition_penalty"],
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    shout("POST-TRAIN TEST INFERENCE — generation hparams", gen_kwargs)
    shout("POST-TRAIN TEST INFERENCE — training hparams in effect", train_summary)
    model.eval()
    device = next(model.parameters()).device
    eot = tokenizer.convert_tokens_to_ids("<|/entry|>")
    eos_tokens = [t for t in [eot, tokenizer.eos_token_id] if t is not None]
    
    for name, prompt in TEST_PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, eos_token_id=eos_tokens,
                                 **gen_kwargs)
        text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=False)
        print(f"\n----- [{name}] -----")
        print(f"PROMPT  : {prompt!r}")
        print(f"OUTPUT  : {text}")
        print(f"(gen: temp={gen_kwargs['temperature']} top_p={gen_kwargs['top_p']} "
              f"rep_pen={gen_kwargs['repetition_penalty']} "
              f"max_new={gen_kwargs['max_new_tokens']})", flush=True)


# ------------------------------------------------------------ main

def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    t0 = time.time()

    meta = json.loads((Path(cfg["dataset_dir"]) / "meta.json").read_text())
    special_tokens = meta["special_tokens"]

    shout("GHOST DIARY TRAINING — resolved hyperparameters",
          {**cfg, "special_tokens": special_tokens,
           "cuda": torch.cuda.is_available(),
           "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens})
    print(f"[tok] added {n_added} special tokens; vocab now {len(tokenizer)}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = load_and_pack(cfg, tokenizer)
    model = build_model(cfg, tokenizer, special_tokens)

    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        seed=cfg["seed"],
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        max_grad_norm=cfg["max_grad_norm"],
        bf16=cfg["bf16"] and torch.cuda.is_available(),
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="no",
        save_total_limit=cfg["save_total_limit"],
        report_to=[],
    )

    embed_lr = float(cfg["learning_rate"]) * float(cfg.get("embed_lr_multiplier", 10.0))
    base_lr = float(cfg["learning_rate"])
    
    embed_params, base_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embed_tokens" in name or "lm_head" in name:
            embed_params.append(param)
        else:
            base_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {"params": embed_params, "lr": embed_lr},
        {"params": base_params, "lr": base_lr},
    ], weight_decay=cfg["weight_decay"])
    
    print(f"[optim] Differential LRs configured: Embeddings/Head @ {embed_lr:.1e}, Base @ {base_lr:.1e}")

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        optimizers=(optimizer, None)
    )

    eff_bs = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(train_ds) / eff_bs)
    print(f"[train] effective batch size = {eff_bs} "
          f"({steps_per_epoch} steps/epoch, "
          f"{steps_per_epoch * cfg['num_train_epochs']} total)")

    trainer.train()
    metrics = trainer.evaluate()
    ppl = math.exp(metrics["eval_loss"]) if metrics.get("eval_loss") else float("nan")
    print(f"[train] final eval_loss={metrics.get('eval_loss'):.4f}  ppl={ppl:.2f}")

    # ---- save
    if cfg["save"]:
        out = Path(cfg["output_dir"]) / "final"
        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))
        run_record = {**cfg, "final_eval_loss": metrics.get("eval_loss"),
                      "final_ppl": ppl, "train_blocks": len(train_ds),
                      "wallclock_sec": round(time.time() - t0, 1)}
        (out / "run_config.json").write_text(json.dumps(run_record, indent=2))
        shout("SAVED — final model + tokenizer + run_config.json", {"path": str(out)})
    else:
        print("\n[save] Skipping model save (--save flag not provided).")

    # ---- prove the ghost speaks
    if cfg["run_test_inference"]:
        train_summary = {k: cfg[k] for k in
                         ("model_name", "learning_rate", "num_train_epochs",
                          "per_device_train_batch_size",
                          "gradient_accumulation_steps", "max_seq_len",
                          "lr_scheduler_type", "warmup_ratio", "seed", "lora")}
        train_summary["final_eval_loss"] = metrics.get("eval_loss")
        test_inference(cfg, model, tokenizer, train_summary)


if __name__ == "__main__":
    main()
