#!/usr/bin/env bash
# Ghost Diary — RunPod bootstrap. Run from repo root on a fresh GPU pod.
set -euo pipefail
echo "=== ghost-diary runpod setup ==="
pip install -U torch transformers datasets accelerate peft pyyaml
python scripts/clean_corpora.py          # raw -> clean (raw texts ship in repo)
python scripts/build_dataset.py          # clean -> train/val jsonl
echo "=== launching full fine-tune (edit configs/train_qwen3b.yaml to taste) ==="
python scripts/train.py --config configs/train_qwen3b.yaml
echo "=== done. summon with: ==="
echo "python scripts/sample.py --model outputs/ghost-qwen3b/final --interactive"
