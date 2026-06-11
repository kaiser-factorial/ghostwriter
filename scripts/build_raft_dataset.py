#!/usr/bin/env python3
import json
import random
import argparse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PERSONAS = ["pepys", "vangogh", "mansfield", "maclane"]

GENERIC_PROMPTS = [
    "What is on your mind today?",
    "Write about your day.",
    "What happened today?",
    "How are you feeling, honestly?",
    "What do you keep returning to?",
    "Set down the truth of this day.",
]

def format_raft_prompt(question: str, context_chunks: list[str]) -> str:
    # Formats the RAG context and the question
    ctx_str = "\n\n".join([f"--- MEMORY ---\n{c}" for c in context_chunks])
    return f"You are a specific historical figure. You have the following memories to draw upon:\n\n{ctx_str}\n\nBased on your memories and persona, answer the following prompt:\n{question}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="data/clean", type=Path)
    ap.add_argument("--out-dir", default="data/raft_dataset", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-contexts", type=int, default=3)
    ap.add_argument("--personas", type=str, default="pepys,vangogh,mansfield,maclane", 
                    help="Comma-separated list of personas to include")
    args = ap.parse_args()
    
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raft_data = []
    
    selected_personas = [p.strip() for p in args.personas.split(',')]

    for p in selected_personas:
        if p not in PERSONAS:
            print(f"Warning: Skipping unknown persona '{p}'")
            continue
        entries = [json.loads(l)["text"].strip() for l in (args.clean_dir / f"{p}.jsonl").open() if len(json.loads(l)["text"].strip()) > 50]
        
        if len(entries) < args.num_contexts + 1:
            continue

        # Use TF-IDF to find semantically similar entries to act as "Retrieved Context"
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(entries)
        
        for idx, target_entry in enumerate(entries):
            # Find similar entries to act as the retrieved memory
            cosine_similarities = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
            
            # Get top indices, excluding the target entry itself
            related_docs_indices = cosine_similarities.argsort()[:-args.num_contexts-2:-1]
            related_docs_indices = [i for i in related_docs_indices if i != idx][:args.num_contexts]
            
            memories = [entries[i] for i in related_docs_indices]
            question = rng.choice(GENERIC_PROMPTS)
            
            user_prompt = format_raft_prompt(question, memories)
            
            # ChatML format string for Qwen
            chat_text = (
                f"<|im_start|>system\nYou are {p.capitalize()}. You must emulate their exact speaking and writing style.<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{target_entry}<|im_end|>\n"
            )
            
            raft_data.append({"text": chat_text, "persona": p})

    # Shuffle and split
    rng.shuffle(raft_data)
    split = int(len(raft_data) * 0.95)
    train, val = raft_data[:split], raft_data[split:]

    for name, docs in [("train", train), ("val", val)]:
        with (args.out_dir / f"{name}.jsonl").open("w") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Write meta.json so train.py doesn't crash when looking for special tokens
    (args.out_dir / "meta.json").write_text(json.dumps({"special_tokens": []}))

    print(f"✅ Generated RAFT dataset: {len(train)} train, {len(val)} val.")

if __name__ == "__main__":
    main()
