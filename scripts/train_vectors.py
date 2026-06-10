import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from repeng import ControlVector, ControlModel, DatasetEntry

# --- CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Change this to your exact Qwen variant!
LAYER_RANGE = list(range(-5, -15, -1)) # Extract vectors from these layers
SAMPLES_PER_PERSONA = 200 # How many examples to use for training

def load_corporate_emails(limit=500):
    print("Downloading corporate emails for negative baseline...")
    # AESLC dataset contains Enron corporate emails
    dataset = load_dataset("aeslc", split="train")
    emails = []
    for row in dataset:
        body = row['email_body'].strip()
        if len(body) > 100: # Filter out too-short emails
            emails.append(body)
            if len(emails) >= limit:
                break
    return emails

def load_persona_entries(persona_name, limit=200):
    path = f"data/clean/{persona_name}.jsonl"
    entries = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if len(data['text']) > 100:
                entries.append(data['text'].strip())
                if len(entries) >= limit:
                    break
    return entries

def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Qwen may not have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    model = ControlModel(model, LAYER_RANGE)

    corporate_emails = load_corporate_emails(limit=1000)

    personas = ['van_gogh', 'mansfield', 'pepys', 'maclane']

    for persona in personas:
        print(f"\n--- Extracting Control Vector for {persona} ---")
        positive_texts = load_persona_entries(persona, limit=SAMPLES_PER_PERSONA)
        
        # Ensure we have enough negative examples
        negative_texts = corporate_emails[:len(positive_texts)]

        if len(positive_texts) == 0:
            print(f"Skipping {persona}, no data found.")
            continue

        # Create pairs!
        dataset = []
        for pos, neg in zip(positive_texts, negative_texts):
            # We add the ChatML format just in case the model is an Instruct model.
            # But since we are extracting raw stylistic vectors, raw text works too!
            # We'll just pass the raw text.
            dataset.append(DatasetEntry(positive=pos, negative=neg))

        print(f"Training on {len(dataset)} positive/negative pairs...")
        # Train the vector!
        vector = ControlVector.train(model, tokenizer, dataset, max_batch_size=8)
        
        # Save it
        out_name = f"{persona}_vector.gguf"
        vector.export_gguf(out_name)
        print(f"✅ Saved {out_name}")

if __name__ == "__main__":
    main()
