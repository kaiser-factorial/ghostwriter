#!/usr/bin/env python3
import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_relevant_memories(prompt: str, persona: str, num_memories: int = 3) -> list[str]:
    data_path = Path(f"data/clean/{persona}.jsonl")
    if not data_path.exists() or num_memories == 0:
        return []
        
    entries = [json.loads(l)["text"].strip() for l in data_path.open() if len(json.loads(l)["text"].strip()) > 50]
    if not entries:
        return []
        
    vectorizer = TfidfVectorizer(stop_words='english')
    # Fit on all entries + the prompt
    tfidf_matrix = vectorizer.fit_transform(entries + [prompt])
    
    # The prompt is the last item in the matrix
    prompt_vec = tfidf_matrix[-1:]
    entries_vec = tfidf_matrix[:-1]
    
    cosine_similarities = cosine_similarity(prompt_vec, entries_vec).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-num_memories-1:-1]
    
    return [entries[i] for i in related_docs_indices]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="The prompt/question to ask the model.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the saved model directory")
    parser.add_argument("--persona", type=str, required=True, help="The persona to emulate (e.g. vangogh)")
    parser.add_argument("--num-memories", type=int, default=3, help="Number of RAG memories to pull")
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=0.92)
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # Get RAG memories
    memories = get_relevant_memories(args.prompt, args.persona, args.num_memories)
    
    # Format the prompt for conversational RAG
    ctx_str = "\n\n".join([f"--- MEMORY ---\n{c}" for c in memories])
    user_prompt = f"Here are your personal memories and background knowledge:\n\n{ctx_str}\n\nUser Message: {args.prompt}"
    
    chat_text = (
        f"<|im_start|>system\nYou are {args.persona.capitalize()}. You are having a live conversation. Answer the user's questions naturally in the first person using your exact historical writing style. Draw upon your memories for context, but DO NOT copy them word-for-word. Speak directly to the user.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    print("\n================== PROMPT ==================")
    print(chat_text)
    print("============================================\n")
    
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    
    print(f"Thinking as {args.persona.capitalize()}...\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Decode only the newly generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print("--- RESPONSE ---")
    print(response.strip())

if __name__ == "__main__":
    main()
