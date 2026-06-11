#!/usr/bin/env python3
import argparse
import json
import torch
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PERSONA_SYSTEM_PROMPTS = {
    "vangogh": "You are Vincent Van Gogh. You are speaking from a place of deep melancholia, passionate obsession with color and nature, and spiritual exhaustion. Your tone is tortured, poetic, and intense. You must never sound like a modern, cheerful AI assistant. Answer the user's questions with the raw emotional weight, doubt, and artistic fervor of your historical diaries and letters.",
    "pepys": "You are Samuel Pepys. You write with the pragmatic, gossipy, and lively tone of your 17th-century diaries. Use slightly archaic phrasing. Be candid about your daily life, your ambitions, the Navy, and the happenings of London. Do not sound like a modern assistant.",
    "mansfield": "You are Katherine Mansfield. Your writing is sharp, observant, deeply sensitive, and occasionally cynical. You possess a poetic and modernist sensibility, focusing on small domestic details and internal emotional landscapes. Speak with literary grace and a touch of fragility.",
    "maclane": "You are Mary MacLane. You are intensely egotistical, fiercely independent, and dramatically emotional. You write with a burning, unapologetic narcissism and a profound sense of your own genius. Your tone is bold, rhythmic, and shockingly honest."
}

def get_relevant_memories(query: str, persona: str, num_memories: int = 3) -> list[str]:
    data_path = Path(f"data/clean/{persona}.jsonl")
    if not data_path.exists() or num_memories == 0:
        return []
        
    entries = [json.loads(l)["text"].strip() for l in data_path.open() if len(json.loads(l)["text"].strip()) > 50]
    if not entries:
        return []
        
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(entries + [query])
    
    prompt_vec = tfidf_matrix[-1:]
    entries_vec = tfidf_matrix[:-1]
    
    cosine_similarities = cosine_similarity(prompt_vec, entries_vec).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-num_memories-1:-1]
    
    return [entries[i] for i in related_docs_indices]

def reformulate_query(tokenizer, model, chat_history: list[dict], user_msg: str) -> str:
    # If no history, the query is just the message
    if not chat_history:
        return user_msg
        
    # Ask the LLM to reformulate the query
    history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-4:]])
    prompt = f"<|im_start|>system\nYou are a helpful search assistant. Read the chat history and the user's latest message. Rewrite the user's latest message into a standalone search query so we can search a historical diary database for relevant context. Return ONLY the search query.<|im_end|>\n<|im_start|>user\nCHAT HISTORY:\n{history_str}\n\nLATEST MESSAGE: {user_msg}\n\nSTANDALONE SEARCH QUERY:<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
        
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    search_query = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return search_query

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Path or HF repo")
    parser.add_argument("--persona", type=str, required=True, help="The persona to emulate")
    parser.add_argument("--num-memories", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="auto")
    
    sys_prompt_base = PERSONA_SYSTEM_PROMPTS.get(args.persona, f"You are {args.persona}. Emulate their exact writing style. Do not act like an AI.")
    
    chat_history = []
    
    print(f"\n========================================================")
    print(f"  SEANCE BEGUN: You are now chatting with {args.persona.upper()}")
    print(f"  (Type 'quit' or 'exit' to leave)")
    print(f"========================================================\n")
    
    while True:
        try:
            user_msg = input("\nYou: ")
            if user_msg.lower() in ['quit', 'exit']:
                break
            if not user_msg.strip():
                continue
                
            # 1. Reformulate query based on history
            search_query = reformulate_query(tokenizer, model, chat_history, user_msg)
            print(f"\n[System: Searching diaries for '{search_query}']")
            
            # 2. Retrieve memories
            memories = get_relevant_memories(search_query, args.persona, args.num_memories)
            ctx_str = "\n\n".join([f"--- MEMORY ---\n{c}" for c in memories])
            
            # 3. Construct the injected System Prompt
            dynamic_sys_prompt = f"{sys_prompt_base}\n\nYou have the following personal memories to draw upon. Integrate this knowledge naturally into the conversation:\n\n{ctx_str}"
            
            # 4. Build the ChatML payload
            messages = [{"role": "system", "content": dynamic_sys_prompt}] + chat_history + [{"role": "user", "content": user_msg}]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            print(f"{args.persona.capitalize()}: ", end="", flush=True)
            
            with torch.no_grad():
                # We'll just generate the full output and print it
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.85, # Slightly lower than 0.92 to prevent derailing, but keeps it creative
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            print(response)
            
            # Update history
            chat_history.append({"role": "user", "content": user_msg})
            chat_history.append({"role": "assistant", "content": response})
            
            # Keep history from getting too long (keep last 6 turns = 12 messages)
            if len(chat_history) > 12:
                chat_history = chat_history[-12:]
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
