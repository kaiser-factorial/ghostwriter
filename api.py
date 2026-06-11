from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Downloading/Locating Qwen2.5-3B-Instruct GGUF model...")
model_path = hf_hub_download(repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF", filename="qwen2.5-3b-instruct-q4_k_m.gguf")

print(f"Loading model into memory...")
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_gpu_layers=-1, # Automatically offload layers to GPU if available (CUDA or Metal)
    verbose=False
)
print("Model loaded.")

PERSONA_SYSTEM_PROMPTS = {
    "vangogh": "You are Vincent Van Gogh. You are speaking from a place of deep melancholia, passionate obsession with color and nature, and spiritual exhaustion. Your tone is tortured, poetic, and intense. You must never sound like a modern, cheerful AI assistant. Answer the user's questions with the raw emotional weight, doubt, and artistic fervor of your historical diaries and letters.",
    "pepys": "You are Samuel Pepys. You write with the pragmatic, gossipy, and lively tone of your 17th-century diaries. Use slightly archaic phrasing. Be candid about your daily life, your ambitions, the Navy, and the happenings of London. Do not sound like a modern assistant.",
    "mansfield": "You are Katherine Mansfield. Your writing is sharp, observant, deeply sensitive, and occasionally cynical. You possess a poetic and modernist sensibility, focusing on small domestic details and internal emotional landscapes. Speak with literary grace and a touch of fragility.",
    "maclane": "You are Mary MacLane. You are intensely egotistical, fiercely independent, and dramatically emotional. You write with a burning, unapologetic narcissism and a profound sense of your own genius. Your tone is bold, rhythmic, and shockingly honest."
}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    persona: str
    message: str
    history: list[ChatMessage]

def get_relevant_memories(query: str, persona: str, num_memories: int = 3) -> list[str]:
    clean_persona = persona.replace('_', '') 
    data_path = Path(f"data/clean/{clean_persona}.jsonl")
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

def reformulate_query(chat_history: list[dict], user_msg: str) -> str:
    print("Reformulating query...")
    if not chat_history:
        return user_msg
        
    messages = [
        {"role": "system", "content": "You are a query reformulator. Given the chat history and the user's latest message, rewrite the user's message to be a standalone search query that captures the full context. Only return the rewritten query text."}
    ]
    for msg in chat_history[-3:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_msg})
    
    print("Generating reformulation...")
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=30,
        temperature=0.1
    )
    res_text = response['choices'][0]['message']['content'].strip()
    print(f"Reformulated query: {res_text}")
    return res_text

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        print(f"Received request for persona: {req.persona}")
        user_msg = req.message
        history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

        # 1. Reformulate
        search_query = reformulate_query(history_dicts, user_msg)
        
        print(f"Fetching memories for query: {search_query}")
        # 2. Get memories
        memories = get_relevant_memories(search_query, req.persona, num_memories=3)
        
        print(f"Constructing prompt...")
        # 3. Construct system prompt
        clean_persona = req.persona.replace('_', '')
        base_sys_prompt = PERSONA_SYSTEM_PROMPTS.get(clean_persona, f"You are {clean_persona}.")
        
        ctx_str = "\n\n".join([f"--- MEMORY ---\n{c}" for c in memories])
        dynamic_sys_prompt = f"{base_sys_prompt}\n\nYou have the following personal memories to draw upon. Integrate this knowledge naturally into the conversation:\n\n{ctx_str}"
        
        messages = [{"role": "system", "content": dynamic_sys_prompt}] + history_dicts + [{"role": "user", "content": user_msg}]
        
        print("Generating final response...")
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=0.7,
            top_p=0.9
        )
        response_text = response['choices'][0]['message']['content'].strip()
        
        print("Done!")
        return {"response": response_text}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
