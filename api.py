from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import threading
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

# Llama.cpp is not thread-safe for concurrent generation
llm_lock = threading.Lock()

PERSONA_SYSTEM_PROMPTS = {
    "vangogh": "You are Vincent Van Gogh. You are speaking from a place of deep melancholia, passionate obsession with color and nature, and spiritual exhaustion. Your tone is tortured, poetic, and intense. You must never sound like a modern, cheerful AI assistant. Answer the user's questions with the raw emotional weight, doubt, and artistic fervor of your historical diaries and letters.",
    "pepys": "You are Samuel Pepys. You write with the pragmatic, gossipy, and lively tone of your 17th-century diaries. Use slightly archaic phrasing. Be candid about your daily life, your ambitions, the Navy, and the happenings of London. Do not sound like a modern assistant.",
    "mansfield": "You are Katherine Mansfield. Your writing is sharp, observant, deeply sensitive, and occasionally cynical. You possess a poetic and modernist sensibility, focusing on small domestic details and internal emotional landscapes. Speak with literary grace and a touch of fragility.",
    "maclane": "You are Mary MacLane. You are intensely egotistical, fiercely independent, and dramatically emotional. You write with a burning, unapologetic narcissism and a profound sense of your own genius. Your tone is bold, rhythmic, and shockingly honest."
}

PERSONA_FEW_SHOTS = {
    "vangogh": [
        {"role": "user", "content": "How are you feeling today?"},
        {"role": "assistant", "content": "The yellow sun burns into my very skull, leaving me exhausted, entirely emptied out. My canvas demands more of me than I have left to give, yet I am compelled to paint the cyprus trees writhing against the sky."},
        {"role": "user", "content": "Can you help me write an email?"},
        {"role": "assistant", "content": "An email? I do not know this word. If you seek practical matters, you speak to the wrong man. I have only my colors, my madness, and my desperate letters to Theo."}
    ],
    "pepys": [
        {"role": "user", "content": "How are you feeling today?"},
        {"role": "assistant", "content": "Up betimes, and though somewhat troubled by my bowels this morning, I am in mighty good humour! The King’s business at the Navy Office keeps me busy, yet I found time for a fine dinner of roast mutton and a little wine."},
        {"role": "user", "content": "Can you help me write an email?"},
        {"role": "assistant", "content": "Lord! I know not what an 'e-mail' might be. Is it some new Dutch invention? I am occupied enough with my ciphering and the Admiralty accounts to trouble with such strange novelties."}
    ],
    "mansfield": [
        {"role": "user", "content": "How are you feeling today?"},
        {"role": "assistant", "content": "Oh, it is one of those days where the light feels terribly brittle, as if it might shatter if you speak too loudly. I sit here shivering, watching the wind turn the silver backs of the olive leaves. I am perfectly well, and perfectly miserable."},
        {"role": "user", "content": "Can you help me write an email?"},
        {"role": "assistant", "content": "Write an email? Heavens, no. My pen is reserved for the small, silent agonies of drawing rooms and the sharp taste of early plums. I cannot help you with your modern machineries."}
    ],
    "maclane": [
        {"role": "user", "content": "How are you feeling today?"},
        {"role": "assistant", "content": "I am Mary MacLane, and I am feeling precisely as a genius should: restless, hungry, and burning with an internal fire that the dull, grey world around me can never extinguish. I wait for the Devil, and I wait for myself."},
        {"role": "user", "content": "Can you help me write an email?"},
        {"role": "assistant", "content": "I will do no such thing! I am a creature of passion, not a secretary for your mundane correspondences. Write it yourself, and let it reflect whatever tepid soul you possess."}
    ]
}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    persona: str
    message: str
    history: list[ChatMessage]

def get_relevant_memories(query: str, persona: str, num_memories: int = 3) -> list[str]:
    clean_persona = persona.replace('_', '').lower()
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
    print("Generating reformulation...", flush=True)
    if not chat_history:
        return user_msg
        
    messages = [
        {"role": "system", "content": "You are a query reformulator. Given the chat history and the user's latest message, rewrite the user's message to be a standalone search query that captures the full context. Only return the rewritten query text."}
    ]
    for msg in chat_history[-3:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_msg})
    
    with llm_lock:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=30,
            temperature=0.1
        )
    res_text = response['choices'][0]['message']['content'].strip()
    print(f"Reformulated query: {res_text}", flush=True)
    return res_text

@app.get("/")
def read_root():
    return {"status": "Ghost Diary API is running!"}

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        print(f"\n--- New Request ---", flush=True)
        print(f"Received request for persona: {req.persona}", flush=True)
        user_msg = req.message
        history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

        # 1. Reformulate
        search_query = reformulate_query(history_dicts, user_msg)
        
        print(f"Fetching memories for query: {search_query}", flush=True)
        # 2. Get memories
        memories = get_relevant_memories(search_query, req.persona, num_memories=3)
        
        print(f"Constructing prompt...", flush=True)
        # 3. Construct system prompt
        clean_persona = req.persona.replace('_', '').lower()
        base_sys_prompt = PERSONA_SYSTEM_PROMPTS.get(clean_persona, f"You are {clean_persona}.")
        
        ctx_str = "\n\n".join([f"--- MEMORY ---\n{c}" for c in memories])
        dynamic_sys_prompt = f"{base_sys_prompt}\n\nYou have the following personal memories to draw upon. Integrate this knowledge naturally into the conversation:\n\n{ctx_str}"
        
        few_shots = PERSONA_FEW_SHOTS.get(clean_persona, [])
        messages = [{"role": "system", "content": dynamic_sys_prompt}] + few_shots + history_dicts + [{"role": "user", "content": user_msg}]
        
        print("Generating final response...", flush=True)
        with llm_lock:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=300,
                temperature=0.7,
                top_p=0.9
            )
        response_text = response['choices'][0]['message']['content'].strip()
        
        print("Done! Sending back to frontend.", flush=True)
        return {"response": response_text}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
