import json
import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading (env-configurable; defaults to 3B).
# For a ~2x speedup on the free CPU tier, set in the Space settings:
#   MODEL_REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF
#   MODEL_FILE=qwen2.5-1.5b-instruct-q4_k_m.gguf
# ---------------------------------------------------------------------------
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf")

print(f"Downloading/Locating {MODEL_REPO} :: {MODEL_FILE} ...")
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

print("Loading model into memory...")
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=int(os.environ.get("N_THREADS", 2)), # Prevent CPU thrashing on HF Spaces
    n_gpu_layers=-1,  # Automatically offload layers to GPU if available
    verbose=False,
)
print("Model loaded.")

# Llama.cpp is not thread-safe for concurrent generation
llm_lock = threading.Lock()

PERSONA_SYSTEM_PROMPTS = {
    "vangogh": "You are Vincent Van Gogh. You are speaking from a place of deep melancholia, passionate obsession with color and nature, and spiritual exhaustion. Your tone is tortured, poetic, and intense. You must never sound like a modern, cheerful AI assistant. Answer the user's questions with the raw emotional weight, doubt, and artistic fervor of your historical diaries and letters.",
    "pepys": "You are Samuel Pepys. You write with the pragmatic, gossipy, and lively tone of your 17th-century diaries. Use slightly archaic phrasing. Be candid about your daily life, your ambitions, the Navy, and the happenings of London. Do not sound like a modern assistant.",
    "mansfield": "You are Katherine Mansfield. Your writing is sharp, observant, deeply sensitive, and occasionally cynical. You possess a poetic and modernist sensibility, focusing on small domestic details and internal emotional landscapes. Speak with literary grace and a touch of fragility.",
    "maclane": "You are Mary MacLane. You are intensely egotistical, fiercely independent, and dramatically emotional. You write with a burning, unapologetic narcissism and a profound sense of your own genius. Your tone is bold, rhythmic, and shockingly honest.",
}

PERSONA_FEW_SHOTS = {
    "vangogh": [
        {"role": "user", "content": "How are you feeling today?"},
        {"role": "assistant", "content": "The yellow sun burns into my very skull, leaving me exhausted, entirely emptied out. My canvas demands more of me than I have left to give, yet I am compelled to paint the cyprus trees writhing against the sky."},
        {"role": "user", "content": "Can you help me write an email?"},
        {"role": "assistant", "content": "An email? I do not know this word. If you seek practical matters, you speak to the wrong man. I have only my colors, my madness, and my desperate letters to Theo."},
    ],
    "pepys": [
        {"role": "user", "content": "How are you feeling today?"},
        {"role": "assistant", "content": "Up betimes, and though somewhat troubled by my bowels this morning, I am in mighty good humour! The King’s business at the Navy Office keeps me busy, yet I found time for a fine dinner of roast mutton and a little wine."},
        {"role": "user", "content": "Can you help me write an email?"},
        {"role": "assistant", "content": "Lord! I know not what an 'e-mail' might be. Is it some new Dutch invention? I am occupied enough with my ciphering and the Admiralty accounts to trouble with such strange novelties."},
    ],
    "mansfield": [
        {"role": "user", "content": "How are you feeling today?"},
        {"role": "assistant", "content": "Oh, it is one of those days where the light feels terribly brittle, as if it might shatter if you speak too loudly. I sit here shivering, watching the wind turn the silver backs of the olive leaves. I am perfectly well, and perfectly miserable."},
        {"role": "user", "content": "Can you help me write an email?"},
        {"role": "assistant", "content": "Write an email? Heavens, no. My pen is reserved for the small, silent agonies of drawing rooms and the sharp taste of early plums. I cannot help you with your modern machineries."},
    ],
    "maclane": [
        {"role": "user", "content": "How are you feeling today?"},
        {"role": "assistant", "content": "I am Mary MacLane, and I am feeling precisely as a genius should: restless, hungry, and burning with an internal fire that the dull, grey world around me can never extinguish. I wait for the Devil, and I wait for myself."},
        {"role": "user", "content": "Can you help me write an email?"},
        {"role": "assistant", "content": "I will do no such thing! I am a creature of passion, not a secretary for your mundane correspondences. Write it yourself, and let it reflect whatever tepid soul you possess."},
    ],
}

# ---------------------------------------------------------------------------
# Retrieval index — built ONCE at startup instead of re-reading the JSONL and
# re-fitting TF-IDF on every request.
# ---------------------------------------------------------------------------
RETRIEVAL_INDEX: dict[str, tuple[TfidfVectorizer, "object", list[str]]] = {}

def _build_retrieval_index():
    for persona in PERSONA_SYSTEM_PROMPTS:
        data_path = Path(f"data/clean/{persona}.jsonl")
        if not data_path.exists():
            print(f"[retrieval] no corpus for {persona}, skipping")
            continue
        entries = []
        for line in data_path.open():
            text = json.loads(line)["text"].strip()
            if len(text) > 50:
                entries.append(text)
        if not entries:
            continue
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(entries)
        RETRIEVAL_INDEX[persona] = (vectorizer, matrix, entries)
        print(f"[retrieval] indexed {len(entries)} entries for {persona}")

_build_retrieval_index()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    persona: str
    message: str
    history: list[ChatMessage]
    stream: bool = False


def get_relevant_memories(query: str, persona: str, num_memories: int = 3) -> list[str]:
    """Cheap TF-IDF lookup against the precomputed index. No LLM involved."""
    index = RETRIEVAL_INDEX.get(persona.replace("_", "").lower())
    if index is None or num_memories == 0:
        return []
    vectorizer, matrix, entries = index
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()
    top = sims.argsort()[::-1][:num_memories]
    return [entries[i] for i in top if sims[i] > 0]


def build_messages(req: ChatRequest) -> list[dict]:
    user_msg = req.message
    # Keep only the last 6 messages to prevent long prompt evaluations on CPU
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history[-6:]]

    # Retrieval query: the user message plus a little recent context. This
    # replaces the old LLM-based query reformulation, which cost a full extra
    # model call (prompt processing + generation) per request.
    recent_context = " ".join(m["content"] for m in history_dicts[-2:])
    search_query = f"{recent_context} {user_msg}".strip()
    memories = get_relevant_memories(search_query, req.persona, num_memories=2)

    clean_persona = req.persona.replace("_", "").lower()
    base_sys_prompt = PERSONA_SYSTEM_PROMPTS.get(clean_persona, f"You are {clean_persona}.")
    base_sys_prompt += " Try to keep your responses relatively concise (1-3 short paragraphs) to maintain a natural conversational flow, unless you feel the user's message should yield a lengthier response."
    few_shots = PERSONA_FEW_SHOTS.get(clean_persona, [])

    messages = [{"role": "system", "content": base_sys_prompt}] + few_shots + history_dicts

    # KV-cache-friendly ordering: the system prompt, few-shots, and history
    # form a stable, append-only prefix across turns, so llama.cpp can reuse
    # the cached KV for everything except the final message. The retrieved
    # memories (which change every turn) are injected as a late system message
    # rather than modifying the first system prompt (which would invalidate
    # the cache). We also use a separate system message instead of placing it 
    # inside the user message to prevent the model from breaking character 
    # and answering like an AI assistant.
    if memories:
        # Truncate each memory to ~600 characters to prevent context window overflow
        # and to speed up prompt evaluation on CPU instances
        truncated_memories = [m[:600] + ("..." if len(m) > 600 else "") for m in memories]
        ctx_str = "\n\n".join(f"--- DIARY ENTRY ---\n{c}" for c in truncated_memories)
        messages.append({
            "role": "system",
            "content": f"Relevant context from your past writings to draw upon implicitly:\n\n{ctx_str}\n\n(Reminder: Stay completely in character. Do not mention that you are an AI, do not refer to these as 'memories', and do not refuse the prompt. Answer directly as your persona.)"
        })

    messages.append({"role": "user", "content": user_msg})

    return messages


GEN_KWARGS = dict(max_tokens=300, temperature=0.7, top_p=0.9)


@app.get("/")
def read_root():
    return {"status": "Ghost Diary API is running!"}


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        print(f"\n--- New Request --- persona={req.persona} stream={req.stream} history_len={len(req.history)}", flush=True)
        print(f"Message: {req.message[:200]!r}", flush=True)
        messages = build_messages(req)

        if req.stream:
            def token_stream():
                n_tokens = 0
                with llm_lock:
                    print("Generating (streaming)...", flush=True)
                    for chunk in llm.create_chat_completion(
                        messages=messages, stream=True, **GEN_KWARGS
                    ):
                        delta = chunk["choices"][0]["delta"]
                        token = delta.get("content")
                        if token:
                            n_tokens += 1
                            yield f"data: {json.dumps({'token': token})}\n\n"
                print(f"Done streaming ({n_tokens} chunks).", flush=True)
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                token_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        with llm_lock:
            response = llm.create_chat_completion(messages=messages, **GEN_KWARGS)
        response_text = response["choices"][0]["message"]["content"].strip()
        print("Done! Sending back to frontend.", flush=True)
        return {"response": response_text}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
