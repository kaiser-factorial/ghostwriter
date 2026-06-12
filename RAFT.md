# RAFT (Retrieval-Augmented Fine-Tuning) Analysis

I just did a deep dive into the `lumpenspace/raft` repository, and it completely explains the behavior we just saw! It also provides a brilliant blueprint for how we can make our Ghostwriter chatbots significantly more intelligent in the future.

## How True RAFT Works
The RAFT methodology outlined in the repo relies on **two distinct datasets**:
1. **The Target's Past Writings** (Essays, diaries, tweets).
2. **Interview Transcripts** (Actual Q&A conversations featuring the target).

To train the model, the RAFT pipeline does the following:
1. It looks at an interview question (e.g., *"Gary, what do you think of AI scaling?"*).
2. It searches the **Past Writings** database for relevant context.
3. An LLM *rephrases* and evaluates that memory to ensure it's useful context.
4. The model is fine-tuned on the sequence: `[Question] -> [Rephrased Memory Context] -> [The target's actual conversational Answer]`.

## Where We Went Wrong
Our makeshift RAFT script (`build_raft_dataset.py`) missed the most critical component: **Conversational Interview Data**. 

Because Van Gogh and Pepys didn't leave behind conversational interview transcripts, we just used their diaries for *both* the memory context and the target answer. We paired generic questions ("Write about your day") with a diary entry, and used 3 similar diary entries as context.

As a result, we didn't train a conversational agent. We trained a model that thinks: *"When someone talks to me, I need to read the memories provided and write a diary entry that looks exactly like them."* This is why it stubbornly regurgitated the memory when you asked it a question!

## The Solution for Ghostwriter
If we want to build true conversational agents using RAFT in the future, we need to **synthesize interview transcripts**.

We can use a powerful LLM (like GPT-4 or Claude 3.5 Sonnet) to read a diary entry and generate a realistic "Interview Transcript" where an interviewer asks a specific question about the entry, and the persona answers it conversationally. 

### Future RAFT Pipeline for Ghostwriter
1. **Synthesize Interviews:** Use a large LLM to convert the historical diaries into realistic Q&A chat transcripts.
2. **Build the Database:** Chunk and embed the original, raw diaries into a Vector DB.
3. **Generate RAFT Data:** For every synthetic Q&A pair, retrieve the raw diary chunks, and format them into the prompt.
4. **Fine-Tune:** Train the Instruct model to answer the interview questions *using* the raw diary context.

### For Now: Zero-Shot RAG is King
Until we synthesize thousands of Q&A pairs for fine-tuning, the best approach for the website's Chat Bot is **Zero-Shot RAG** (what we are doing now with the updated `chat.py`). By using the base Qwen Instruct model and feeding it the diaries as context, we leverage its pre-trained conversational abilities without "overwriting" them!
