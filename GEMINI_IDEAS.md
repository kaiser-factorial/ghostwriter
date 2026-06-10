# GEMINI_IDEAS

Ideas, tweaks, and experiments generated during Gemini's time working on the `ghost-diary` project. Modeled after the excellent `CLAUDE_IDEAS.md`, this document tracks both what was built today and the "crazy ideas" backlog for future exploration. 

## Implemented

- **LLM OCR Polish Pass:** Wrote a script (`scripts/polish_mansfield.py`) that uses Gemini 2.5 Flash to scrub the OCR grit from Mansfield's corpus. To ensure we didn't accidentally rewrite her distinct modernist voice or erase her erratic punctuation, the script enforces a strict Levenshtein edit-distance check, rejecting any LLM outputs that drift too far from the original text. Added a rate-limiter to gracefully handle the Free Tier quota!
- **Temporal Chunking:** Rewrote the core loop in `build_dataset.py` to support chronological "multi-day chunks" (`--chunk-prob` and `--chunk-max-size`). The script now probabilistically glues 2 to 4 consecutive days together into single documents before packing. This allows the model to naturally learn day-to-day thematic carry-over during the primary fine-tune, eliminating the need for a secondary LoRA pass.
- **Sequence Length Extension:** Bumped the `max_seq_len` from 1024 to 2048 across the configs and scripts to leverage Qwen 2.5's massive context window, ensuring that even Pepys's longest daily brain dumps aren't truncated during dataset packing.

## Considered, not implemented — the triage list

**Near-term / pragmatic:**
- **Dynamic Temperature Scaling for the Blend:** Currently, `sample.py` uses a static temperature. What if the temperature was dynamic depending on the presence of a persona token? The "blended ghost" might benefit from a slightly higher temperature (to encourage the synthesis of weird, emergent thoughts from the four writers), while the strict personas might need a lower temperature to stay tightly bound to their historical voices.
- **Diffing the Polish:** Once the Mansfield polish script finishes, we should run a script that highlights *only* the diffs across the entire 380KB corpus so you can easily review what the LLM changed without reading the whole book.

**The "Crazy Ideas" / Speculative Tier:**

- **The Time-Synchronized Séance:** Right now, the synthetic dates for Van Gogh and MacLane are just arbitrarily assigned to preserve order. What if we algorithmically mapped their emotional arcs to the *current calendar year*? For example, Van Gogh's frantic "Arles fever" letters could be mathematically mapped to the peak of Summer. If you use the model in July, you get a manic ghost. If you use it in December, you get a melancholic ghost.
- **The Modern Intrusion (Anachronism Training):** What happens if a 17th-century ghost sees a smartphone? We could use an LLM to generate synthetic journal prompts involving modern 21st-century concepts, and have the LLM write a few "fake" historical reactions to them in the voices of our writers. By injecting a tiny fraction (~1%) of these anachronistic entries into the training data, we could teach the ghosts how to philosophize about the modern world without breaking their archaic vernacular. 
- **Cross-Era Dialogue (The Haunted Chatroom):** Instead of just giving the ghosts generic prompts, we could use them to talk to *each other*. We could write an inference wrapper that takes today's entry from MacLane, and uses it as the `[Prompt: ...]` for Mansfield's entry tomorrow. You'd get an autonomous, endlessly evolving dialogue between four dead writers happening in real-time on your blog.
- **Steerable Mood Vectors:** During dataset generation, we could have an LLM score every single entry on a scale of Valence (Happy/Sad) and Arousal (Calm/Frantic). We could append these tiny scores to the start of the entries in the training data (e.g. `[Mood: V-2, A+4]`). At inference, you wouldn't just ask for Van Gogh—you could dial in the exact emotional state of the ghost you want to summon. 
