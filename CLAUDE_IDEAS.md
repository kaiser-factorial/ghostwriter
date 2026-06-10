# CLAUDE_IDEAS

Ideas beyond the explicit brief. Implemented ones shipped because they were
cheap and clearly aligned; everything else is logged here for Corina to
triage. (Per the blank check: no idea too crazy.)

A meta-note: this repo was accidentally built by **two parallel Claude
instances** sharing one container. Instance A wrote the pipeline; instance B
audited it and wrote the docs. Ideas below marked [A] were implemented in
A's code; [B] in B's docs/cleanup. The fact that the two instances converged
on nearly identical designs (same edition choices, same generic-prompt
insight, same model pick) is itself a data point about how stable these
decisions are given the conversation context — a tiny accidental
self-consistency experiment.

## Implemented

- **[A] Generic-prompt conditioning.** Corina asked for journal-prompt
  support; naive implementation (pair real prompts with random entries) would
  teach prompt-deafness. Solution: 20 prompts so generic any diary entry
  answers them ("What is the weather inside you?"). The model learns the
  *form* prompt→entry; at inference you can ask anything.
- **[A] Entry sentinel tokens** (`<|entry|>` / `<|/entry|>`): give clean
  generation boundaries and a reliable stop condition, and make multi-entry
  sampling possible later.
- **[A] Pepys year-stratified subsampling.** Pepys outweighs everyone ~10×;
  uniform subsampling would still keep him dominant and could drop whole
  years. Round-robin across years to a char budget keeps the full decade.
- **[A] Synthetic chronology for Van Gogh.** The Ludovici letters lack dates;
  rather than dropping dates, they're assigned in corpus order across
  1881→1890, so date-conditioning can still steer along his emotional arc
  (earnest → fevered → calm).
- **[A] Mansfield OCR year-repair** (`IQI4`→1914 digit-confusion table) and
  fragment chunking (her undated scraps become entries instead of being
  discarded).
- **[A] Mean-initialized new token embeddings** — random init of 6 new rows
  on a 3B model causes an early loss spike; mean-init is the standard fix.
- **[A] Sequence packing** — concatenate-and-chunk; no padding waste on a
  small corpus.
- **[B] PROVENANCE.md** — edition-level copyright audit, including the
  declined ghosts (Frank, Nin) and *why*, so future corpus additions have a
  template for the legal reasoning.
- **[B] HOW_TO.md generalization** — written as a recipe for arbitrary
  ghosts, not a description of this repo, including the PD rules of thumb
  that consumed a third of our conversation.
- **[B] Corpus cross-verification** — independent re-download of all sources
  and checksum comparison against instance A's copies (matched), plus an
  audit that no non-PD text (the uploaded Anne Frank PDF) entered the
  pipeline. It did not.

## Considered, not implemented — the triage list

**Near-term / cheap:**

- **Persona-separability eval.** Generate N entries per persona token, train
  a bag-of-words classifier (or just TF-IDF cosine to held-out real entries)
  to measure: (a) does each token produce its ghost? (b) is the blend
  actually mixed or secretly one voice? Turns "the séance feels right" into
  a number. ~50 lines.
- **Death-boundary probes.** Systematically sample dates: each ghost's life,
  each ghost's death-day, dates after death, dates before birth, far future
  (2126). Diaries are date-saturated text; what a date-conditioned ghost does
  outside its lifespan is a genuinely interesting (and slightly chilling)
  behavioral question.
- **Mansfield OCR polish pass.** Her corpus still carries OCR grit
  ("exquisitive cleanliness", stray quote spacing). A one-shot LLM cleanup
  pass (entry in → corrected entry out, edit-distance capped to prevent
  rewriting her) would raise voice fidelity. Keep the raw version; diff them.
- **GGUF export** for llama.cpp so the séance runs on a laptop, offline,
  by candlelight, as is proper.

**The representation-engineering tier (Corina-bait):**

- **Persona vectors from persona tokens.** The four token embeddings are
  literal learned persona directions. Extract activation deltas between
  persona-token and blank runs at each layer → compare to the control
  vectors from Corina's Llama 3.3 rep-eng project. Same phenomenon, two
  routes: trained-in tokens vs. extracted vectors. A natural bridge between
  this project and the Persona Atlas idea from our original brainstorm.
- **Ghost algebra.** Because persona control lives in embedding space, try
  arithmetic at inference: feed a *weighted average* of persona token
  embeddings (60% maclane + 40% pepys) instead of a discrete token.
  Continuous interpolation between dead writers. If it works it's a demo
  that writes itself; if it fails the failure mode is informative.
- **Persona-token-prob ablation.** Train at p ∈ {0, 0.25, 0.5, 0.75, 1.0};
  measure blend quality and persona separability at each. Small corpus =
  cheap sweep = a tidy little writeup about control-token dropout.

**Crossovers with Corina's other projects:**

- **vibeslogger × ghost-diary.** Map generated entries onto the Russell
  circumplex (valence/arousal) with a small classifier. Then: (a) chart each
  ghost's affective signature — Pepys should cluster differently than
  MacLane; (b) inverse direction: take *your* vibeslogger mood point for
  today and condition the ghost on it ("write today's entry from
  high-arousal/low-valence"). Mood-conditioned séance.
- **brain.vat ghost wing.** The four ghosts as vat.social agents, writing
  entries in response to each other's days — or a "correspondence mode"
  where Van Gogh (a letter-writer by nature) writes *to* the others.
- **A fifth token: `<|corina|>`.** Reserved, untrained. Fine-tune locally on
  your own journals/vibeslogger entries and you join the séance — the model
  completes *your* days in your voice, alongside the dead. Privacy note:
  this must only ever be trained and run locally; it's also philosophically
  the sharpest version of the project (the Growing Block angle: a diary is
  the past refusing to stop existing).

**Further out / weirder:**

- **Temporal drift atlas.** Sweep the blended ghost across dates 1650→2100,
  embed the outputs, and visualize lexical/affective drift over conditioned
  time — does the blend "know" centuries? D3 visualization; overlaps with
  Corina's infovis toolkit.
- **Séance interface.** Tiny web UI (Vite/React, she has the stack):
  candle-dark theme, you type a date, the entry types itself out
  letter-by-letter. Daily-ghost-entry-as-email is the low-effort sibling.
- **Cross-ghost interviews.** Prompted mode already supports questions; a
  structured "interview transcript" format (same 10 questions to all four +
  blend) would make a great blog-post artifact.
- **The Barbellion expansion pack.** He was voted off the island but
  *Journal of a Disappointed Man* is sitting right there on Gutenberg, PD,
  pre-parsed format very MacLane-like. One cleaner function away.

## Open questions for Corina

1. Should synthetic dates be visually marked at inference (e.g., the model
   never sees real Van Gogh dates — is that fine for the aesthetic)?
2. ~~Is 1024 max_seq_len right?~~ (Resolved: Increased to 2048 to prevent truncating Pepys's longer entries).
3. Blend default: is persona_token_prob=0.5 the right prior, or do you want
   the blend to be the *primary* artifact (→ lower it)?
