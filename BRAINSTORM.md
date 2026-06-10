# BRAINSTORM

A scratchpad for future features, wild ideas, and frontend mechanics that we want to remember but aren't implementing just yet.

### Frontend & Interaction Ideas
1. **In-Line Ghost Chat:** Add a chat interface embedded directly alongside or beneath each journal entry. Users can chat with the specific persona (or the blended model) about the contents of that exact day's entry, providing the model with the entry as context for the conversation. 
2. **Ephemeral Regeneration:** Allow users to hit a "Regenerate" button on any journal entry and select a different persona token. The frontend would hit the API and generate a new ephemeral entry for that same day/context but from the perspective of a different writer. This would act as a fascinating A/B test to see how Van Gogh reacts to the exact same prompt as Mansfield!
3. **Chameleon UI:** Change the color, theme, or texture of the post block based on the active persona.
4. **Drawers / Expandable Sidebars:** Use collapsible side drawers to hold the chat interactions or the historical timeline exploration, keeping the main reading pane clean until invoked.
5. **Timeline Controls:** Include a toggle for true chronological vs. reverse chronological ordering, and robust filters (by persona, date, or theme).
6. **Dual-Type System:** Pair a beautiful structural sans-serif with an elegant serif for the journal content.
7. **Persona Accordions:** When generating ephemeral entries from multiple personas on the same day, stack them in a disclosure accordion so the user can flip between them seamlessly.
8. **Blended Mode Tooltips (Asides):** When reading the "Blended Ghost", hovering over specific sentences or phrases reveals a tooltip showing which of the 4 historical writers heavily influenced that specific phrase!
9. **Mutually Exclusive Double Sidebars:** Have a left drawer (timeline/filters) and a right drawer (chat/regen), but enforce that only *one* can be open at a time. This guarantees the screen never feels too cluttered and always defaults back to the pristine, centered reading view.
10. **Context-Menu Regeneration:** Use a shadcn-style context menu (triggered by a double-click or right-click on the journal text) to pop up a sleek, floating list of the 4 personas. Clicking one instantly triggers the ephemeral regeneration.
11. **Popover for Personality Vectors:** Use the shadcn Popover component for "Blended Mode" entries. Clicking an icon next to the entry (or the entry's header) triggers a popover showing a sleek visual breakdown of the exact personality vector percents (e.g., 60% Woolf, 30% Van Gogh, 10% Mansfield) that generated the text.
12. **RepEng Control Vectors:** Regenerate entries not just by persona, but by injecting targeted control vectors extracted via Representation Engineering (e.g., Emotion vectors like 'melancholy', 'mania', or 'hope'). Offer a preset list of vectors for the user to inject into the text to dynamically steer the generation in real-time.
13. **RAFT Training Analysis:** In the future, run an experimental training pass using the RAFT (Retrieval Augmented Fine Tuning) approach (via `lumpenspace/raft`) to emulate a specific speaker accurately. Compare the generations of a purely RAFT-trained persona against the current architecture of Token-Conditioned Fine-Tuning + RepEng Control Vectors.
