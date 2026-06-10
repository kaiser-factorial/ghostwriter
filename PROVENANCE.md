# PROVENANCE

Edition-level sourcing and copyright reasoning for every text in `data/raw/`.
All determinations are for **US public domain** status; the operative rule is
that works published before January 1, 1930 are PD in the US as of 2026.

## In the corpus

| file | work | edition used | PD basis |
|---|---|---|---|
| `pepys_complete.txt` | Samuel Pepys, *Diary* (written 1660–69) | Wheatley transcription, via Project Gutenberg #4200 | Published 1893–99; author d. 1703 |
| `vangogh_letters.txt` | Vincent van Gogh, letters | *The Letters of a Post-Impressionist*, trans. **Anthony Ludovici**, 1912 — PG #40393 | Translation published 1912. NB: the letters themselves are PD, but most English translations are NOT (the standard 1958 edition is under copyright). Only pre-1930 translations qualify. |
| `mansfield_journal_1927.txt` | Katherine Mansfield, *Journal* | Knopf, New York, 1927 (file is the 1928 second printing of the 1927 edition), Archive.org OCR | Published Sept 1927. **Not** the 1954 "definitive edition," which restored suppressed passages and carries separate copyright — do not substitute it. Murry's editorial bracket insertions are stripped in cleaning, partly as conservatism about editorial-apparatus copyright. |
| `maclane_story.txt` | Mary MacLane, *The Story of Mary MacLane* | 1902, PG #43696 | Published 1902 |
| `maclane_i.txt` | Mary MacLane, *I, Mary MacLane* | 1917, PG #43556 | Published 1917 |

Integrity: Gutenberg files were downloaded twice independently (gutenberg.org
ebook endpoint / aleph.pglaf.org mirror) during the build; checksums matched.

## Considered and declined — and why

These were discussed for the fourth ghost and rejected on rights grounds.
Recorded so the reasoning survives:

- **Anne Frank** — diary written 1942–44 but copyright runs from
  *publication* (1947+); Dutch original under copyright into the 2030s,
  English translations longer. A personally owned PDF copy confers no
  training rights (license to read ≠ license to reproduce). Declined even
  though a copy was available.
- **Anaïs Nin** — diaries published 1966+; d. 1977; estate active. Locked
  for decades.
- **Sylvia Plath** — journals published 1982. Locked.
- **Alice James** — perfect sensibility for this project; diary published
  1934. Misses the PD line by four years; revisit in 2030.
- **Anne Lister** — raw diaries (d. 1840) are PD, but the famous decoded
  texts are 1980s editions. Usable in principle via 1887–92 Halifax
  Guardian excerpts + open archive transcriptions; deferred for sourcing
  effort, not rights.

## House rules for adding ghosts

1. Verify the *edition*, not just the author. Publication date < 1930.
2. Translations: the translation's own publication date governs.
3. Strip modern editorial apparatus during cleaning.
4. Record the source URL, edition, and reasoning in this file.
5. When in doubt, the ghost stays buried.
