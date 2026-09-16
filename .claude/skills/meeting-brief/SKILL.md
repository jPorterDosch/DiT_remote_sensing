---
name: meeting-brief
description: Produce drop-in bullet corrections/additions for the user's meeting notes from RESEARCH_NOTES sections newer than their last update — retractions and superseded numbers flagged first.
---

Input: the user's current notes (pasted) and/or a "since" date or section number.

1. Read RESEARCH_NOTES.md sections newer than the given point, PLUS every `[SUPERSEDED…]`
   / `[CORRECTED…]` / `RETRACTED` stamp regardless of date — stale numbers in the user's
   draft are the #1 hazard (a retracted concat gain and a 4.8x ratio both nearly reached
   meetings).
2. Diff against their draft. Output ONLY drop-ins, in their notes' own voice and format:
   - REPLACE blocks for bullets contradicted by current results (quote the exact sentence
     to replace),
   - ADD bullets for new results, each with number + CI + one-line meaning,
   - never a rewrite of untouched bullets, never surrounding commentary inside the
     drop-ins.
3. Numbers: always the CURRENT quotable value with its honest interval; if a number is
   provisional (gate pending, re-run in flight), say so in the bullet rather than omitting.
4. Close with a short "if asked" list: known weak points, scope caveats (protocol
   non-comparabilities, n=500 results, image-level vs trajectory-level pairing), and any
   claim whose evidence grade differs from how it reads.
5. Verify any external citation the user plans to lean on (exact quote, venue) before it
   goes in a bullet.
