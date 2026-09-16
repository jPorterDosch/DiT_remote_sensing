---
name: supersede
description: The correction workflow for when an instrument bug or re-derivation changes a recorded result — fix, re-run, independently verify, stamp the old claim in place, reconcile, and propagate.
---

Execute the full correction cycle for the result named in the arguments. Order matters.

1. FIX the instrument. Never adjust the number by hand.
2. RE-RUN, preserving the old log (rename, don't overwrite) and keeping a `--legacy` path
   or the old log so the superseded number stays regenerable.
3. INDEPENDENTLY VERIFY: re-derive the headline CI from the cached per-image vectors with
   a DIFFERENT bootstrap seed and resample count; require agreement.
4. RECONCILE: state, quantitatively, why the old instrument produced the old number from
   the same data (e.g. "42x vs 6x compression"; "eviction of 357/512 directions"). A
   correction without a reconciliation is not done.
5. STAMP IN PLACE: mark the old claim `[SUPERSEDED <date> by §N …]` at its original
   location in RESEARCH_NOTES — never delete, never rely on readers finding the new
   section. Then write the new section with the table, CIs, gate verdicts, and
   consequences (which downstream claims move, which are untouched — check §12-style
   knock-ons explicitly).
6. PROPAGATE: if this is a NEW defect class, add a numbered rule to CLAUDE.md citing the
   section. Update any skill that encodes the old discipline.
7. NOTIFY: if a standing headline number moved or an interpretation was retracted,
   PushNotification with the direction and size — the user may be citing it right now.

Tone rule: report the correction plainly, including when the error was ours; "the old
number was X because we did Y wrong" beats any softer phrasing.
