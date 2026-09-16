---
name: audit-experiment
description: Instrument-audit a named experiment script against this project's known defect classes (selection-on-eval, eviction/compression, unfalsifiable guards, bootstrap unit, provenance) BEFORE its numbers are trusted.
---

Audit the experiment file named in the arguments. Read it fully, then check each defect
class below. For each: state PASS / FAIL / N-A with the line number and a one-line
justification. Do not fix anything in this pass unless asked — the deliverable is findings.

## Checklist (each item is a materialized failure from this project)
1. SELECTION: every `max()`/`argmax`/"best" — is it computed on data disjoint from what
   produces the reported number? (concat best-t 6m; protocol_shape endpoints 6o-B;
   max-of-probes z 6o-E.)
2. TRANSFORM FITTING: is any scaler/PCA/whitening fit on a matrix that includes the
   candidate being evaluated, or does it compress one arm harder than another?
   (eviction 6o-A; compression-null §13.) Nulls need this audit as much as positives.
3. NULL VALIDITY: does every null/control match the candidate in width AND distribution?
   Is there a positive control that would read impossibly if the instrument broke — and
   would anyone notice? (anchor at −0.015 footnoted for two weeks, 6p.)
4. BOTH BARS: does any "adds beyond" claim rest on a null-corrected delta alone?
   (redundant blocks beat their nulls, §11 / rule 3b.)
5. GUARDS: for each validity check, can it actually fail? Fire-test it mentally with the
   mismatch it exists to catch: labels-only pairing passes disjoint subsets (6o-F);
   fitting indices pass wrong-dataset caches (6s-1).
6. BOOTSTRAP UNIT: are CIs/z over images, or over (seed,fold)/pairs/rows derived from the
   same images? (A1; 6n-A: 42 pairs/image ⇒ n=images.)
7. OPERATING POINT: same C / convergence regime / dims cap across compared arms? Is
   non-convergence measured per arm? (6m-3, 6p.)
8. REGULARIZATION SCALE: any absolute ridge/epsilon against a spectrum it cannot floor?
   (1e-4 vs eigenvalues 0.13–5e4, 6o-D — use scale-relative shrinkage.)
9. PROVENANCE & IDENTITY: outputs stamped via `env_provenance(cfg, honors=…)`; cache loads
   verify subset_indices + meta ensemble size + timesteps; CSV appends validate headers;
   no config default doubling as a pin (6o-1).
10. RECOVERABILITY: are per-image vectors / pre-norm features persisted so re-analysis is
    offline? (rule 12.)

## Output
A ranked findings list (worst first) in the reply, each with file:line, defect class, and
the concrete failure scenario. If everything passes, say so explicitly per item — silence
is not a verdict. Record real findings in RESEARCH_NOTES per the /supersede flow if any
touch reported numbers.
