---
name: pair-check
description: Verify that two or more feature caches are genuinely image-paired before any cross-cache comparison (subset_indices, labels, timesteps, ensemble size, provenance suffixes).
---

Given two or more `.npz` cache paths (or globs), verify pairing. Run this BEFORE writing
any comparison — labels-only checks pass for DISJOINT subsets (sorted stratified labels
depend only on (subset_size, n_classes); 6o-F).

Checks, in order, each printed PASS/FAIL:
1. `subset_indices` byte-equal across all caches (THE identity; missing key = FAIL, not skip).
2. `labels` equal (secondary; catches dataset-version drift at same indices).
3. `timesteps` equal, when the comparison indexes by position (6o secondary: eta_collapse).
4. `ensemble_size` from each cache's `_meta.json` matches what the comparison assumes
   (filenames don't encode it; swapped ens1/ens8 flips signs silently).
5. Provenance: no `_RANDINIT/_FIXEDCOND/_DEG` suffix mismatch across the set, and no
   control cache satisfying a glob meant for vanilla caches.
6. If the caches come from separate chain runs: note that VAE-posterior draws differ →
   pairing is image-level, not trajectory-level; state which direction that biases
   (dilutes gains, cannot fake them).

Use a short inline python snippet via Bash; print the verdict table; refuse to proceed
with the downstream comparison on any FAIL.
