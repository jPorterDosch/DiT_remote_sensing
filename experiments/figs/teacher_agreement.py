"""Shared vs teacher-specific k-NN agreement gain (2026-10-26, second code review).
For every adapted student with a k-NN diagnostic row (results/knn/web/per_dataset/*/knn_diag.json,
eval.knn_diag with --spec-models over both teachers), compare the gain over frozen FLUX in k-NN-set
agreement with its OWN teacher and with the OTHER teacher, at k = 10 and k = 20 (both fixed in advance).
Descriptive only: 'specific' = gain_own - gain_other. The earlier 'mixture ceiling' test was dropped
(Jaccard k-NN agreement is not monotone under feature interpolation, so max(frozen, teacher-teacher)
is not a bound). Writes results/knn/web/teacher_agreement_summary.md and .json.
    python experiments/figs/teacher_agreement.py
"""

import glob
import json
import os

import numpy as np

ROOT = "results/knn/web"
FROZEN = "flux-pretrained-k28"
STUDENTS = (
    ("flux-proj", "sat", "web"),
    ("flux-naive", "sat", "web"),
    ("flux-proj-web", "web", "sat"),
    ("flux-naive-web", "web", "sat"),
)


def main() -> int:
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/per_dataset/*/knn_diag.json")):
        rows += json.load(open(f))["rows"]
    if not rows:
        raise SystemExit(f"no rows under {ROOT}/per_dataset")

    def g(model, ds, key):
        r = [
            r
            for r in rows
            if r["model"] == model and r["dataset"] == ds and r["candidate"] in ("t100", "cls")
        ]
        if len(r) != 1:
            raise SystemExit(f"{model} / {ds}: expected one diagnostic row (t100 or cls), got {len(r)}")
        return r[0][key]

    out, md = [], []
    md.append(
        "# k-NN agreement with BOTH teachers (t100 features, DINO cls): shared vs teacher-specific gain\n"
    )
    md.append(
        "Source: results/knn/web/per_dataset/*/knn_diag.json (eval/knn_diag.py, job array 6477998); this table: experiments/figs/teacher_agreement.py."
    )
    md.append(
        "gain = agreement(student) - agreement(frozen FLUX k28); own = the student's own teacher, other = the other DINOv3 teacher;"
    )
    md.append(
        "specific = gain_own - gain_other; tt = agreement between the two teachers (reference scale). Descriptive, no bound is claimed.\n"
    )
    for k in (10, 20):
        md.append(f"## k = {k}\n")
        md.append(
            "| dataset | student | teacher | frozen own | frozen other | gain own | gain other | specific | tt |"
        )
        md.append("|---|---|---|---|---|---|---|---|---|")
        spec, shared = [], []
        for ds in sorted({r["dataset"] for r in rows}):
            tt = g("dinov3-sat", ds, f"agree_dinov3-web@{k}")
            for m, own, oth in STUDENTS:
                if not any(r["model"] == m and r["dataset"] == ds for r in rows):
                    continue
                a_own, a_oth = g(m, ds, f"agree_dinov3-{own}@{k}"), g(m, ds, f"agree_dinov3-{oth}@{k}")
                f_own, f_oth = (
                    g(FROZEN, ds, f"agree_dinov3-{own}@{k}"),
                    g(FROZEN, ds, f"agree_dinov3-{oth}@{k}"),
                )
                go, gt = a_own - f_own, a_oth - f_oth
                spec.append(go - gt)
                shared.append(min(go, gt))
                out.append(
                    {
                        "k": k,
                        "dataset": ds,
                        "student": m,
                        "teacher": own,
                        "frozen_own": f_own,
                        "frozen_other": f_oth,
                        "gain_own": go,
                        "gain_other": gt,
                        "specific": go - gt,
                        "teacher_teacher": tt,
                    }
                )
                md.append(
                    f"| {ds} | {m} | dinov3-{own} | {f_own:.3f} | {f_oth:.3f} | {go:+.3f} | {gt:+.3f} | {go - gt:+.3f} | {tt:.3f} |"
                )
        md.append(
            f"\nk = {k}: mean shared gain (min of the two) {np.mean(shared):+.3f}, mean teacher-specific residual {np.mean(spec):+.3f} "
            f"(range {min(spec):+.3f} .. {max(spec):+.3f}, n = {len(spec)}); residual > 0 in {sum(s > 0 for s in spec)}/{len(spec)}.\n"
        )
    md.append(
        "Reading: most of the neighbourhood-agreement gain is shared by both teachers; the teacher-specific residual is small relative to it."
    )
    md.append(
        "Caveat: part of the shared gain is expected from the teachers' own overlap (tt); a synthetic-mixture null is not implemented, so this is descriptive."
    )
    os.makedirs(ROOT, exist_ok=True)
    open(f"{ROOT}/teacher_agreement_summary.md", "w").write("\n".join(md) + "\n")
    json.dump(out, open(f"{ROOT}/teacher_agreement_summary.json", "w"), indent=1)
    print("\n".join(md))
    print("TEACHER_AGREEMENT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
