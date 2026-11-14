"""Per-class probe gain of each student arm vs the per-class advantage of its teacher over the
frozen student (free analysis from the matrix cells' per-image predictions). Distillation predicts
the student gains most on the classes its teacher knows best (positive Spearman, per dataset);
reorganization predicts no such dependence. Uses all fractions / seeds, test split, per-class
accuracy = mean(correct | label)."""

import glob
import json
import numpy as np
from scipy.stats import spearmanr

C = "results/matrix/cells"  # run from the worktree root: python experiments/figs/perclass_gain.py
DS = ["m_eurosat", "m_brick_kiln", "m_pv4ger", "m_so2sat", "m_forestnet", "m_bigearthnet"]
FR = ["0.01", "0.05", "0.1", "0.25"]


def per_class(model, ds, fr):
    accs = []
    for f in sorted(glob.glob(f"{C}/{model}/{ds}/f{fr}_s*.npz")):
        d = np.load(f, allow_pickle=True)
        lab = d["labels"]
        if (
            lab.ndim == 2
        ):  # multilabel (bigearthnet): per-class F1 from pred/labels, nan where a class has no support
            pr = d["pred"].astype(bool)
            lb = lab.astype(bool)
            tp, fp, fn = (pr & lb).sum(0), (pr & ~lb).sum(0), (~pr & lb).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                accs.append(np.where(tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan))
        else:
            cor = d["correct"].astype(float)
            accs.append(np.array([cor[lab == c].mean() for c in range(lab.max() + 1)]))
    if (
        accs and len(accs) != 3
    ):  # the matrix protocol has seeds 0/1/2 below 100 %; a partial arm is refused, not averaged (review 2026-10-27)
        raise SystemExit(f"{model} {ds} f={fr}: {len(accs)} seed cells, protocol has 3 -- arm incomplete")
    return np.mean(accs, 0) if accs else None


out = {}
for student, teacher in (
    ("flux-proj", "dinov3-sat"),
    ("flux-naive", "dinov3-sat"),
    ("flux-proj-web", "dinov3-web"),
    ("flux-naive-web", "dinov3-web"),
    ("flux-proj", "dinov3-web"),
    ("flux-proj-web", "dinov3-sat"),
):
    rows = []
    for ds in DS:
        for fr in FR:
            s, t, f0 = (
                per_class(student, ds, fr),
                per_class(teacher, ds, fr),
                per_class("flux-pretrained-k28", ds, fr),
            )
            if s is None or t is None or f0 is None:
                continue
            gain, adv = s - f0, t - f0
            ok = np.isfinite(gain) & np.isfinite(adv)
            gain, adv = gain[ok], adv[ok]
            if len(gain) < 3:
                continue
            r = spearmanr(gain, adv).correlation
            rows.append((ds, fr, len(gain), float(r), float(np.corrcoef(gain, adv)[0, 1])))
    if not rows:
        continue
    rs = [r[3] for r in rows]
    print(
        f"== {student} vs teacher {teacher}: mean Spearman(per-class gain, teacher advantage) {np.mean(rs):+.3f} over {len(rows)} (dataset, fraction) cells; by dataset:"
    )
    for ds in DS:
        sub = [r for r in rows if r[0] == ds]
        if sub:
            print(
                f"   {ds:16s} classes {sub[0][2]:2d}  rho {np.mean([r[3] for r in sub]):+.3f}  pearson {np.mean([r[4] for r in sub]):+.3f}"
            )
    out[f"{student}|{teacher}"] = rows
json.dump(out, open("results/knn/perclass_gain_vs_teacher_advantage.json", "w"), indent=1)
print("PERCLASS_OK")

# --- differential test: cancels the shared frozen baseline. For classes c, rho( s_sat(c)-s_web(c), t_sat(c)-t_web(c) ).
# Both sides subtract nothing in common with the frozen probe, so a regression-to-the-mean confound (weak frozen class
# => every model gains there) drops out. Distillation of teacher-specific knowledge predicts rho > 0; shared-only transfer predicts ~0.
# Null: per-(dataset,fraction) permutation of the class order of the teacher difference, 2000 draws, two-sided p on the mean rho.
rng = np.random.default_rng(0)
print("\n== differential: rho(student_sat - student_web, teacher_sat - teacher_web) per class")
for a, b in (("flux-proj", "flux-proj-web"), ("flux-naive", "flux-naive-web")):
    rows = []
    null = []
    for ds in DS:
        for fr in FR:
            sa, sb, ta, tb, f0 = (
                per_class(a, ds, fr),
                per_class(b, ds, fr),
                per_class("dinov3-sat", ds, fr),
                per_class("dinov3-web", ds, fr),
                per_class("flux-pretrained-k28", ds, fr),
            )
            if any(x is None for x in (sa, sb, ta, tb, f0)):
                continue
            ds_, dt = sa - sb, ta - tb
            ok = np.isfinite(ds_) & np.isfinite(dt)
            ds_, dt = ds_[ok], dt[ok]
            if len(ds_) < 3:
                continue
            rows.append(
                (
                    ds,
                    fr,
                    len(ds_),
                    float(spearmanr(ds_, dt).correlation),
                    float(spearmanr(ds_, (f0[ok])).correlation),
                )
            )
            null.append([spearmanr(ds_, rng.permutation(dt)).correlation for _ in range(2000)])
    if not rows:
        continue
    rs = np.array([r[3] for r in rows])
    nm = np.mean(np.array(null), 0)
    p = float((np.abs(nm) >= abs(rs.mean())).mean())
    print(
        f"   {a} - {b}: mean rho {rs.mean():+.3f} over {len(rows)} cells (perm p = {p:.3f}; null sd {nm.std():.3f}); rho(student diff, frozen acc) {np.mean([r[4] for r in rows]):+.3f}"
    )
    for ds in DS:
        sub = [r for r in rows if r[0] == ds]
        if sub:
            print(
                f"      {ds:16s} classes {sub[0][2]:2d}  rho {np.mean([r[3] for r in sub]):+.3f}  per fraction {[round(r[3], 2) for r in sub]}"
            )
    out[f"diff:{a}-{b}"] = rows


# --- regression-to-the-mean check: partial Spearman of gain vs advantage controlling for the frozen per-class accuracy
def partial(x, y, z):
    from scipy.stats import rankdata

    x, y, z = rankdata(x), rankdata(y), rankdata(z)
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(rx, ry)[0, 1])


print("\n== partial Spearman(gain, teacher advantage | frozen per-class acc)")
for student, teacher in (
    ("flux-proj", "dinov3-sat"),
    ("flux-proj-web", "dinov3-web"),
    ("flux-proj", "dinov3-web"),
    ("flux-proj-web", "dinov3-sat"),
):
    ps = []
    for ds in DS:
        for fr in FR:
            s, t, f0 = (
                per_class(student, ds, fr),
                per_class(teacher, ds, fr),
                per_class("flux-pretrained-k28", ds, fr),
            )
            if s is None or t is None or f0 is None:
                continue
            ok = np.isfinite(s) & np.isfinite(t) & np.isfinite(f0)
            if ok.sum() < 4:
                continue
            ps.append(partial((s - f0)[ok], (t - f0)[ok], f0[ok]))
    print(f"   {student} vs {teacher}: {np.mean(ps):+.3f} over {len(ps)} cells")
json.dump(out, open("results/knn/perclass_gain_vs_teacher_advantage.json", "w"), indent=1)
print("PERCLASS2_OK")
