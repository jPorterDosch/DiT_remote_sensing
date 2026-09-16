"""Learning-curve extrapolation: probe accuracy vs n_train on cached DiT features,
power-law fit acc(n) = a - b*n^(-c), extrapolated to the full train split.
NESTED subsets (unlike learning_curve.py's independent draws) + fixed held-out eval."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit

CFG = {
    "eurosat": (
        "models/n5000_eurosat_oneshot_ens8/eurosat_flux_23ee82c7+42/multistep_train_feats_oneshot_g1.0.npz",
        1,
        21600,
    ),  # t=180 idx1, full train 27000*0.8
    "resisc45": (
        "models/n5000_resisc45_oneshot_ens8/resisc45_flux_4118f153+42/multistep_train_feats_oneshot_g1.0.npz",
        2,
        25200,
    ),  # t=260 idx2, 31500*0.8
}
SIZES = [500, 1000, 2000, 3000, 4000]
C = 0.1

for ds, (path, tidx, full_n) in CFG.items():
    d = np.load(path)
    X = d["feats"][:, tidx, :]
    y = d["labels"]
    t = int(d["timesteps"][tidx])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    ev = perm[:1000]
    pool = perm[1000:]
    accs = {}
    for n in SIZES:
        a = []
        for s in range(3):
            rs = np.random.default_rng(s)
            tr = pool[rs.permutation(len(pool))[:n]]
            sc = StandardScaler().fit(X[tr])
            A, B = sc.transform(X[tr]), sc.transform(X[ev])
            k = min(512, n - 1, X.shape[1])
            if k < X.shape[1]:
                p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(A)
                A, B = p.transform(A), p.transform(B)
            m = LogisticRegression(C=C, max_iter=2000).fit(A, y[tr])
            a.append(float((m.predict(B) == y[ev]).mean()))
        accs[n] = (np.mean(a), np.std(a))
    ns = np.array(SIZES, float)
    ys = np.array([accs[n][0] for n in SIZES])

    def f(n, a, b, c):
        return a - b * np.power(n, -c)

    try:
        popt, _ = curve_fit(
            f,
            ns,
            ys,
            p0=[ys[-1] + 0.02, 1.0, 0.5],
            maxfev=20000,
            bounds=([ys[-1], 1e-6, 0.05], [1.0, 50, 1.5]),
        )
        a_, b_, c_ = popt
        pred_full = f(full_n, *popt)
        pred_10k = f(10000, *popt)
        asym = a_
    except Exception:
        pred_full = pred_10k = asym = float("nan")
    print(f"\n{ds} (ens8, t={t}, C={C}, eval n=1000 fixed):")
    for n in SIZES:
        print(f"  n={n:<6} acc={accs[n][0]:.4f} ±{accs[n][1]:.4f}")
    print(f"  power-law fit: asymptote a={asym:.4f}, exponent c={c_:.3f}")
    print(f"  predicted @ n=10,000: {pred_10k:.4f}")
    print(f"  predicted @ full split n={full_n}: {pred_full:.4f}")
    print(f"  gain over n=4000: {pred_full - accs[4000][0]:+.4f}")
