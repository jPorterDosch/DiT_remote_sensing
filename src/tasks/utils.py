import numpy as np


def _subsample_by_fraction(
    feats: np.ndarray, labels: np.ndarray, fraction: float, seed: int, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    # class-balanced subsample: take `fraction` percent of each class independently
    rng = np.random.default_rng(seed)
    keep_idx: list[int] = []
    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        n_keep = max(1, int(len(cls_idx) * fraction / 100.0))
        chosen = rng.choice(cls_idx, size=n_keep, replace=False)
        keep_idx.extend(chosen.tolist())
    keep_idx_np = np.array(keep_idx)
    return feats[keep_idx_np], labels[keep_idx]
