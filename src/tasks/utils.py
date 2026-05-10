import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import seed_worker


class LinearProbe(nn.Module):
    """Single linear layer trained on top of frozen DiT features."""

    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# TODO: currently unused, allow cfg to specify probe type (and add corresponding Enum to support later expansion)
class MLPProbe(nn.Module):
    """Small MLP trained on top of frozen DiT features."""

    def __init__(self, feat_dim: int, num_classes: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.inference_mode()
def extract_features(cfg, model, dataloader, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract and return (features, labels) for all images in dataloader."""
    # TODO: change call sites to pass in underlying Flux model directly instead of wrapper,
    # which adds extra unnecessary calls to access underlying model.
    model._inner.model.eval()
    all_feats: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    device = torch.device(cfg.device)

    print("saving %s images' features..." % split_name)
    for batch in tqdm(dataloader):
        img = batch["img"].to(device)  # B, 3, H, W
        label = batch["label"]  # B

        for single_img, single_label in zip(img, label, strict=True):
            # TODO: if GPU can tolerate higher batch sizes, we can extract features for the whole batch at once instead of looping through images one by one.
            feat = model.extract(
                single_img,
                timestep=cfg.t,
                block_idx=cfg.k,
                ensemble_size=cfg.model.ensemble_size,
            )  # 1, C, H, W

            feat_vec = feat.mean(dim=[2, 3])  # 1, C  — global average pool
            feat_vec = F.normalize(feat_vec, dim=1)

            all_feats.append(feat_vec.cpu())
            all_labels.append(single_label.cpu())

    feats = torch.cat(all_feats, dim=0).float().numpy()  # N, C
    labels = torch.stack(all_labels, dim=0).numpy()  # N
    return feats, labels


def train_linear_probe(train_feats, train_labels, num_epochs, lr, batch_size, device, num_classes: int):
    X = torch.from_numpy(train_feats).float().to(device)
    y = torch.from_numpy(train_labels).long().to(device)

    ds = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    probe = LinearProbe(X.shape[1], num_classes).to(device)
    # TODO: make optimizer and loss configurable (e.g. SGD, label smoothing)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    total_steps = 0
    t0 = time.perf_counter()
    probe.train()
    for epoch in range(num_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()
            total_steps += 1

    return probe, total_steps, time.perf_counter() - t0


@torch.no_grad()
def evaluate_probe(probe, test_feats, test_labels, device):
    probe.eval()
    X = torch.from_numpy(test_feats).float().to(device)
    preds = probe(X).cpu().numpy().argmax(axis=1)
    top1 = (preds == test_labels).mean() * 100.0
    macro_f1 = f1_score(test_labels, preds, average="macro") * 100.0
    weighted_f1 = f1_score(test_labels, preds, average="weighted") * 100.0
    per_class_f1 = f1_score(test_labels, preds, average=None) * 100.0
    return top1, macro_f1, weighted_f1, per_class_f1
