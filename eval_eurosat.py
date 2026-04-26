import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from src.flux.feat_flux import Featurizer4Eval
import os
import json
import time
from einops import rearrange
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import warnings

warnings.filterwarnings('ignore')

from eurosat_dataloader import (
    EuroSATDataset,
    get_eurosat_categories,
    EUROSAT_CLASSES,
)

#### layernorm of our adaln for dit feature, 3072 is feature dimension of flux.
pre_norm = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6)

NUM_CLASSES = 10  # EuroSAT has 10 land-use classes


class LinearProbe(nn.Module):
    """Single linear layer trained on top of frozen DiT features."""
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def extract_features(args, model, dataloader, split_name):
    """Extract and return (features, labels) for all images in dataloader."""
    all_feats = []
    all_labels = []

    print("saving %s images' features..." % split_name)
    for batch in tqdm(dataloader):
        img = batch["img"].cuda()    # 1, 3, H, W
        label = batch["label"]       # 1

        feat_raw, ada = model.forward(args,
                                      img.squeeze(0),
                                      timestep=args.t,
                                      block_idx=args.k,
                                      ensemble_size=args.ensemble_size)

        B,C,H,W = feat_raw.shape

        # Channel discard
        # We suppress Massive Activations (MAs) in DiT features by discarding their channels,
        # preventing LayerNorm from propagating their adverse influence to the remaining dimensions.
        # For a given DiT, the MA dimensions are fixed and easy to identify; we simply zero those channels.
        if args.cd:
            feat_raw[:,154,:,:]=0.0
            feat_raw[:,1446,:,:]=0.0

        feat = rearrange(feat_raw, "b c h w -> b (h w) c")
        feat = pre_norm(feat)
        feat = rearrange(feat, "b (h w) c -> b c h w", h=H, w=W)

        ada_shift = ada[0][0]
        ada_scale = ada[0][1]

        ada_shift = ada_shift.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        ada_scale = ada_scale.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        feat = (1 + ada_scale) * feat + ada_shift

        # global average pool over spatial dims -> classification vector
        feat_vec = feat.mean(dim=[2, 3])           # B, C
        feat_vec = F.normalize(feat_vec, dim=1)

        all_feats.append(feat_vec.cpu())
        all_labels.append(label)

    feats  = torch.cat(all_feats,  dim=0).numpy()  # N, C
    labels = torch.cat(all_labels, dim=0).numpy()  # N
    return feats, labels


def subsample_by_fraction(feats, labels, fraction, seed):
    # class-balanced subsample: take `fraction` percent of each class independently
    rng = np.random.default_rng(seed)
    keep_idx = []
    for cls in range(NUM_CLASSES):
        cls_idx = np.where(labels == cls)[0]
        n_keep = max(1, int(len(cls_idx) * fraction / 100.0))
        chosen = rng.choice(cls_idx, size=n_keep, replace=False)
        keep_idx.extend(chosen.tolist())
    keep_idx = np.array(keep_idx)
    return feats[keep_idx], labels[keep_idx]


def train_linear_probe(train_feats, train_labels, num_epochs, lr, batch_size, device):
    X = torch.from_numpy(train_feats).float().to(device)
    y = torch.from_numpy(train_labels).long().to(device)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    probe     = LinearProbe(X.shape[1], NUM_CLASSES).to(device)
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

    wall_clock = time.perf_counter() - t0
    return probe, total_steps, wall_clock


@torch.no_grad()
def evaluate_probe(probe, test_feats, test_labels, device):
    probe.eval()
    X = torch.from_numpy(test_feats).float().to(device)
    logits = probe(X).cpu().numpy()
    preds = logits.argmax(axis=1)

    top1_acc = (preds == test_labels).mean() * 100.0
    macro_f1 = f1_score(test_labels, preds, average="macro") * 100.0
    return top1_acc, macro_f1


def main(args):
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            print('%s: %s' % (str(arg), str(value)))

    torch.cuda.set_device(0)

    cat_list = get_eurosat_categories()

    if args.dit_model == 'flux':
        dit_model = Featurizer4Eval(cat_list=cat_list, ensemble_size=args.ensemble_size)
    else:
        raise Exception("model must be in [flux] ")

    # build train/test datasets with a fixed 80/20 split per class
    train_dataset = EuroSATDataset(args.dataset_path, split="train", img_size=args.img_size)
    test_dataset  = EuroSATDataset(args.dataset_path, split="test",  img_size=args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())

    os.makedirs(args.save_path, exist_ok=True)
    train_feat_path = os.path.join(args.save_path, "train_feats.npz")
    test_feat_path  = os.path.join(args.save_path, "test_feats.npz")

    # load cached features if available, otherwise extract and save
    if os.path.exists(train_feat_path) and not args.overwrite_features:
        print("loading cached train features from %s" % train_feat_path)
        d = np.load(train_feat_path)
        train_feats, train_labels = d["feats"], d["labels"]
    else:
        train_feats, train_labels = extract_features(args, dit_model, train_loader, "train")
        np.savez(train_feat_path, feats=train_feats, labels=train_labels)

    if os.path.exists(test_feat_path) and not args.overwrite_features:
        print("loading cached test features from %s" % test_feat_path)
        d = np.load(test_feat_path)
        test_feats, test_labels = d["feats"], d["labels"]
    else:
        test_feats, test_labels = extract_features(args, dit_model, test_loader, "test")
        np.savez(test_feat_path, feats=test_feats, labels=test_labels)

    result = {}

    print("Label fractions: %s" % args.label_fractions)
    for frac in args.label_fractions:
        sub_feats, sub_labels = subsample_by_fraction(
            train_feats, train_labels, fraction=frac, seed=args.seed
        )

        probe, steps, elapsed = train_linear_probe(
            sub_feats, sub_labels,
            num_epochs=args.clf_epochs,
            lr=args.clf_lr,
            batch_size=args.clf_batch_size,
            device=torch.device("cuda"),
        )

        top1, f1 = evaluate_probe(probe, test_feats, test_labels, torch.device("cuda"))

        # per-class accuracy breakdown
        probe.eval()
        with torch.no_grad():
            X = torch.from_numpy(test_feats).float().cuda()
            preds = probe(X).cpu().numpy().argmax(axis=1)
        per_class = {}
        for cls_idx, cls_name in enumerate(EUROSAT_CLASSES):
            mask = test_labels == cls_idx
            cls_acc = (preds[mask] == test_labels[mask]).mean() * 100.0
            per_class[cls_name] = round(float(cls_acc), 2)

        result[frac] = {
            "label_fraction_pct" : frac,
            "n_train_samples"    : int(len(sub_labels)),
            "top1_accuracy"      : round(top1, 2),
            "macro_f1"           : round(f1, 2),
            "training_steps"     : steps,
            "wall_clock_seconds" : round(elapsed, 2),
            "per_class_accuracy" : per_class,
        }

        print('%s%% labels  top1: %.2f  macro-f1: %.2f  n=%d  steps=%d  time=%.1fs' % (
            frac, top1, f1, len(sub_labels), steps, elapsed))

        torch.cuda.empty_cache()

    # 判断目录是否存在
    save_dir = 'results_eurosat/%s' % args.dit_model
    if not os.path.exists(save_dir):
        # 如果目录不存在，则创建它
        os.makedirs(save_dir)
    with open('results_eurosat/%s/t%s_b%s_e%s_seed%s.json' % (
            args.dit_model, args.t, args.k, args.ensemble_size, args.seed), 'w+') as json_file:
        json.dump(result, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # print("test")
    parser = argparse.ArgumentParser(description='EuroSAT Evaluation Script')
    parser.add_argument('--dataset_path', type=str, default='/dataset/EuroSAT', help='path to eurosat dataset')
    parser.add_argument('--save_path', type=str, default='Features/eurosat_flux', help='path to save features')
    parser.add_argument('--dit_model', choices=['flux'], default='flux', help="which dit version to use")
    parser.add_argument('--img_size', type=int, default=224,
                        help='''resize input image to this size before fed into diffusion model,
                            rounded up to nearest multiple of 16. by default is 224x224.''')
    parser.add_argument('--t', default=260, type=int, help='t for diffusion') ###调参[1,1000]
    parser.add_argument('--k', nargs='+', type=int, default=[28], help='which dit block to extract the ft map') ###调参[0,57]
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    parser.add_argument("--cd", action="store_true", default=False, help='whether adopt channel discard.')
    parser.add_argument('--label_fractions', nargs='+', type=float, default=[1, 5, 10, 50, 100],
                        help='label percentages to sweep over (e.g. 1 5 10 50 100)')
    parser.add_argument('--clf_epochs', type=int, default=50, help='epochs to train the linear probe')
    parser.add_argument('--clf_lr', type=float, default=1e-3, help='learning rate for linear probe')
    parser.add_argument('--clf_batch_size', type=int, default=256, help='batch size for linear probe training')
    parser.add_argument('--seed', type=int, default=42, help='random seed for label-fraction subsampling')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--overwrite_features', action='store_true',
                        help='re-extract features even if cache exists')
    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # print(args)
    main(args)
