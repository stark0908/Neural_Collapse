
import os
import itertools
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from itertools import cycle
import clip

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# CONFIG
# ==============================
DATA_ROOT = "/home/23dcs505/datasets/PACS/pacs_data/pacs_data"
OUT_FILE = "clip_pacs_results.csv"

epochs = 30
batch_size = 32
lr = 5e-5
weight_decay = 1e-4

lambda_coral = 1.0
lambda_nc = 0.01
nc_start_epoch = 5

fractions = [0.1, 0.2, 0.5, 0.8]
domains = ["art_painting", "cartoon", "photo", "sketch"]

methods = [
    {"name": "ERM", "coral": False, "nc": False},
    {"name": "ERM+NC", "coral": False, "nc": True},
    {"name": "CORAL", "coral": True, "nc": False},
    {"name": "CORAL+NC", "coral": True, "nc": True},
]

num_workers = 8  # or higher if DGX

# ==============================
# SEED
# ==============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ==============================
# CLIP TRANSFORMS
# ==============================
clip_mean = (0.48145466, 0.4578275, 0.40821073)
clip_std  = (0.26862954, 0.26130258, 0.27577711)

train_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(clip_mean, clip_std)
])

val_tf = train_tf

# ==============================
# MODEL
# ==============================
class CLIPModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model, _ = clip.load("ViT-B/16", device=device)

        # Freeze CLIP backbone
        for p in self.model.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(512, num_classes, bias=False)

    def forward(self, x, return_feats=False):
        with torch.no_grad():
            feats = self.model.encode_image(x)

        # 🔥 FIX: dtype mismatch
        feats = feats.float()

        feats = F.normalize(feats, dim=1)

        logits = self.fc(feats)

        return (logits, feats) if return_feats else logits

# ==============================
# LOSSES
# ==============================
def coral_loss(feats_list):
    covs = []
    for feats in feats_list:
        if feats.size(0) < 2:
            continue
        xm = feats - feats.mean(0, keepdim=True)
        cov = xm.T @ xm / (feats.size(0) - 1)
        covs.append(cov)

    if len(covs) == 0:
        return torch.tensor(0.0, device=device)

    d = covs[0].size(0)
    loss, count = 0.0, 0

    for i in range(len(covs)):
        for j in range(i+1, len(covs)):
            diff = covs[i] - covs[j]
            loss += torch.norm(diff, p='fro')**2 / (4 * d * d)
            count += 1

    if isinstance(loss, float):
        return torch.tensor(loss / max(count, 1), device=device)
    return loss / max(count, 1)


def nc_loss(feats, labels):
    classes = torch.unique(labels)
    means = []

    for c in classes:
        cls_feats = feats[labels == c]
        if len(cls_feats) > 1:
            means.append(cls_feats.mean(0))

    if len(means) < 2:
        return torch.tensor(0.0, device=device)

    means = F.normalize(torch.stack(means), dim=1)
    G = means @ means.T
    mask = ~torch.eye(G.size(0), device=device).bool()

    return (G[mask] ** 2).mean()

# ==============================
# TRAIN
# ==============================
def train_model(train_domains, test_domains, frac, cfg):

    print(f"\nTRAIN: {train_domains} → TEST: {test_domains} | frac={frac} | {cfg['name']}")

    train_loaders = []

    for d in train_domains:
        ds = datasets.ImageFolder(os.path.join(DATA_ROOT, d), train_tf)

        idx = torch.randperm(len(ds))
        k = int(frac * len(ds))
        subset = Subset(ds, idx[:k])

        loader = DataLoader(subset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=num_workers,
                            pin_memory=True)

        train_loaders.append(loader)

    test_loaders = []
    for d in test_domains:
        ds = datasets.ImageFolder(os.path.join(DATA_ROOT, d), val_tf)
        loader = DataLoader(ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
        test_loaders.append(loader)

    num_classes = len(datasets.ImageFolder(
        os.path.join(DATA_ROOT, train_domains[0])).classes)

    model = CLIPModel(num_classes).to(device)

    optimizer = optim.Adam(model.fc.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    # Balanced iteration across domains
    train_iters = [cycle(loader) for loader in train_loaders]
    steps_per_epoch = min(len(l) for l in train_loaders)

    for epoch in range(epochs):
        model.train()

        for _ in range(steps_per_epoch):

            batches = [next(it) for it in train_iters]

            optimizer.zero_grad()
            total_loss = 0.0
            feats_list = []

            for x, y in batches:
                x, y = x.to(device), y.to(device)

                logits, feats = model(x, return_feats=True)
                total_loss += criterion(logits, y)
                feats_list.append(feats)

                if cfg["nc"] and epoch >= nc_start_epoch:
                    total_loss += lambda_nc * nc_loss(feats, y)

            if cfg["coral"] and len(train_domains) > 1:
                total_loss += lambda_coral * coral_loss(feats_list)

            total_loss.backward()
            optimizer.step()

    # ======================
    # EVAL
    # ======================
    model.eval()
    domain_accs = []

    with torch.no_grad():
        for loader in test_loaders:
            correct, total = 0, 0

            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)

                correct += (preds == y).sum().item()
                total += y.size(0)

            domain_accs.append(correct / total)

    avg_acc = np.mean(domain_accs)
    worst_acc = np.min(domain_accs)

    print(f"Avg Acc: {avg_acc:.4f}, Worst: {worst_acc:.4f}")

    return avg_acc, worst_acc

# ==============================
# MAIN
# ==============================
def run():

    if os.path.exists(OUT_FILE):
        df = pd.read_csv(OUT_FILE)
        print(f"[INFO] Resuming from {len(df)} runs")
    else:
        df = pd.DataFrame(columns=[
            "train_domains", "test_domains",
            "fraction", "method",
            "avg_acc", "worst_acc"
        ])

    regimes = []

    # 3→1
    for test in domains:
        train = [d for d in domains if d != test]
        regimes.append((train, [test]))

    # 2→2
    for train_pair in itertools.combinations(domains, 2):
        test_pair = [d for d in domains if d not in train_pair]
        regimes.append((list(train_pair), test_pair))

    # 1→3
    for train in domains:
        test = [d for d in domains if d != train]
        regimes.append(([train], test))

    for train_domains, test_domains in regimes:
        for frac in fractions:
            for cfg in methods:

                mask = (
                    (df["train_domains"] == str(train_domains)) &
                    (df["test_domains"] == str(test_domains)) &
                    (np.isclose(df["fraction"], frac)) &
                    (df["method"] == cfg["name"])
                )

                if mask.any():
                    print(f"[SKIP] {train_domains} | {frac} | {cfg['name']}")
                    continue

                avg_acc, worst_acc = train_model(
                    train_domains,
                    test_domains,
                    frac,
                    cfg
                )

                df.loc[len(df)] = [
                    str(train_domains),
                    str(test_domains),
                    frac,
                    cfg["name"],
                    avg_acc,
                    worst_acc
                ]

                # atomic save (no corruption)
                tmp_file = OUT_FILE + ".tmp"
                df.to_csv(tmp_file, index=False)
                os.replace(tmp_file, OUT_FILE)

    print("\nAll experiments completed.")

if __name__ == "__main__":
    run()
