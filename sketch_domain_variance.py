import os
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
import argparse
from tqdm import trange
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=32, help="Batch size")
parser.add_argument("--gpu",   type=int, default=0,  help="GPU ID to use")
args, _ = parser.parse_known_args()

# ==============================
# DEVICE
# ==============================
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# ==============================
# CONFIG
# ==============================
DATA_ROOT = "/home/23dcs505/datasets/PACS/pacs_data/pacs_data"
OUT_FILE  = "sketch_baseline_domain.csv"

TRAIN_DOMAINS = ["art_painting", "cartoon", "photo"]
TEST_DOMAINS  = ["sketch"]

epochs         = 100          # 10 checkpoints at every 10th epoch
eval_interval  = 10            # evaluate every N epochs to track best (log once per run)
num_runs       = 1           # independent runs per (fraction, method)
batch_size     = args.batch
lr             = 5e-5
weight_decay   = 1e-4

lambda_coral   = 1.0
lambda_nc      = 0.01
lambda_dm      = 0.1   # start small
nc_start_epoch = 5

fractions = [0.1, 0.2, 0.5, 0.8]

methods = [
    {"name": "ERM",      "coral": False, "nc": False},
    {"name": "ERM+NC",   "coral": False, "nc": True},
    {"name": "CORAL",    "coral": True,  "nc": False},
    {"name": "CORAL+NC", "coral": True,  "nc": True},
]

num_workers = 16

# ==============================
# SEED
# ==============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==============================
# CLIP TRANSFORMS
# ==============================
clip_mean = (0.48145466, 0.4578275,  0.40821073)
clip_std  = (0.26862954, 0.26130258, 0.27577711)

train_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(clip_mean, clip_std),
])
val_tf = train_tf

# ==============================
# MODEL
# ==============================
class CLIPModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model, _ = clip.load("ViT-B/16", device=device)
        for p in self.model.parameters():
            p.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.fc = nn.Linear(512, num_classes, bias=False)

    def forward(self, x, return_feats=False):
        with torch.no_grad():
            feats = self.model.encode_image(x)
        feats      = feats.float()
        feats      = self.mlp(feats)
        feats_raw  = feats                      # pre-norm: use for CORAL
        feats_norm = F.normalize(feats, dim=1)  # post-norm: use for CE + NC
        logits     = self.fc(feats_norm)
        return (logits, feats_norm, feats_raw) if return_feats else logits

# ==============================
# LOSSES
# ==============================
def coral_loss(feats_list):
    covs = []
    for feats in feats_list:
        if feats.size(0) < 2:
            continue
        xm  = feats - feats.mean(0, keepdim=True)
        cov = xm.T @ xm / (feats.size(0) - 1)
        covs.append(cov)

    if len(covs) < 2:
        return torch.tensor(0.0, device=device)

    d     = covs[0].size(0)
    loss  = torch.tensor(0.0, device=device)
    count = 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            diff   = covs[i] - covs[j]
            loss  += torch.norm(diff, p='fro') ** 2 / (4 * d * d)
            count += 1
    return loss / max(count, 1)


def nc_loss(feats, labels):
    classes = torch.unique(labels)
    means   = []
    for c in classes:
        cls_feats = feats[labels == c]
        if len(cls_feats) > 1:
            means.append(cls_feats.mean(0))

    if len(means) < 2:
        return torch.tensor(0.0, device=device)

    means = F.normalize(torch.stack(means), dim=1)
    G     = means @ means.T
    mask  = ~torch.eye(G.size(0), device=device).bool()
    return (G[mask] ** 2).mean()

def domain_mean_variance_loss(feats_list, labels_list):
    """
    feats_list  : list of [B_i, D] tensors (per domain)
    labels_list : list of [B_i] tensors (per domain)
    """

    num_domains = len(feats_list)
    device = feats_list[0].device

    # collect all classes present in batch
    all_labels = torch.cat(labels_list)
    classes = torch.unique(all_labels)

    loss = torch.tensor(0.0, device=device)
    count = 0

    for c in classes:
        domain_means = []

        for feats, labels in zip(feats_list, labels_list):
            mask = (labels == c)
            if mask.sum() > 0:
                domain_means.append(feats[mask].mean(dim=0))

        # need at least 2 domains for variance
        if len(domain_means) < 2:
            continue

        M = torch.stack(domain_means, dim=0)  # [D, 512]

        # variance across domain axis
        var = M.var(dim=0, unbiased=False).mean()

        loss += var
        count += 1

    return loss / max(count, 1)

# ==============================
# EVAL HELPER
# ==============================
def evaluate(model, test_loaders, criterion):
    """Returns (avg_acc, worst_acc, avg_val_loss) over all test loaders."""
    model.eval()
    domain_accs   = []
    domain_losses = []

    with torch.no_grad():
        for loader in test_loaders:
            correct, total, val_loss = 0, 0, 0.0
            for x, y in loader:
                x, y   = x.to(device), y.to(device)
                logits  = model(x)
                preds   = logits.argmax(1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
                val_loss += criterion(logits, y).item() * y.size(0)
            domain_accs.append(correct / total)
            domain_losses.append(val_loss / max(total, 1))

    return (
        float(np.mean(domain_accs)),
        float(np.min(domain_accs)),
        float(np.mean(domain_losses)),
    )

# ==============================
# TRAIN
# ==============================
def train_model(frac, cfg, run_id, df):
    """
    Trains one model and appends checkpoint rows to df.
    Returns updated df.
    """
    print(f"\n[Run {run_id}] frac={frac} | {cfg['name']} | "
          f"TRAIN: {TRAIN_DOMAINS} → TEST: {TEST_DOMAINS}")

    # ---------- data ----------
    train_loaders = []
    for d in TRAIN_DOMAINS:
        ds     = datasets.ImageFolder(os.path.join(DATA_ROOT, d), train_tf)
        idx    = torch.randperm(len(ds))
        k      = max(1, int(frac * len(ds)))
        subset = Subset(ds, idx[:k])
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        train_loaders.append(loader)

    test_loaders = []
    for d in TEST_DOMAINS:
        ds     = datasets.ImageFolder(os.path.join(DATA_ROOT, d), val_tf)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loaders.append(loader)

    num_classes = len(
        datasets.ImageFolder(os.path.join(DATA_ROOT, TRAIN_DOMAINS[0])).classes
    )

    # ---------- model ----------
    model     = CLIPModel(num_classes).to(device)
    optimizer = optim.Adam(
        list(model.mlp.parameters()) + list(model.fc.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = max(len(l) for l in train_loaders)

    # Best-checkpoint trackers
    best_avg_acc   = -1.0
    best_worst_acc = -1.0
    best_val_loss  = float("inf")
    best_epoch     = -1
    best_ce        = 0.0
    best_coral     = 0.0
    best_nc        = 0.0
    best_train     = 0.0
    best_dm        = 0.0

    eval_interval  = 10   # evaluate every 10 epochs internally to find best

    pbar = trange(epochs, desc=f"Run {run_id} | {cfg['name']} | frac={frac}")

    for epoch in pbar:
        train_iters = [
            cycle(iter(loader)) if len(loader) < steps_per_epoch else iter(loader)
            for loader in train_loaders
        ]

        model.train()
        epoch_ce_loss    = 0.0
        epoch_coral_loss = 0.0
        epoch_nc_loss    = 0.0
        epoch_total_loss = 0.0
        epoch_steps      = 0
        epoch_dm_loss    = 0.0

        for _ in range(steps_per_epoch):
            batches = [next(it) for it in train_iters]
            optimizer.zero_grad()

            ce_val    = torch.tensor(0.0, device=device)
            coral_val = torch.tensor(0.0, device=device)
            nc_val    = torch.tensor(0.0, device=device)
            dm_val    = torch.tensor(0.0, device=device)

            feats_norm_list = []   # post-norm → NC loss
            feats_raw_list  = []   # pre-norm  → CORAL loss
            labels_list     = []

            for x, y in batches:
                x, y = x.to(device), y.to(device)
                logits, feats_norm, feats_raw = model(x, return_feats=True)
                ce_val += criterion(logits, y)
                feats_norm_list.append(feats_norm)
                feats_raw_list.append(feats_raw)
                labels_list.append(y)

            if len(TRAIN_DOMAINS) > 1:
                dm_val = domain_mean_variance_loss(feats_norm_list, labels_list)

            if cfg["nc"] and epoch >= nc_start_epoch:
                all_feats  = torch.cat(feats_norm_list, dim=0)
                all_labels = torch.cat(labels_list,     dim=0)
                nc_val     = nc_loss(all_feats, all_labels)

            if cfg["coral"] and len(TRAIN_DOMAINS) > 1:
                if all(f.size(0) >= 2 for f in feats_raw_list):
                    coral_val = coral_loss(feats_raw_list)
                else:
                    print("[WARN] Skipping CORAL this step: a domain batch has < 2 samples")

            total = ce_val + lambda_coral * coral_val + lambda_nc * nc_val + lambda_dm * dm_val
            total.backward()
            optimizer.step()

            epoch_ce_loss    += ce_val.item()
            epoch_coral_loss += coral_val.item()
            epoch_nc_loss    += nc_val.item()
            epoch_total_loss += total.item()
            epoch_steps      += 1
            epoch_dm_loss    += dm_val.item()

        avg_train = epoch_total_loss / max(epoch_steps, 1)
        pbar.set_postfix(loss=f"{avg_train:.4f}", best=f"{best_avg_acc:.4f}")

        # ---------- evaluate every eval_interval epochs to track best ----------
        if (epoch + 1) % eval_interval == 0:
            avg_acc, worst_acc, avg_val_loss = evaluate(model, test_loaders, criterion)

            print(
                f"  Epoch {epoch+1:3d} | "
                f"Acc={avg_acc:.4f}  Worst={worst_acc:.4f} | "
                f"CE={epoch_ce_loss/max(epoch_steps,1):.4f}  "
                f"CORAL={epoch_coral_loss/max(epoch_steps,1):.4f}  "
                f"NC={epoch_nc_loss/max(epoch_steps,1):.4f}  "
                f"Train={avg_train:.4f}  Val={avg_val_loss:.4f} "
                f"DM={epoch_dm_loss/max(epoch_steps,1):.4f}"
            )

            if avg_acc > best_avg_acc:
                best_avg_acc   = avg_acc
                best_worst_acc = worst_acc
                best_val_loss  = avg_val_loss
                best_epoch     = epoch + 1
                best_ce        = epoch_ce_loss    / max(epoch_steps, 1)
                best_coral     = epoch_coral_loss / max(epoch_steps, 1)
                best_nc        = epoch_nc_loss    / max(epoch_steps, 1)
                best_train     = avg_train
                best_dm = epoch_dm_loss / max(epoch_steps, 1)

    # ---------- log single best row for this run ----------
    print(
        f"\n  [BEST] Epoch {best_epoch} | "
        f"Acc={best_avg_acc:.4f}  Worst={best_worst_acc:.4f} | "
        f"CE={best_ce:.4f}  CORAL={best_coral:.4f}  NC={best_nc:.4f}  DM={best_dm:.4f}  "
        f"Train={best_train:.4f}  Val={best_val_loss:.4f}"
    )

    df.loc[len(df)] = [
        run_id,
        str(TRAIN_DOMAINS),
        str(TEST_DOMAINS),
        frac,
        cfg["name"],
        best_epoch,
        round(best_avg_acc,   4),
        round(best_worst_acc, 4),
        round(best_ce,        4),
        round(best_coral,     4),
        round(best_nc,        4),
        round(best_dm,        4),
        round(best_train,     4),
        round(best_val_loss,  4),
    ]

    tmp = OUT_FILE + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, OUT_FILE)

    return df

# ==============================
# MAIN
# ==============================
def run():
    columns = [
        "run_id",
        "train_domains", "test_domains",
        "fraction", "method",
        "epoch",
        "avg_acc", "worst_acc",
        "ce_loss", "coral_loss", "nc_loss", "dm_loss",
        "train_loss", "val_loss",
    ]

    if os.path.exists(OUT_FILE):
        df = pd.read_csv(OUT_FILE)
        print(f"[INFO] Resuming from {len(df)} logged checkpoints")
    else:
        df = pd.DataFrame(columns=columns)

    for frac in fractions:
        for cfg in methods:
            for run_id in range(1, num_runs + 1):

                # Check if ALL 10 checkpoints for this (run_id, frac, method) exist
                mask = (
                    (df["run_id"]   == run_id)       &
                    (df["fraction"] == frac)          &
                    (df["method"]   == cfg["name"])
                )
                if mask.sum() >= 1:  # 1 row per run (best epoch only)
                    print(f"[SKIP] run={run_id} frac={frac} method={cfg['name']}")
                    continue

                # If partially done, drop incomplete checkpoints and redo the run
                df = df[~mask].reset_index(drop=True)

                df = train_model(frac, cfg, run_id, df)

    print("\nAll experiments completed.")

if __name__ == "__main__":
    run()
