import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import trange, tqdm
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================
# ARGS
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",     type=int,   default=0)
parser.add_argument("--batch",   type=int,   default=32)
parser.add_argument("--epochs",  type=int,   default=100)
parser.add_argument("--workers", type=int,   default=4)
parser.add_argument("--method",  type=str,   default="all",
                    choices=["ERM", "ERM+NC1", "ERM+NC1+NC2", "all"])
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# ==============================
# CONFIG
# ==============================
DATA_ROOT = "/home/23dcs505/datasets/PACS/pacs_data/pacs_data"
OUT_FILE  = "pacs_ood_results.csv"

TRAIN_DOMAINS = ["art_painting", "cartoon", "photo"]
TEST_DOMAIN   = "sketch"

# PACS classes (ImageFolder sorts alphabetically):
# 0:dog  1:elephant  2:giraffe  3:guitar  4:horse  5:house  6:person
ALL_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

# Axis-3 split
# ID  (seen during training):  dog, elephant, horse, person  → indices 0,1,4,6
# OOD (never seen):            giraffe, guitar, house        → indices 2,3,5
ID_CLASS_NAMES  = ["dog", "elephant", "horse", "person"]
OOD_CLASS_NAMES = ["giraffe", "guitar", "house"]

ID_ORIG_INDICES  = [ALL_CLASSES.index(c) for c in ID_CLASS_NAMES]   # [0,1,4,6]
OOD_ORIG_INDICES = [ALL_CLASSES.index(c) for c in OOD_CLASS_NAMES]  # [2,3,5]

NUM_ID_CLASSES = len(ID_CLASS_NAMES)  # 4

# Training hyper-params
lr           = 5e-5
weight_decay = 1e-4
batch_size   = args.batch
epochs       = args.epochs
num_workers  = args.workers
eval_interval = 10

# NC loss weights
lambda_nc1 = 0.1
lambda_nc2 = 0.1
nc_start_epoch = 5

methods = [
    {"name": "ERM",        "nc1": False, "nc2": False},
    {"name": "ERM+NC1",    "nc1": True,  "nc2": False},
    {"name": "ERM+NC1+NC2","nc1": True,  "nc2": True},
]

if args.method != "all":
    methods = [m for m in methods if m["name"] == args.method]

# ==============================
# SEED
# ==============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==============================
# TRANSFORMS  (ResNet-50 ImageNet stats)
# ==============================
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)

train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

# ==============================
# DATASET HELPERS
# ==============================
def filter_by_orig_indices(dataset, orig_indices):
    """
    Keep only samples whose original ImageFolder label is in orig_indices.
    Returns a Subset and a label-remap dict {orig_label -> new_label}.
    """
    remap = {orig: new for new, orig in enumerate(sorted(orig_indices))}

    # FIX: Iterate over dataset.targets, NOT the dataset itself
    kept = [i for i, y in enumerate(dataset.targets) if y in orig_indices]

    return Subset(dataset, kept), remap


class RemappedSubset(torch.utils.data.Dataset):
    """Wraps a Subset and remaps labels so ID classes are 0..C-1."""
    def __init__(self, subset, remap):
        self.subset = subset
        self.remap  = remap

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, self.remap[y]


def make_train_loaders():
    loaders = []
    for d in TRAIN_DOMAINS:
        print(f"  [data] loading train domain: {d} ...", flush=True)
        ds_full = datasets.ImageFolder(os.path.join(DATA_ROOT, d), train_tf)
        subset, remap = filter_by_orig_indices(ds_full, ID_ORIG_INDICES)
        ds = RemappedSubset(subset, remap)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=(num_workers > 0),
            persistent_workers=(num_workers > 0),
        )
        print(f"         → {len(ds)} samples, {len(loader)} steps", flush=True)
        loaders.append(loader)
    return loaders


def make_test_loaders():
    """
    Returns two loaders for the sketch domain:
      - id_loader  : sketch images of ID classes   (labels remapped 0..3)
      - ood_loader : sketch images of OOD classes  (labels are original — not used for acc)
    """
    print(f"  [data] loading test domain: {TEST_DOMAIN} ...", flush=True)
    ds_full = datasets.ImageFolder(os.path.join(DATA_ROOT, TEST_DOMAIN), val_tf)

    id_subset,  id_remap  = filter_by_orig_indices(ds_full, ID_ORIG_INDICES)
    ood_subset, _         = filter_by_orig_indices(ds_full, OOD_ORIG_INDICES)

    id_ds  = RemappedSubset(id_subset,  id_remap)
    ood_ds = ood_subset

    id_loader  = DataLoader(id_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(num_workers > 0),
                            persistent_workers=(num_workers > 0))
    ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(num_workers > 0),
                            persistent_workers=(num_workers > 0))

    print(f"         → ID: {len(id_ds)} samples | OOD: {len(ood_ds)} samples", flush=True)
    return id_loader, ood_loader

# ==============================
# MODEL  (mirrors resnet.py)
# ==============================
class ResNet50Model(nn.Module):
    """
    Frozen ResNet-50 backbone → proj → MLP head → cosine classifier.
    Identical architecture to CLIPModel in resnet.py but uses ImageNet weights.
    """
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # [B,2048,1,1]
        self.proj    = nn.Linear(2048, 512)

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.fc = nn.Linear(512, num_classes, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))
        nn.init.normal_(self.fc.weight, std=0.01)

    def encode_image(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
            feats = feats.view(feats.size(0), -1)
        feats = self.proj(feats)
        return feats

    def forward(self, x, return_feats=False):
        feats      = self.encode_image(x).float()
        feats      = self.mlp(feats)
        
        # Stability normalization
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-6)
        feats_norm = F.normalize(feats, dim=1)
        
        # Unbounded logits via scaling
        logits = self.scale * self.fc(feats)
        
        if return_feats:
            return logits, feats_norm
        return logits

# ==============================
# LOSSES
# ==============================
def nc1_loss(feats, labels):
    """
    NC1 — within-class compactness.
    Minimise mean intra-class feature variance (on the hypersphere).
    """
    classes = torch.unique(labels)
    loss    = torch.tensor(0.0, device=feats.device)
    count   = 0
    for c in classes:
        cls_feats = feats[labels == c]
        if cls_feats.size(0) < 2:
            continue
        # variance on unit sphere: mean of squared deviations from class mean
        mean = F.normalize(cls_feats.mean(0, keepdim=True), dim=1)
        loss += (1 - (cls_feats * mean).sum(1)).mean()
        count += 1
    return loss / max(count, 1)


def nc2_loss(feats, labels, num_classes):
    """
    NC2 — ETF structure (standard formulation).
    Target: all pairwise cosine similarities between class means = -1/(C-1).
    Loss: MSE between actual Gram matrix and target Gram matrix.

    G_target[i,i] = 1
    G_target[i,j] = -1/(C-1)  for i ≠ j
    """
    classes = torch.unique(labels)
    means   = []
    for c in classes:
        cls_feats = feats[labels == c]
        if cls_feats.size(0) >= 1:
            means.append(F.normalize(cls_feats.mean(0), dim=0))

    if len(means) < 2:
        return torch.tensor(0.0, device=feats.device)

    C    = num_classes
    M    = torch.stack(means, dim=0)          # [K, D], K ≤ C present in batch
    G    = M @ M.T                            # [K, K] cosine similarities

    # Build ETF target for the K classes present
    target = torch.full((len(means), len(means)),
                        fill_value=-1.0 / (C - 1),
                        device=feats.device)
    target.fill_diagonal_(1.0)

    return F.mse_loss(G, target)

# ==============================
# OOD SCORE  (energy)
# ==============================
def energy_score(logits):
    """Lower energy → more ID-like."""
    return -torch.logsumexp(logits, dim=1)

# ==============================
# EVALUATION
# ==============================
@torch.no_grad()
def evaluate_classification(model, id_loader, criterion):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    for x, y in id_loader:
        x, y    = x.to(device), y.to(device)
        logits  = model(x)
        preds   = logits.argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
        val_loss += criterion(logits, y).item() * y.size(0)
    acc  = correct / max(total, 1)
    loss = val_loss / max(total, 1)
    return acc, loss


@torch.no_grad()
def collect_energy_scores(model, loader):
    model.eval()
    scores = []
    for x, _ in loader:
        x      = x.to(device)
        logits = model(x)
        scores.extend(energy_score(logits).cpu().numpy())
    return np.array(scores)


@torch.no_grad()
def compute_nc1_metric(model, loader):
    """
    Compute NC1 (within-class feature variance) on a given loader.
    Lower is better (tighter clusters).
    """
    model.eval()
    all_feats, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        _, feats = model(x, return_feats=True)
        all_feats.append(feats.cpu())
        all_labels.append(y)
    feats  = torch.cat(all_feats)
    labels = torch.cat(all_labels)

    nc1, count = 0.0, 0
    for c in torch.unique(labels):
        f_c = feats[labels == c]
        if f_c.size(0) < 2:
            continue
        nc1   += torch.var(f_c, dim=0).mean().item()
        count += 1
    return nc1 / max(count, 1)


def compute_ood_metrics(id_scores, ood_scores):
    """
    id_scores  : energy scores for ID samples  (lower = more ID)
    ood_scores : energy scores for OOD samples (higher = more OOD)

    We flip sign so that higher score = more likely ID → standard AUROC convention.
    """
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([-id_scores, -ood_scores])   # flip: ID gets higher score

    auroc = roc_auc_score(labels, scores)

    fpr, tpr, _ = roc_curve(labels, scores)
    idx   = np.searchsorted(tpr, 0.95)
    fpr95 = fpr[min(idx, len(fpr) - 1)]

    energy_gap = ood_scores.mean() - id_scores.mean()   # positive = good separation

    return auroc, fpr95, energy_gap

# ==============================
# TRAIN ONE MODEL
# ==============================
def train_model(cfg, run_id, df):
    set_seed(42 + run_id)

    print(f"\n{'='*60}")
    print(f"[Run {run_id}] Method={cfg['name']}")
    print(f"  Train: {TRAIN_DOMAINS}  |  Test domain: {TEST_DOMAIN}")
    print(f"  ID classes : {ID_CLASS_NAMES}")
    print(f"  OOD classes: {OOD_CLASS_NAMES}")
    print(f"{'='*60}")

    # ---------- data ----------
    train_loaders          = make_train_loaders()
    test_id_loader, test_ood_loader = make_test_loaders()

    steps_per_epoch = max(len(l) for l in train_loaders)

    # ---------- model ----------
    print(f"  [model] initialising ResNet50Model on {device} ...", flush=True)
    model     = ResNet50Model(NUM_ID_CLASSES).to(device)
    print(f"  [model] done. starting training loop ...", flush=True)
    optimizer = torch.optim.Adam(
        list(model.proj.parameters()) +
        list(model.mlp.parameters()) +
        list(model.fc.parameters()) +
        [model.scale],
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # Best-checkpoint trackers
    best = dict(
        epoch=0, acc=0.0, val_loss=float("inf"),
        auroc=0.0, fpr95=1.0, energy_gap=0.0,
        nc1_train=0.0, nc1_test=0.0,
        ce=0.0, nc1_l=0.0, nc2_l=0.0,
    )

    #pbar = trange(epochs, desc=f"{cfg['name']} | run={run_id}")

    for epoch in range(epochs):
        model.train()
        ep_ce = ep_nc1 = ep_nc2 = ep_total = ep_steps = 0.0

        # cycle shorter loaders so all domains see the same number of steps
        from itertools import cycle as icycle
        iters = [
            icycle(iter(l)) if len(l) < steps_per_epoch else iter(l)
            for l in train_loaders
        ]

        pbar = tqdm(range(steps_per_epoch),
            desc=f"{cfg['name']} | Ep {epoch+1}/{epochs}",
            leave=False)

        for _ in pbar:
            batches = [next(it) for it in iters]
            optimizer.zero_grad()

            ce_val  = torch.tensor(0.0, device=device)
            nc1_val = torch.tensor(0.0, device=device)
            nc2_val = torch.tensor(0.0, device=device)

            all_feats, all_labels = [], []

            for x, y in batches:
                x, y = x.to(device), y.to(device)
                logits, feats_norm = model(x, return_feats=True)
                ce_val += criterion(logits, y)
                all_feats.append(feats_norm)
                all_labels.append(y)

            if cfg["nc1"] and epoch >= nc_start_epoch:
                feats_cat  = torch.cat(all_feats)
                labels_cat = torch.cat(all_labels)
                nc1_val    = nc1_loss(feats_cat, labels_cat)

            if cfg["nc2"] and epoch >= nc_start_epoch:
                feats_cat  = torch.cat(all_feats)
                labels_cat = torch.cat(all_labels)
                nc2_val    = nc2_loss(feats_cat, labels_cat, NUM_ID_CLASSES)

            total = ce_val + lambda_nc1 * nc1_val + lambda_nc2 * nc2_val
            total.backward()
            optimizer.step()

            ep_ce    += ce_val.item()
            ep_nc1   += nc1_val.item()
            ep_nc2   += nc2_val.item()
            ep_total += total.item()
            ep_steps += 1

            avg_total = ep_total / max(ep_steps, 1)
            pbar.set_postfix(loss=f"{avg_total:.4f}", best_auroc=f"{best['auroc']:.4f}")

        # ---------- evaluate every eval_interval epochs ----------
        if (epoch + 1) % eval_interval == 0:
            acc, val_loss = evaluate_classification(model, test_id_loader, criterion)

            id_scores  = collect_energy_scores(model, test_id_loader)
            ood_scores = collect_energy_scores(model, test_ood_loader)
            auroc, fpr95, energy_gap = compute_ood_metrics(id_scores, ood_scores)

            nc1_train = compute_nc1_metric(model, train_loaders[0])  # proxy (art domain)
            nc1_test  = compute_nc1_metric(model, test_id_loader)

            print(
                f"  Ep {epoch+1:3d} | Scale={model.scale.item():.2f} | "
                f"Acc={acc:.4f}  ValLoss={val_loss:.4f} | "
                f"AUROC={auroc:.4f}  FPR95={fpr95:.4f}  EGap={energy_gap:.4f} | "
                f"NC1_train={nc1_train:.5f}  NC1_test={nc1_test:.5f} | "
                f"CE={ep_ce/ep_steps:.4f}  NC1_L={ep_nc1/ep_steps:.4f}  NC2_L={ep_nc2/ep_steps:.4f}"
            )

            # track best by AUROC (primary OOD metric)
            if auroc > best["auroc"]:
                best.update(
                    epoch     = epoch + 1,
                    acc       = acc,
                    val_loss  = val_loss,
                    auroc     = auroc,
                    fpr95     = fpr95,
                    energy_gap= energy_gap,
                    nc1_train = nc1_train,
                    nc1_test  = nc1_test,
                    ce        = ep_ce  / ep_steps,
                    nc1_l     = ep_nc1 / ep_steps,
                    nc2_l     = ep_nc2 / ep_steps,
                )

    print(
        f"\n  [BEST] Ep={best['epoch']} | "
        f"Acc={best['acc']:.4f} | "
        f"AUROC={best['auroc']:.4f}  FPR95={best['fpr95']:.4f}  EGap={best['energy_gap']:.4f} | "
        f"NC1_test={best['nc1_test']:.5f}"
    )

    df.loc[len(df)] = [
        run_id,
        cfg["name"],
        str(TRAIN_DOMAINS),
        TEST_DOMAIN,
        str(ID_CLASS_NAMES),
        str(OOD_CLASS_NAMES),
        best["epoch"],
        round(best["acc"],        4),
        round(best["val_loss"],   4),
        round(best["auroc"],      4),
        round(best["fpr95"],      4),
        round(best["energy_gap"], 4),
        round(best["nc1_train"],  6),
        round(best["nc1_test"],   6),
        round(best["ce"],         4),
        round(best["nc1_l"],      4),
        round(best["nc2_l"],      4),
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
        "run_id", "method",
        "train_domains", "test_domain",
        "id_classes", "ood_classes",
        "epoch",
        "acc_id",        # classification acc on ID classes in sketch
        "val_loss",
        "auroc",         # OOD detection AUROC  (ID vs OOD in sketch)
        "fpr95",         # FPR @ 95% TPR
        "energy_gap",    # mean(OOD energy) - mean(ID energy), higher = better
        "nc1_train",     # within-class variance on train domain (art proxy)
        "nc1_test",      # within-class variance on sketch ID
        "ce_loss",
        "nc1_loss",
        "nc2_loss",
    ]

    if os.path.exists(OUT_FILE):
        df = pd.read_csv(OUT_FILE)
        print(f"[INFO] Resuming — {len(df)} rows already logged")
    else:
        df = pd.DataFrame(columns=columns)

    for cfg in methods:
        mask = (df["method"] == cfg["name"])
        if mask.sum() >= 1:
            print(f"[SKIP] {cfg['name']} already done")
            continue
        df = df[~mask].reset_index(drop=True)
        df = train_model(cfg, run_id=1, df=df)

    print("\nAll experiments completed.")
    print(df[["method", "acc_id", "auroc", "fpr95", "energy_gap",
              "nc1_test"]].to_string(index=False))

if __name__ == "__main__":
    run()
