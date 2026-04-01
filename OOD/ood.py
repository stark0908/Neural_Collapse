import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

# =========================
# Args
# =========================
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--use_nc', action='store_true')
args = parser.parse_args()

device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

# =========================
# Dataset (80 / 20 split)
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_full = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

id_classes = list(range(80))
ood_classes = list(range(80, 100))

def filter_dataset(dataset, class_list):
    indices = [i for i, (_, y) in enumerate(dataset) if y in class_list]
    return Subset(dataset, indices)

train_id = filter_dataset(train_full, id_classes)
test_id = filter_dataset(test_full, id_classes)
test_ood = filter_dataset(test_full, ood_classes)

trainloader = DataLoader(train_id, batch_size=args.batch, shuffle=True,
                         num_workers=8, pin_memory=True)

testloader_id = DataLoader(test_id, batch_size=args.batch, shuffle=False,
                           num_workers=8, pin_memory=True)

testloader_ood = DataLoader(test_ood, batch_size=args.batch, shuffle=False,
                            num_workers=8, pin_memory=True)

# =========================
# Model (with feature extraction)
# =========================
import torchvision.models as models

class ResNet18Feat(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        base = models.resnet18()
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feat=False):
        feat = self.encoder(x)
        feat = feat.view(feat.size(0), -1)

        if return_feat:
            return feat

        logits = self.fc(feat)
        return logits

model = ResNet18Feat(num_classes=80).to(device)

# =========================
# Optimizer & Loss
# =========================
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1 if args.use_nc else 0.0)

# =========================
# NC1 computation
# =========================
def compute_nc1(model, dataloader):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            feat = model(x, return_feat=True)

            features.append(feat.cpu())
            labels.append(y)

    features = torch.cat(features)
    labels = torch.cat(labels)

    nc1 = 0
    classes = torch.unique(labels)

    for c in classes:
        idx = labels == c
        f_c = features[idx]

        if f_c.shape[0] < 2:
            continue

        var = torch.var(f_c, dim=0).mean()
        nc1 += var.item()

    nc1 /= len(classes)
    return nc1

def compute_ood_metrics(id_scores, ood_scores):
    labels = np.concatenate([
        np.ones_like(id_scores),   # ID = 1
        np.zeros_like(ood_scores)  # OOD = 0
    ])

    scores = np.concatenate([id_scores, ood_scores])

    # IMPORTANT: flip sign (since energy lower = ID)
    scores = -scores

    # AUROC
    auroc = roc_auc_score(labels, scores)

    # FPR@95TPR
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]

    return auroc, fpr95

# =========================
# Energy score
# =========================
def energy_score(logits):
    return -torch.logsumexp(logits, dim=1)

# =========================
# OOD evaluation
# =========================
@torch.no_grad()
def evaluate_ood():
    model.eval()
    id_scores = []
    ood_scores = []

    for x, _ in testloader_id:
        x = x.to(device, non_blocking=True)
        logits = model(x)

        if args.use_nc:
            logits = F.normalize(logits, dim=1)

        energy = energy_score(logits)
        id_scores.extend(energy.cpu().numpy())

    for x, _ in testloader_ood:
        x = x.to(device, non_blocking=True)
        logits = model(x)

        if args.use_nc:
            logits = F.normalize(logits, dim=1)

        energy = energy_score(logits)
        ood_scores.extend(energy.cpu().numpy())

    return np.array(id_scores), np.array(ood_scores)

# =========================
# Training
# =========================


def train():
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(x)

            if args.use_nc:
                logits = F.normalize(logits, dim=1)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            avg_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # =========================
        # After each epoch
        # =========================

        # NC1
        nc1 = compute_nc1(model, testloader_id)

        # OOD scores
        id_scores, ood_scores = evaluate_ood()

        # Metrics
        auroc, fpr95 = compute_ood_metrics(id_scores, ood_scores)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {total_loss:.3f} | "
            f"NC1: {nc1:.6f} | "
            f"AUROC: {auroc:.4f} | "
            f"FPR95: {fpr95:.4f}"
        )

# =========================
# Metrics
# =========================
def evaluate_metrics(id_scores, ood_scores):
    print("\n=== OOD Detection ===")
    print(f"ID mean energy: {id_scores.mean():.4f}")
    print(f"OOD mean energy: {ood_scores.mean():.4f}")

    separation = ood_scores.mean() - id_scores.mean()
    print(f"Separation (OOD - ID): {separation:.4f}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    print("Training...")
    train()

    print("\nEvaluating OOD...")
    id_scores, ood_scores = evaluate_ood()

    evaluate_metrics(id_scores, ood_scores)
