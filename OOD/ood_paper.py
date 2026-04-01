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
parser.add_argument('--gpu', type=int, default=3)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--entropy_weight', type=float, default=0.0)  # NEW
args = parser.parse_args()

device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

# =========================
# Dataset (80 / 20 split)
# =========================
transform = transforms.ToTensor()

train_full = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=transform)
test_full = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=transform)

id_classes = list(range(80))
ood_classes = list(range(80, 100))

def filter_dataset(dataset, class_list):
    indices = [i for i, (_, y) in enumerate(dataset) if y in class_list]
    return Subset(dataset, indices)

train_id = filter_dataset(train_full, id_classes)
test_id = filter_dataset(test_full, id_classes)
test_ood = filter_dataset(test_full, ood_classes)

trainloader = DataLoader(train_id, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)
testloader_id = DataLoader(test_id, batch_size=args.batch, shuffle=False, num_workers=8, pin_memory=True)
testloader_ood = DataLoader(test_ood, batch_size=args.batch, shuffle=False, num_workers=8, pin_memory=True)

# =========================
# Model
# =========================
import torchvision.models as models

class ResNet18Feat(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        base = models.resnet18()
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, return_both=False):
        feat = self.encoder(x)
        feat = feat.view(feat.size(0), -1)

        # 🔥 standard linear with scaling (unbounded logits for energy OOD, raw features)
        logits = self.scale * self.fc(feat)

        if return_both:
            return logits, feat

        return logits

model = ResNet18Feat().to(device)

# =========================
# Optimizer
# =========================
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# =========================
# Entropy loss (NEW)
# =========================
def entropy_loss(feat, eps=1e-8):
    # normalize features
    feat = F.normalize(feat, eps=eps, p=2, dim=-1)
    # pairwise dot products (= inverse distance)
    dots = torch.mm(feat, feat.t())
    n = feat.shape[0]
    dots.view(-1)[:: (n + 1)].fill_(-1)  # fill diagonal with -1
    # max inner prod -> min distance
    _, I = torch.max(dots, dim=1)
    
    # pairwise distance and loss calculation
    distances = F.pairwise_distance(feat, feat[I], p=2, eps=eps)
    loss = -torch.log(distances + eps).mean()
    return loss

# =========================
# NC1
# =========================
def compute_nc1(model, dataloader):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            _, feat = model(x, return_both=True)
            features.append(feat.cpu())
            labels.append(y)

    features = torch.cat(features)
    labels = torch.cat(labels)

    nc1 = 0
    classes = torch.unique(labels)

    for c in classes:
        f_c = features[labels == c]
        if f_c.shape[0] < 2:
            continue
        nc1 += torch.var(f_c, dim=0).mean().item()

    return nc1 / len(classes)

# =========================
# OOD metrics
# =========================
def energy_score(logits):
    return -torch.logsumexp(logits, dim=1)

def compute_ood_metrics(id_scores, ood_scores):
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr95 = fpr[np.argmin(np.abs(tpr - 0.95))]

    return auroc, fpr95

# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate_ood():
    model.eval()
    id_scores, ood_scores = [], []

    for x, _ in testloader_id:
        x = x.to(device)
        logits = model(x)
        id_scores.extend(energy_score(logits).cpu().numpy())

    for x, _ in testloader_ood:
        x = x.to(device)
        logits = model(x)
        ood_scores.extend(energy_score(logits).cpu().numpy())

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
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # 🔥 Fix: single forward pass
            logits, feat = model(x, return_both=True)

            ce_loss = criterion(logits, y)

            # 🔥 entropy regularization removed for strict OOD calibration (as per paper)
            ent_loss = 0

            loss = ce_loss + args.entropy_weight * ent_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss/(pbar.n+1)})

        # ===== evaluation =====
        nc1 = compute_nc1(model, testloader_id)
        id_s, ood_s = evaluate_ood()
        auroc, fpr95 = compute_ood_metrics(id_s, ood_s)

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {total_loss/len(trainloader):.4f} | "
            f"NC1: {nc1:.6f} | "
            f"AUROC: {auroc:.4f} | "
            f"FPR95: {fpr95:.4f}"
        )

# =========================
# Main
# =========================
if __name__ == "__main__":
    train()

    id_s, ood_s = evaluate_ood()
    print("\n=== Final OOD ===")
    print("ID mean:", id_s.mean())
    print("OOD mean:", ood_s.mean())
