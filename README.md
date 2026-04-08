# Out-of-Distribution Detection under Domain Shift

This project focuses on detecting **out-of-distribution (OOD)** samples using deep learning models trained on limited and shifting data distributions.

The goal is to build a system that can identify inputs that **do not follow the training distribution**, which is important for real-world applications where data is noisy and unpredictable.

---

##  Overview

- Train models on in-distribution (ID) data
- Detect samples that are **different from training data**
- Handle **domain shift** (train on one domain, test on another)
- Improve feature representation for better separation between ID and OOD

---

##  Key Ideas

### 1. Energy-based OOD Detection
We use **energy scores** to detect anomalies:
- Lower energy → likely in-distribution  
- Higher energy → likely out-of-distribution  

---

### 2. Neural Collapse (NC1, NC2)
We improve feature quality using:
- **NC1**: makes features of the same class compact  
- **NC2**: enforces separation between different classes  

Neural Collapse is a known phenomenon where features converge to structured geometry during training :contentReference[oaicite:0]{index=0}

---

### 3. ETF Classifier
We use **Equiangular Tight Frame (ETF)** classifiers to:
- enforce uniform class separation  
- stabilize training  

---

### 4. Domain Generalization
We train on multiple domains and test on an unseen domain:
- Train: art, cartoon, photo  
- Test: sketch  

---

### 5. Outlier Exposure (Synthetic OOD)
We generate synthetic anomalies during training to improve robustness.

---

##  Results

| Metric | Value |
|------|------|
| AUROC | ~0.74 |
| FPR@95 | ~0.73 |
| Energy Gap | 0.03 → 0.43 |

Key observation:
- Significant improvement in **energy separation between ID and OOD samples**
- Better calibration of anomaly detection

---

##  Datasets

- CIFAR-100 (controlled OOD setup)
- PACS dataset (domain shift setup)

---

## ⚙️ Training

Example command:

```bash
python ood_dg.py --method all --gpu 0 --batch 128 --epochs 100
