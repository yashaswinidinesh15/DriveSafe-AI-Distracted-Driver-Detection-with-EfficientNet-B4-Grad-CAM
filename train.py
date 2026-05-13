"""
DriveSafe-AI — Full Training Pipeline
EfficientNet-B4 + MobileViT-S | Dual Dataset | Grad-CAM
Authors: Keerthana Murulidharan & Yashaswini Dinesh
CMPE 258 · SJSU · Spring 2025
"""

import os, random, warnings, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Seed & Device ──────────────────────────────────────────────────────────────
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)
print(f"Device: {DEVICE}")

# ── Classes ────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Safe Driving","Texting (Right)","Phone (Right)","Texting (Left)",
    "Phone (Left)","Radio Adjustment","Drinking","Reaching Behind",
    "Hair & Makeup","Talking Passenger",
]
SF_FOLDERS  = [f"c{i}" for i in range(10)]
NUM_CLASSES = 10

# ── Transforms ─────────────────────────────────────────────────────────────────
IMG_SIZE = 224

def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.3),
        A.Affine(translate_percent=0.05, scale=(0.9,1.1), rotate=(-15,15), p=0.5),
        A.OneOf([A.MotionBlur(5), A.GaussianBlur(5), A.MedianBlur(5)], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(0.2, 0.2),
            A.HueSaturationValue(10, 20, 10),
        ], p=0.4),
        A.CoarseDropout(num_holes_range=(1,8), hole_height_range=(8,16),
                        hole_width_range=(8,16), fill=0, p=0.3),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])

# ── Datasets ───────────────────────────────────────────────────────────────────
class StateFarmDataset(Dataset):
    def __init__(self, root, split="train", transform=None, val_frac=0.15):
        root = Path(root); self.transform = transform; self.samples = []
        img_dir = root / "imgs" / "train"
        if not img_dir.exists():
            print(f"[WARN] State Farm not found at {img_dir}"); return
        all_s = []
        for i, f in enumerate(SF_FOLDERS):
            for p in (img_dir / f).glob("*.jpg"):
                all_s.append((str(p), i))
        csv = root / "driver_imgs_list.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            drivers = df["subject"].unique()
            np.random.seed(42); np.random.shuffle(drivers)
            val_d = set(drivers[:max(1, int(len(drivers)*val_frac))])
            dmap  = {r["img"]: r["subject"] for _, r in df.iterrows()}
            if split == "train":
                self.samples = [(p,l) for p,l in all_s if dmap.get(Path(p).name,"") not in val_d]
            else:
                self.samples = [(p,l) for p,l in all_s if dmap.get(Path(p).name,"") in val_d]
        else:
            cut = int(len(all_s)*(1-val_frac))
            self.samples = all_s[:cut] if split=="train" else all_s[cut:]
        print(f"  StateFarm [{split}]: {len(self.samples):,} samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, l = self.samples[i]
        img = np.array(Image.open(p).convert("RGB"))
        if self.transform: img = self.transform(image=img)["image"]
        return img, l


class AUCDataset(Dataset):
    FOLDER_MAP = {
        "c0":0,"c1":1,"c2":2,"c3":3,"c4":4,"c5":5,"c6":6,"c7":7,"c8":8,"c9":9,
        "normal":0,"safe":0,"texting_right":1,"talking_right":2,"texting_left":3,
        "talking_left":4,"radio":5,"adjusting_radio":5,"drinking":6,"drink":6,
        "reaching_behind":7,"reaching":7,"hair_makeup":8,"makeup":8,
        "hair_and_makeup":8,"talking_passenger":9,"passenger":9,
    }
    def __init__(self, root, transform=None):
        root = Path(root); self.transform = transform; self.samples = []
        if not root.exists():
            print(f"[WARN] AUC not found at {root}. Skipping."); return
        for folder in root.iterdir():
            if not folder.is_dir(): continue
            lbl = self.FOLDER_MAP.get(folder.name.lower())
            if lbl is None: continue
            for ext in ("*.jpg","*.jpeg","*.png"):
                for p in folder.glob(ext):
                    self.samples.append((str(p), lbl))
        print(f"  AUC Dataset: {len(self.samples):,} samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, l = self.samples[i]
        img = np.array(Image.open(p).convert("RGB"))
        if self.transform: img = self.transform(image=img)["image"]
        return img, l


class FusedDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = [d for d in datasets if len(d) > 0]
        self.lengths  = [len(d) for d in self.datasets]
        self.total    = sum(self.lengths)
        offs, acc = [], 0
        for l in self.lengths: offs.append(acc); acc += l
        self.offsets = offs
        print(f"  Fused: {self.total:,} samples from {len(self.datasets)} sources")

    def __len__(self): return self.total
    def __getitem__(self, idx):
        for i,(o,l) in enumerate(zip(self.offsets, self.lengths)):
            if idx < o + l: return self.datasets[i][idx - o]
        raise IndexError(idx)
    def get_labels(self):
        return [s[1] for d in self.datasets for s in d.samples]


def make_sampler(labels):
    counts  = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

# ── Models ─────────────────────────────────────────────────────────────────────
class DriveSafeNet(nn.Module):
    """EfficientNet-B4 with custom head + Grad-CAM hooks."""
    def __init__(self, num_classes=10, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=pretrained, num_classes=0, global_pool="avg")
        d = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(dropout),
            nn.Linear(d, 512), nn.SiLU(),
            nn.Dropout(dropout/2), nn.Linear(512, num_classes),
        )
        self.gradients = None; self.activations = None
        self.backbone.blocks[-1].register_forward_hook(self._save_act)
        self.backbone.blocks[-1].register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o): self.activations = o
    def _save_grad(self, m, gi, go): self.gradients = go[0]

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def unfreeze(self, stage):
        for p in self.backbone.parameters(): p.requires_grad = (stage == 2)
        if stage >= 1:
            for p in self.backbone.blocks[-2:].parameters(): p.requires_grad = True
        for p in self.classifier.parameters(): p.requires_grad = True


class MobileViTNet(nn.Module):
    """MobileViT-S — lightweight comparison model."""
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilevit_s", pretrained=pretrained, num_classes=num_classes)
    def forward(self, x): return self.backbone(x)

# ── Loss ───────────────────────────────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, classes=10, smoothing=0.1):
        super().__init__(); self.s = smoothing; self.c = classes
    def forward(self, pred, target):
        oh = torch.full_like(pred, self.s/(self.c-1))
        oh.scatter_(1, target.unsqueeze(1), 1.0-self.s)
        return -(oh * F.log_softmax(pred, 1)).sum(1).mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0): super().__init__(); self.g = gamma
    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction="none")
        return ((1-torch.exp(-ce))**self.g * ce).mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__(); self.ls = LabelSmoothingCE(); self.fl = FocalLoss()
    def forward(self, p, t): return 0.5*self.ls(p,t) + 0.5*self.fl(p,t)

# ── Train/Val loops ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, crit, scaler):
    model.train(); loss_sum = correct = n = 0
    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        with autocast():
            out  = model(imgs); loss = crit(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        loss_sum += loss.item()*len(imgs); correct += (out.argmax(1)==labels).sum().item(); n += len(imgs)
    return loss_sum/n, correct/n

@torch.no_grad()
def val_epoch(model, loader, crit):
    model.eval(); loss_sum = correct = n = 0
    preds_all, labels_all, probs_all = [], [], []
    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs); loss = crit(out, labels)
        prob = F.softmax(out, 1)
        loss_sum += loss.item()*len(imgs); correct += (out.argmax(1)==labels).sum().item(); n += len(imgs)
        preds_all.extend(out.argmax(1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        probs_all.extend(prob.cpu().numpy())
    return loss_sum/n, correct/n, np.array(preds_all), np.array(labels_all), np.array(probs_all)

# ── Main training function ──────────────────────────────────────────────────────
def train(model, name, train_dl, val_dl, epochs=30, lr=3e-4, patience=7):
    crit  = CombinedLoss()
    scaler = GradScaler()
    sched  = CosineAnnealingWarmRestarts
    opt    = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=epochs//3)
    hist  = {"tl":[],"vl":[],"ta":[],"va":[]}
    best  = 0; patience_cnt = 0
    best_preds = best_labels = best_probs = None

    for epoch in range(epochs):
        if hasattr(model, "unfreeze"):
            if epoch == 0:
                model.unfreeze(0); print("  [FT] Stage 1: head only")
                opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
            elif epoch == epochs//3:
                model.unfreeze(1); print("  [FT] Stage 2: top-2 blocks")
                opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr*0.1, weight_decay=1e-4)
            elif epoch == (2*epochs)//3:
                model.unfreeze(2); print("  [FT] Stage 3: full backbone")
                opt = torch.optim.AdamW(model.parameters(), lr=lr*0.01, weight_decay=1e-4)

        t0 = time.time()
        tl, ta = train_epoch(model, train_dl, opt, crit, scaler)
        vl, va, preds, labels, probs = val_epoch(model, val_dl, crit)
        scheduler.step()

        hist["tl"].append(tl); hist["vl"].append(vl)
        hist["ta"].append(ta); hist["va"].append(va)

        flag = ""
        if va > best:
            best = va; patience_cnt = 0; flag = "  ✓"
            torch.save({"model":model.state_dict(),"val_acc":va,"epoch":epoch},
                       f"checkpoints/best_{name}.pt")
            best_preds, best_labels, best_probs = preds, labels, probs
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stop at epoch {epoch+1}"); break

        print(f"  Ep {epoch+1:02d}/{epochs} | TrL={tl:.4f} TrA={ta*100:.2f}% | "
              f"VlL={vl:.4f} VlA={va*100:.2f}% | {time.time()-t0:.1f}s{flag}")

    print(f"\n  ✅ {name} best val acc = {best*100:.2f}%")
    return hist, best_preds, best_labels, best_probs

# ── Main entry ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sf_root",    default="data/state_farm")
    p.add_argument("--auc_root",   default="data/auc_dataset")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=3e-4)
    args = p.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("\nLoading datasets...")
    sf_train = StateFarmDataset(args.sf_root, "train", get_train_transforms())
    sf_val   = StateFarmDataset(args.sf_root, "val",   get_val_transforms())
    auc_ds   = AUCDataset(args.auc_root, get_train_transforms())
    fused    = FusedDataset(sf_train, auc_ds)
    sampler  = make_sampler(fused.get_labels())

    train_dl = DataLoader(fused,   batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(sf_val,  batch_size=args.batch_size, shuffle=False,   num_workers=4, pin_memory=True)

    for model_cls, mname, ep in [
        (DriveSafeNet,  "efficientnet_b4", args.epochs),
        (MobileViTNet,  "mobilevit_s",     min(args.epochs, 28)),
    ]:
        print(f"\n{'='*55}\n  Training {mname}\n{'='*55}")
        model = model_cls(NUM_CLASSES, pretrained=True).to(DEVICE)
        hist, preds, labels, probs = train(model, mname, train_dl, val_dl, ep, args.lr)

        f1  = f1_score(labels, preds, average="weighted")
        try:
            lb  = label_binarize(labels, classes=list(range(NUM_CLASSES)))
            auc = roc_auc_score(lb, probs, multi_class="ovr", average="macro")
        except: auc = 0.0

        print(f"\n  F1={f1:.4f}  AUC={auc:.4f}")
        print(classification_report(labels, preds, target_names=CLASS_NAMES))

        with open(f"results/metrics_{mname}.json","w") as f:
            json.dump({"val_acc":max(hist['va']),"f1":f1,"auc":auc}, f, indent=2)
