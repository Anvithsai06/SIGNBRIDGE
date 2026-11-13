"""Validation-selected, CPU/MPS-friendly temporal transfer-learning experiments."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.mobilenet_temporal import MobileNetTemporal
from preprocessing.video_preprocessor import VideoPreprocessor


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/annotations/include50_valid_candidates.csv"
DATA_ROOT = ROOT / "datasets/raw/include50"
CACHE = ROOT / "datasets/processed/include50_mobilenetv3_8f160_embeddings.pt"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
EXPERIMENT = "EXP_001_mobilenetv3_bigru_attention"
SEED = 20261112


def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def macro_metrics(target, pred, n_classes=50):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (target, pred), 1)
    tp = np.diag(matrix); support = matrix.sum(1); predicted = matrix.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(n_classes, float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(n_classes, float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n_classes, float), where=(precision + recall) > 0)
    weights = support / max(support.sum(), 1)
    return {"accuracy": float(tp.sum() / max(support.sum(), 1)), "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()), "weighted_f1": float((f1 * weights).sum()), "matrix": matrix, "per_class_accuracy": np.divide(tp, support, out=np.full(n_classes, np.nan), where=support > 0)}


class FeatureDataset(Dataset):
    def __init__(self, features, labels, indices, train=False): self.features, self.labels, self.indices, self.train = features, labels, indices, train
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        x = self.features[self.indices[i]].float()
        if self.train:
            x = x + torch.randn_like(x) * 0.015
            if random.random() < 0.25: x = x * random.uniform(0.92, 1.08)
        return x, self.labels[self.indices[i]]


def build_cache(classes, force=False, start=None, end=None):
    """Extract in resumable shards because source clips are high-resolution."""
    manifest = pd.read_csv(MANIFEST)
    if start is not None:
        end = min(end or len(manifest), len(manifest))
        shard = CACHE.with_name(f"{CACHE.stem}_{start:04d}_{end:04d}.pt")
        if shard.exists() and not force:
            return torch.load(shard, map_location="cpu", weights_only=False)
        dev = device(); model = MobileNetTemporal(len(classes), pretrained=True).to(dev).eval()
        processor = VideoPreprocessor(num_frames=8, image_size=160); vectors = []
        with torch.inference_mode():
            for i in range(start, end):
                frames = processor.load_video(str(DATA_ROOT / manifest.iloc[i].video_path))
                video = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).to(dev)
                vectors.append(model.encode_frames(video).squeeze(0).cpu().half())
        result = {"start": start, "end": end, "features": torch.stack(vectors), "paths": manifest.iloc[start:end].video_path.tolist(), "classes": classes, "frames": 8, "image_size": 160}
        CACHE.parent.mkdir(parents=True, exist_ok=True); torch.save(result, shard); print(f"saved {shard.name}")
        return result
    if CACHE.exists() and not force:
        cached = torch.load(CACHE, map_location="cpu", weights_only=False)
        if cached.get("classes") == classes and len(cached["features"]) == len(cached["paths"]): return cached
    dev = device(); model = MobileNetTemporal(len(classes), pretrained=True).to(dev).eval()
    for p in model.cnn.parameters(): p.requires_grad = False
    processor = VideoPreprocessor(num_frames=8, image_size=160)
    vectors = []
    started = time.time()
    with torch.inference_mode():
        for i, row in manifest.iterrows():
            frames = processor.load_video(str(DATA_ROOT / row.video_path))
            video = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).to(dev)
            vectors.append(model.encode_frames(video).squeeze(0).cpu().half())
            if (i + 1) % 50 == 0: print(f"cached {i + 1}/{len(manifest)} videos ({(time.time()-started)/60:.1f} min)", flush=True)
    cache = {"features": torch.stack(vectors), "paths": manifest.video_path.tolist(), "classes": classes, "frames": 8, "image_size": 160}
    CACHE.parent.mkdir(parents=True, exist_ok=True); torch.save(cache, CACHE)
    return cache


def evaluate(model, loader, dev):
    model.eval(); targets=[]; predictions=[]; probabilities=[]; loss_sum=0.0; criterion=nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x, y in loader:
            logits=model.forward_features(x.to(dev)); loss_sum += criterion(logits, y.to(dev)).item() * len(y)
            probs=torch.softmax(logits, 1).cpu(); targets.extend(y.tolist()); predictions.extend(probs.argmax(1).tolist()); probabilities.extend(probs.tolist())
    metrics=macro_metrics(np.array(targets), np.array(predictions), len(model.classifier.bias)); metrics.update(loss=loss_sum/len(loader.dataset), targets=targets, predictions=predictions, probabilities=probabilities)
    return metrics


def plot_history(history, out):
    e=[h["epoch"] for h in history]; plt.figure(figsize=(8,4)); plt.plot(e,[h["train_loss"] for h in history],label="train"); plt.plot(e,[h["val_loss"] for h in history],label="validation"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("cross-entropy"); plt.tight_layout(); plt.savefig(out.with_name(out.stem + "_loss.png"), dpi=160); plt.close()
    plt.figure(figsize=(8,4)); plt.plot(e,[h["train_accuracy"] for h in history],label="train"); plt.plot(e,[h["val_accuracy"] for h in history],label="validation"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("top-1 accuracy"); plt.tight_layout(); plt.savefig(out.with_name(out.stem + "_accuracy.png"), dpi=160); plt.close()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--epochs",type=int,default=100); parser.add_argument("--batch-size",type=int,default=32); parser.add_argument("--rebuild-cache",action="store_true"); parser.add_argument("--cache-start",type=int); parser.add_argument("--cache-end",type=int); args=parser.parse_args()
    seed_everything(); RESULTS.mkdir(exist_ok=True); CHECKPOINTS.mkdir(exist_ok=True)
    manifest=pd.read_csv(MANIFEST); classes=sorted(manifest.label.unique()); mapping={c:i for i,c in enumerate(classes)}
    if args.cache_start is not None:
        build_cache(classes,args.rebuild_cache,args.cache_start,args.cache_end)
        return
    if not CACHE.exists():
        shards=sorted(CACHE.parent.glob(f"{CACHE.stem}_*.pt"))
        if not shards: raise FileNotFoundError("Build cache shards first with --cache-start/--cache-end.")
        parts=[torch.load(p,map_location="cpu",weights_only=False) for p in shards]
        if sum(len(x["paths"]) for x in parts) != len(manifest): raise ValueError("Incomplete feature cache shards.")
        cache={"features":torch.cat([x["features"] for x in parts]),"paths":sum([x["paths"] for x in parts],[]),"classes":classes,"frames":8,"image_size":160}
        if cache["paths"] != manifest.video_path.tolist(): raise ValueError("Cache shards do not match manifest order.")
        torch.save(cache,CACHE)
    else: cache=build_cache(classes,args.rebuild_cache)
    labels=torch.tensor([mapping[x] for x in manifest.label])
    splits={name: np.flatnonzero((manifest.dataset_split == name).to_numpy()) for name in ("train","val","test")}
    train_ds=FeatureDataset(cache["features"],labels,splits["train"],True); val_ds=FeatureDataset(cache["features"],labels,splits["val"])
    counts=np.bincount(labels[splits["train"]].numpy(),minlength=len(classes)); weights=torch.tensor([1/counts[labels[i]] for i in splits["train"]],dtype=torch.double)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,sampler=WeightedRandomSampler(weights,len(weights),replacement=True)); val_loader=DataLoader(val_ds,batch_size=args.batch_size)
    dev=device(); model=MobileNetTemporal(len(classes),pretrained=False).to(dev); model.cnn.load_state_dict(MobileNetTemporal(len(classes),pretrained=True).cnn.state_dict())
    for p in model.cnn.parameters(): p.requires_grad=False
    criterion=nn.CrossEntropyLoss(label_smoothing=0.08); optimizer=AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=2e-4); scheduler=ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=8)
    best=-1.; wait=0; history=[]; checkpoint=CHECKPOINTS / f"{EXPERIMENT}_best.pth"; started=time.time()
    for epoch in range(1,args.epochs+1):
        model.train(); total=correct=0; loss_sum=0.
        for x,y in train_loader:
            optimizer.zero_grad(set_to_none=True); logits=model.forward_features(x.to(dev)); loss=criterion(logits,y.to(dev)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            total+=len(y); loss_sum+=loss.item()*len(y); correct+=(logits.argmax(1).cpu()==y).sum().item()
        val=evaluate(model,val_loader,dev); scheduler.step(val["macro_f1"])
        row={"epoch":epoch,"train_loss":loss_sum/total,"train_accuracy":correct/total,"val_loss":val["loss"],"val_accuracy":val["accuracy"],"val_macro_f1":val["macro_f1"],"learning_rate":optimizer.param_groups[0]["lr"]}; history.append(row)
        print(json.dumps(row),flush=True)
        if val["macro_f1"] > best + 1e-6:
            best=val["macro_f1"]; wait=0; torch.save({"architecture":"MobileNetTemporal","experiment_id":EXPERIMENT,"model_state_dict":model.state_dict(),"classes":classes,"num_frames":8,"image_size":160,"preprocessing":{"resize":"letterbox","rgb":True,"scale":"[0,1]","normalization":"ImageNet"},"seed":SEED,"epoch":epoch,"validation_metrics":{k:v for k,v in val.items() if k not in {"matrix","per_class_accuracy","targets","predictions","probabilities"}}},checkpoint)
        else: wait+=1
        if wait>=20: print("early stopping",flush=True); break
    history_path=RESULTS/f"{EXPERIMENT}_history.csv"; pd.DataFrame(history).to_csv(history_path,index=False); plot_history(history,history_path)
    print(json.dumps({"best_val_macro_f1":best,"epochs":len(history),"seconds":round(time.time()-started,1),"checkpoint":str(checkpoint)},indent=2))


if __name__ == "__main__": main()
