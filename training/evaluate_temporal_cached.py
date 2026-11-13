"""One-shot exploratory test evaluation for the validation-selected temporal model."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.mobilenet_temporal import MobileNetTemporal
from training.train_temporal_cached import CACHE, EXPERIMENT, MANIFEST, RESULTS, device, evaluate, FeatureDataset


def main():
    manifest = pd.read_csv(MANIFEST); checkpoint_path = Path("checkpoints") / f"{EXPERIMENT}_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False); classes = checkpoint["classes"]
    cache = torch.load(CACHE, map_location="cpu", weights_only=False); mapping = {name:i for i,name in enumerate(classes)}
    labels = torch.tensor([mapping[x] for x in manifest.label]); indices=np.flatnonzero((manifest.dataset_split == "test").to_numpy())
    model=MobileNetTemporal(len(classes),pretrained=False).to(device()); model.load_state_dict(checkpoint["model_state_dict"])
    metrics=evaluate(model,DataLoader(FeatureDataset(cache["features"],labels,indices),batch_size=32),device())
    probs=np.array(metrics.pop("probabilities")); targets=np.array(metrics.pop("targets")); predictions=np.array(metrics.pop("predictions")); matrix=metrics.pop("matrix"); per_class=metrics.pop("per_class_accuracy")
    top5=(np.argpartition(probs,-5,axis=1)[:,-5:] == targets[:,None]).any(1).mean()
    conf=probs.max(1); correct=(predictions==targets); bins=np.linspace(0,1,11); ece=0.0
    for i in range(10):
        mask=(conf>=bins[i]) & (conf < (bins[i+1] if i < 9 else 1.01))
        if mask.any(): ece += abs(correct[mask].mean()-conf[mask].mean()) * mask.mean()
    onehot=np.eye(len(classes))[targets]; brier=float(np.mean(np.sum((probs-onehot)**2,axis=1)))
    metrics.update(top5_accuracy=float(top5), ece=float(ece), brier_score=brier, test_samples=len(targets), evaluated_test_classes=int(np.unique(targets).size), missing_test_classes=[classes[i] for i in range(len(classes)) if not np.any(targets==i)])
    prefix=RESULTS / EXPERIMENT
    pd.DataFrame({"video_path":manifest.iloc[indices].video_path,"actual_sign":[classes[i] for i in targets],"predicted_sign":[classes[i] for i in predictions],"confidence":conf,"correct":correct}).to_csv(prefix.with_name(prefix.name+"_test_predictions.csv"),index=False)
    pd.DataFrame({"class":classes,"test_support":np.bincount(targets,minlength=len(classes)),"test_accuracy":per_class}).to_csv(prefix.with_name(prefix.name+"_per_class.csv"),index=False)
    pd.DataFrame(matrix,index=classes,columns=classes).to_csv(prefix.with_name(prefix.name+"_confusion_matrix.csv"))
    plt.figure(figsize=(14,12)); plt.imshow(matrix,interpolation="nearest",cmap="Blues"); plt.colorbar(); plt.xticks([]); plt.yticks([]); plt.title("Exploratory test confusion matrix"); plt.tight_layout(); plt.savefig(prefix.with_name(prefix.name+"_confusion_matrix.png"),dpi=180); plt.close()
    plt.figure(figsize=(5,5)); xs=[]; ys=[]
    for i in range(10):
        mask=(conf>=bins[i])&(conf<(bins[i+1] if i<9 else 1.01))
        if mask.any(): xs.append(conf[mask].mean()); ys.append(correct[mask].mean())
    plt.plot([0,1],[0,1],"--",color="gray"); plt.plot(xs,ys,"o-"); plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.title("Reliability (uncalibrated)"); plt.tight_layout(); plt.savefig(prefix.with_name(prefix.name+"_reliability.png"),dpi=180); plt.close()
    (prefix.with_name(prefix.name+"_test_metrics.json")).write_text(json.dumps(metrics,indent=2)+"\n")
    torch.save(checkpoint,Path("checkpoints/signbridge_final.pth"))
    print(json.dumps(metrics,indent=2))

if __name__ == "__main__": main()
