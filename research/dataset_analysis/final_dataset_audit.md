# INCLUDE-50 local dataset audit — 2026-11-10

## Verified local inventory

| Item | Result |
| --- | ---: |
| Videos / labels | 929 / 50 |
| Split counts (train / val / test) | 662 / 77 / 190 |
| Missing files | 0 |
| Cannot open or decode first frame | 0 |
| Duplicate annotation paths / conflicting path labels | 0 / 0 |
| Cross-split identical paths | 0 |
| Byte-identical video groups (SHA-256) | 0 |
| Resolution / frame rate | 1920×1080 / 25 fps (all videos) |
| Duration, median (range) | 2.40 s (1.44–6.16 s) |

The prior complete decode audit (`datasets/annotations/downloaded_dataset_audit.csv`) also reports all 929 videos valid. This audit additionally verifies SHA-256 identity and a first-frame decode against the current local data.

Reproduce with:

```bash
python3 research/dataset_analysis/audit_include50.py
```

Outputs: `include50_local_audit.csv`, `include50_local_audit_summary.json`, and `include50_split_class_counts.csv`.

## Distribution and split limitations

- Classes contain 14–21 total samples (median 20); training counts range 7–19 (median ~13).
- `19. House` and `97. dry` have **zero** test examples. Thus the 190-video split cannot report complete-50-class accuracy.
- Validation has only 1–2 examples per class, which makes early stopping and model choice highly variable.
- No signer field exists in the source manifest. File names are not a defensible signer proxy; consequently, no signer-independent claim is possible.
- The source metadata previously contained 14 conflicting-label paths and one exact duplicate, all excluded in the curated 929-row candidate manifest. The retained manifest itself has none.
- Exact-file checks do not rule out near-duplicates or recordings of the same signer/session. Do not describe the split as genuinely unseen-signers.

## Preprocessing parity

Offline preprocessing uniformly samples 16 frames, converts BGR→RGB, letterboxes to 224² with RGB value 114, then scales to `[0,1]`. The models apply ImageNet normalization. Training augmentation is only applied to train and is temporally consistent across frames. Evaluation has no augmentation.

The backend performs the same RGB letterbox and scaling, and model normalization is shared in the model. Its API does not sample video frames; it demands exactly 16 browser frames. A later preprocessing module should own both video decoding and live-capture sampling to eliminate this operational mismatch.

## Recommendation

INCLUDE-50 is adequate for pipeline debugging and a carefully disclosed exploratory baseline. It is not adequate evidence for a 70% genuinely unseen-signer, complete-50-class claim without stronger metadata and more examples. Preserve it; do not silently discard hard data. Create a replacement locked test only after adding more compatible data and collecting/obtaining signer IDs.
