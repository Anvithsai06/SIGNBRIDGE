# Final model report

`checkpoints/signbridge_final.pth` was created from `EXP_001_mobilenetv3_bigru_attention`, selected by validation macro F1 at epoch 66 (0.9020). It contains the 50-class mapping, MobileNetV3 + BiGRU-attention weights, seed, preprocessing configuration, and validation metrics.

## Exploratory evaluation

- Test top-1: **83.68%** (159/190)
- Test top-5: **96.32%**
- Macro precision / recall / F1: 0.8368 / 0.8314 / 0.8066
- Weighted F1: 0.8411
- ECE (uncalibrated): 0.1311; Brier score: 0.2653

This is an actual one-shot test evaluation after validation selection, not a claim of genuine complete-50 unseen-signer performance. The legacy test split has only 48 represented labels (`19. House` and `97. dry` are absent) and lacks signer IDs.

## Errors and limitations

31 test predictions were incorrect. The dominant confusion was `78. Year → 67. Monday` (5), followed by `1. Dog → 4. Bird`, `77. Boy → 78. Girl`, `79. short → 94. good`, and `55. Thank you → 51. Good Morning` (2 each). The weakest represented classes were `84. small little` (25%) and `78. Year` (28.6%).

The FastAPI model loader now recognizes the final architecture and reads its 8-frame/160px preprocessing contract. A real Good Morning video was correctly predicted at 83.56% top-1 confidence on CPU, with five predictions returned. Confidence remains raw softmax; do not interpret it as calibrated probability.

Generated evidence: `results/EXP_001_mobilenetv3_bigru_attention_history_{loss,accuracy}.png`, `..._confusion_matrix.{csv,png}`, `..._test_predictions.csv`, `..._per_class.csv`, and `..._reliability.png`.
