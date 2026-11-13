# Experiment log

| ID | Status | Model / data | Selection metric | Actual recorded outcome |
| --- | --- | --- | --- | --- |
| LEGACY_CNN_LSTM | exploratory | MobileNetV3-Small + LSTM, current 929-video split | validation loss | Existing letterbox checkpoint: epoch 10, val loss 2.2299; recorded test top-1 10.00%. |
| LEGACY_MOBILENET | exploratory | MobileNetV3-Small temporal mean pooling, current split | validation loss | Existing letterbox checkpoint: epoch 10, val loss 2.3259; recorded test top-1 11.05%, macro F1 0.0876. |

Neither run is eligible for final model selection: test was reused, two labels have no test examples, and signer independence is unverified. No new training was run during this audit.

| EXP_001_mobilenetv3_bigru_attention | completed | Frozen ImageNet MobileNetV3 frame encoder (8 uniformly sampled, 160px letterboxed RGB frames) + bidirectional GRU with temporal attention; balanced sampling, AdamW, label smoothing, gradient clipping | validation macro F1 | Best epoch 66/86: validation top-1 93.51%, macro F1 0.9020. One exploratory test run: top-1 83.68%, top-5 96.32%, macro F1 0.8066. |

The model was selected solely by validation macro F1. Feature extraction was cached from all files without labels; the temporal head saw only training examples during optimization. The test result is exploratory because that split was used by legacy work and lacks two classes.
