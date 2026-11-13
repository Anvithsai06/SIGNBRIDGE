# Model comparison

Existing exploratory results on the reused, incomplete 190-video test split:

| Model | Top-1 | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| CNN-LSTM (letterbox) | 10.00% | 0.0707 | 0.0702 |
| MobileNetV3 frame-average (letterbox) | 11.05% | 0.0876 | 0.0860 |

These numbers are copied from `results/model_comparison_letterbox.csv`; they are not final or comparable to a genuine all-50-class unseen-signer result. Further experiments must select on validation only and reserve a newly constructed locked test set.

| Model | Validation top-1 / macro F1 | Exploratory test top-1 / macro F1 | Top-5 |
| --- | ---: | ---: | ---: |
| EXP_001 MobileNetV3 + BiGRU attention | 93.51% / 0.9020 | 83.68% / 0.8066 | 96.32% |

EXP_001 is a substantial improvement over the legacy baselines, but its test estimate is not a valid 50-class unseen-signer benchmark: it has 190 samples from 48 labels and no signer metadata.
