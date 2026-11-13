# Experiment log

| ID | Status | Model / data | Selection metric | Actual recorded outcome |
| --- | --- | --- | --- | --- |
| LEGACY_CNN_LSTM | exploratory | MobileNetV3-Small + LSTM, current 929-video split | validation loss | Existing letterbox checkpoint: epoch 10, val loss 2.2299; recorded test top-1 10.00%. |
| LEGACY_MOBILENET | exploratory | MobileNetV3-Small temporal mean pooling, current split | validation loss | Existing letterbox checkpoint: epoch 10, val loss 2.3259; recorded test top-1 11.05%, macro F1 0.0876. |

Neither run is eligible for final model selection: test was reused, two labels have no test examples, and signer independence is unverified. No new training was run during this audit.
