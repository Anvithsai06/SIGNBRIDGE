INCLUDE Dataset – Research Notes

1. Dataset Overview

INCLUDE is an Indian Sign Language dataset developed for isolated sign language recognition.

It will be investigated as the initial dataset for SIGNBRIDGE.

Dataset source: AI4Bharat
Hugging Face: https://huggingface.co/datasets/ai4bharat/INCLUDE
Zenodo: https://zenodo.org/records/4010759

2. Dataset Components

The dataset is distributed through two sources:

* Hugging Face: Metadata, labels, video paths and dataset splits.
* Zenodo: Actual video files.

The metadata and video files must be matched using the video paths.

3. Reported Dataset Statistics

Split	Number of Entries
Train	3,816
Validation	425
Test	1,009
Total	5,250

The dataset also identifies an INCLUDE-50 subset.

These statistics must be verified against the downloaded metadata.

4. Dataset Inspection Requirements

We need to investigate:

1. Number of unique sign classes.
2. Number of videos per class.
3. Video resolution.
4. Frames per video.
5. Video duration.
6. Frame rate.
7. Missing or corrupted videos.
8. Signer information, if available.
9. Train, validation and test distribution.
10. Dataset licensing and usage restrictions.

5. Research Considerations

* Preserve original video files.
* Do not mix training and testing data.
* Investigate signer-independent evaluation.
* Identify class imbalance.
* Record preprocessing decisions.
* Respect the privacy of people appearing in the videos.
* Cite the original INCLUDE research paper.

6. SIGNBRIDGE Application

Initial objective:

Build an isolated Indian Sign Language recognition model using video data.

Planned experiments:

* CNN-based baseline.
* CNN + LSTM.
* CNN + GRU.
* Transformer-based temporal architecture.

Compare models using:

* Accuracy.
* Precision.
* Recall.
* F1-score.
* Inference time.

7. Current Status

Activity	Status
Project structure	Completed
Python environment	Completed
GPU verification	Completed
Dataset research	In progress
Metadata inspection	Pending
Video download	Pending
Preprocessing	Pending
Model training	Not started

8. Next Milestone

Download the INCLUDE metadata first.

Inspect its columns, labels, splits and video paths.

After confirming the dataset structure and archive size, download the required video files.

9. Storage Investigation

The Zenodo API was successfully accessed using Python with SSL certificate verification.

Record ID: 4010759

Full Dataset Archive Information

* Total files: 46
* ZIP archives: 44
* Additional files: 2
* Total compressed download size: 52.85 GB

Storage Decision

The complete dataset will not be downloaded immediately because of its substantial storage requirements.

Instead, the INCLUDE-50 subset will be investigated using the include_50 metadata column.

Next Investigation

1. Retrieve Hugging Face metadata.
2. Filter INCLUDE-50 records.
3. Count unique labels and samples.
4. Identify required video paths.
5. Map video paths to Zenodo ZIP archives.
6. Estimate the required storage.

Decision: Full dataset download postponed until subset feasibility is established.