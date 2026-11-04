SIGNBRIDGE – Dataset Audit

1. Purpose

This document records the datasets investigated for developing and evaluating the SIGNBRIDGE Indian Sign Language recognition and translation system.

The objective is to identify suitable datasets for isolated sign recognition, continuous sign recognition and sign-to-text translation.

2. Dataset Shortlist

Dataset	        Primary Task	                Status
INCLUDE	        Isolated sign recognition	    To investigate

Amrita SLR	    Real-time sign recognition	    To investigate

ISL-CSLTR	    Continuous sign recognition 	To investigate
                and translation

ISLTranslate	Sign-to-English translation	    To investigate

iSign	        ISL processing benchmark	    To investigate

ISH-NEWS	    Sentence-level translation  	To investigate
                research

3. Dataset Evaluation Criteria

Each dataset will be evaluated using the following criteria:

1. Availability and accessibility.
2. Dataset licence and permitted usage.
3. Number of samples.
4. Number of sign classes.
5. Number of signers.
6. Video resolution and frame rate.
7. Annotation format.
8. Availability of sentence-level labels.
9. Availability of signer information.
10. Suitability for real-time recognition research.

4. Initial Dataset Selection Strategy

Stage 1 – Isolated Sign Recognition

Investigate INCLUDE and Amrita SLR.

Stage 2 – Temporal Recognition

Investigate video sequences and annotations in ISL-CSLTR.

Stage 3 – Continuous Translation

Investigate ISLTranslate and iSign.

Stage 4 – Advanced Research

Investigate ISH-NEWS and its available research resources.

5. Research Considerations

* Avoid data leakage between training and testing.
* Prefer signer-independent evaluation.
* Preserve original video data.
* Maintain consistent preprocessing.
* Record dataset versions and licences.
* Do not combine incompatible annotations.
* Document dataset limitations.

6. Current Status

Project environment: Ready.

Dataset investigation: Pending.

Model training: Not started.

7. Next Milestone

Download and inspect the selected initial dataset.

Understand its directory structure, video format, annotations and sample distribution before implementing the preprocessing pipeline.