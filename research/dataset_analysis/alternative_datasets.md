# ISL dataset research — 2026-11-10

No dataset was downloaded. “Overlap” below means it needs an explicit, human-reviewed mapping to the current 50 labels; it is not assumed from English glosses.

| Dataset | Task / scale | Access and storage | Fit for this project |
| --- | --- | --- | --- |
| [INCLUDE full corpus](https://huggingface.co/datasets/ai4bharat/INCLUDE) | ISL isolated video; 263 signs / about 4.3k videos; 7 signers | HF hosts metadata; videos are retrieved from Zenodo. Local 929 videos are 12 GiB, so raw full-corpus storage is roughly 55 GiB if the local average is representative; verify Zenodo archive sizes before download. | Best low-risk expansion: same source and likely label semantics. Need signer-aware official split and an explicit 50-label filter. |
| [Amrita SLR](https://data.mendeley.com/datasets/dtg9s785v8/1) | ISL isolated; exactly 50 signs; ~16 samples/class; 7 signers, multi-device/environment | Inspect file list/licence and vocabulary before requesting/download. Estimate unknown. | Useful external robustness set only if labels align. It does not by itself add enough samples; 50 signs may be a different vocabulary. |
| [FDMSE-ISL / HWGAT corpus](https://arxiv.org/abs/2407.14224) | Isolated ISL; 2,002 words / 40,033 videos; 20 deaf adult signers; RGB multi-view plus depth | The paper describes the corpus but this audit did not establish a public download or licence. Storage likely large (tens to hundreds of GiB); request access and terms first. | Highest potential for pose/RGB pretraining, then fine-tune 50 verified overlapping signs. Do not assume permission or mappings. |
| [ISL dictionary re-encode](https://huggingface.co/datasets/silentone0725/Indian_Sign_Language_Data.gov_Rencoded) | Thousands of isolated dictionary clips, derived from ISLRTC/data.gov.in | Publisher reports ~75 GB re-encoded (original ~200 GB). Verify the original portal licence and whether the derivative is permitted for the intended use. | Broad vocabulary but likely exemplar-style rather than multi-signer repetitions; useful only after licence and label/quality review. |
| [iSign](https://huggingface.co/datasets/Exploration-Lab/iSign) | ISL translation benchmark with video and pose resources | Research-only, non-commercial; multi-part archives. Not isolated-sign classification metadata. | Not a direct replacement; keep separate for future continuous/translation work. |

## Recommendation

1. Retain INCLUDE-50 for a reproducible local baseline, but do not claim signer-independent performance.
2. First inspect full INCLUDE metadata and licence without bulk download. If it exposes signer/session information and enough of the current 50 signs, use it to create a signer-disjoint development/locked-test protocol.
3. In parallel, request FDMSE-ISL access/licence. Its scale and 20 signers make it the strongest pretraining candidate, but this is contingent on access and expert-reviewed label overlap.
4. Treat Amrita as an external evaluation source, not automatic training data. Map only exact ISL concepts confirmed by a fluent signer/annotator.

The INCLUDE metadata page identifies Zenodo as the video source; Amrita documents 1920×1080/28 fps clips, diverse conditions, and signer variation. The FDMSE paper reports its scale, 20 signers, and pose-based transfer gains. These are source claims, not reproduced project results.
