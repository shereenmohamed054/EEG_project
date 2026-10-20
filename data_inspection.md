# ZuCo 1.0 & ZuCo 2.0 Dataset Inspection Report

## 1. Executive Summary
- **Total Recorded Trials/Sentences**: 36
- **Datasets Present**: ZuCo1, ZuCo2
- **Total Unique Subjects**: 4 (ZuCo 1: 2, ZuCo 2: 2)
- **Tasks Identified**: NR, SR, TSR
- **Sampling Rate(s)**: 500 Hz
- **Channel Counts Observed**: 105 channels
- **Sentence Duration (sec)**: Mean = 3.42s, Median = 3.40s, Min = 3.00s, Max = 3.80s
- **Mean Word Count per Sentence**: 8.1 words

## 2. Dataset Breakdown by Subject & Task

| Dataset | Subject | Task | Sentence Count | Channels | Mean Duration (s) | Has Raw EEG | Word Fixations |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ZuCo1 | ZAB | NR | 5 | 105 | 3.40s | True | True |
| ZuCo1 | ZAB | SR | 5 | 105 | 3.40s | True | True |
| ZuCo1 | ZDM | NR | 5 | 105 | 3.40s | True | True |
| ZuCo1 | ZDM | SR | 5 | 105 | 3.40s | True | True |
| ZuCo2 | YAC | NR | 4 | 105 | 3.44s | True | True |
| ZuCo2 | YAC | TSR | 4 | 105 | 3.44s | True | True |
| ZuCo2 | YAG | NR | 4 | 105 | 3.44s | True | True |
| ZuCo2 | YAG | TSR | 4 | 105 | 3.44s | True | True |

## 3. Structural Comparison: ZuCo 1.0 vs. ZuCo 2.0

| Dimension | ZuCo 1.0 | ZuCo 2.0 | Reconciled Solution |
|:---|:---|:---|:---|
| **Subject Cohort** | 2 subjects (ZAB, ZDM...) | 2 subjects (YAC, YAG...) | Disjoint subject splits for within-dataset and cross-dataset testing |
| **Tasks** | NR, SR | NR, TSR | Focus cross-dataset evaluation on overlapping tasks (**NR** & **TSR**); evaluate **SR** for sentiment decoding |
| **Electrode Montage** | 128-channel Geodesic (105 scalp channels standard) | 128-channel Geodesic (105 scalp channels standard) | Harmonize to the exact **105 common scalp channels** (or standardized 10-20 subset), re-referenced to average |
| **Sampling Rate** | 500 Hz | 500 Hz | Resample both to **200 Hz** (or 250 Hz for foundation models: LaBraM, BIOT, CBraMod) |
| **Sequence Length** | Variable (reading speed dependent) | Variable (reading speed dependent) | Fixed-window chunking / robust padding with boolean attention masks |
| **Word Fixations & Eye-Tracking** | Detailed `word` structs with FFD, TRT, GD | Detailed `word` structs with FFD, TRT, GD | Retained for token-level EEG representation & confound analysis |

## 4. Key Preprocessing & Reconciliation Pipeline
1. **Channel Alignment**: Extract common 105 scalp channels across all ZuCo 1 and ZuCo 2 recordings, ordered by standard 105 Geodesic montage.
2. **Resampling**: Anti-aliasing low-pass filter followed by decimation to 200 Hz (reducing compute while capturing all canonical delta, theta, alpha, beta, and gamma bands up to 50 Hz).
3. **Filtering**: 0.5 - 50 Hz bandpass filter (zero-phase Butterworth) and 50 Hz notch filter to eliminate line noise.
4. **Normalization**: Per-channel robust z-score standardization computed per trial (to prevent inter-subject amplitude drift).
5. **Artifact & Quality Thresholds**: Flag flat channels (std < 1e-6 uV) and extreme variance (> 5x channel median).
6. **Unified Storage**: Save processed shards as `.pt` / `.npy` tensors accompanied by a master manifest index (`manifest.csv`), enabling memory-mapped streaming without RAM saturation.

## 5. Next Steps
- Save unified manifest index to `data/zuco_unified_index.csv`.
- Proceed to **Step 1**: Construct deterministic subject-independent, sentence-held-out, and cross-dataset splits.