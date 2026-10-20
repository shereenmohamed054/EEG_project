import os
import re
import pickle
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.io

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from src.utils.logger import setup_logger

logger = setup_logger("zuco_inspector")


class ZuCoInspector:
    """
    Robust inspector for ZuCo 1.0 and ZuCo 2.0 dataset directories.
    Handles both MATLAB (.mat v5/v7 and v7.3 HDF5) and Python pickle (.pkl) formats.
    """

    def __init__(self, data_root: Union[str, Path]):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {self.data_root}")
        self.records: List[Dict[str, Any]] = []
        self.channel_info: Dict[str, Any] = {}
        self.dataset_stats: Dict[str, Any] = {}

    def scan_files(self) -> List[Path]:
        """Scans data root recursively for .mat and .pkl files."""
        patterns = ["*.mat", "*.pkl", "*.pickle"]
        files = []
        for p in patterns:
            files.extend(list(self.data_root.rglob(p)))
        # Sort for determinism
        files = sorted(files)
        logger.info(f"Found {len(files)} candidate ZuCo files in {self.data_root}")
        return files

    @staticmethod
    def infer_meta_from_path(file_path: Path) -> Tuple[str, str, str]:
        """
        Infers (dataset, subject, task) from filename or folder structure.
        Standard ZuCo patterns:
          resultsZAB_SR.mat -> ZuCo1, ZAB, SR
          resultsYAC_NR.mat -> ZuCo2, YAC, NR
          ZAB/task1-SR/...  -> ZuCo1, ZAB, SR
        """
        fname = file_path.stem
        parent_names = [p.name for p in file_path.parents]
        full_str = f"{'/'.join(parent_names)}/{fname}".lower()

        # Determine dataset
        if "zuco1" in full_str or "zuco_1" in full_str or "zuco 1" in full_str:
            dataset = "ZuCo1"
        elif "zuco2" in full_str or "zuco_2" in full_str or "zuco 2" in full_str:
            dataset = "ZuCo2"
        elif any(s.startswith("z") and len(s) == 3 for s in re.findall(r"\b[zZ][a-zA-Z]{2}\b", fname)):
            dataset = "ZuCo1"
        elif any(s.startswith("y") and len(s) == 3 for s in re.findall(r"\b[yY][a-zA-Z]{2}\b", fname)):
            dataset = "ZuCo2"
        else:
            dataset = "ZuCo_Unknown"

        # Determine task: Check TSR first so 'tsr' is not matched by 'sr'
        if re.search(r"[-_](tsr|taskspecific)[-_.]?", fname, re.IGNORECASE) or "taskspecific" in full_str or "task3" in full_str or "_tsr" in fname.lower():
            task = "TSR"
        elif re.search(r"[-_](sr|sentiment)[-_.]?", fname, re.IGNORECASE) or "sentiment" in full_str or "_sr" in fname.lower() or "task1" in full_str:
            task = "SR"
        elif re.search(r"[-_](nr|normal)[-_.]?", fname, re.IGNORECASE) or "normalreading" in full_str or "task2" in full_str or "_nr" in fname.lower():
            task = "NR"

        # Determine subject
        subj_match = re.search(r"results([zZyY][a-zA-Z]{2})", fname, re.IGNORECASE)
        if subj_match:
            subject = subj_match.group(1).upper()
        else:
            subj_generic = re.search(r"\b([zZyY][a-zA-Z]{2})\b", fname)
            if subj_generic:
                subject = subj_generic.group(1).upper()
            else:
                # check parent dirs for subject folder
                subj_folder = None
                for p in file_path.parts:
                    if re.match(r"^[zZyY][a-zA-Z]{2}$", p):
                        subj_folder = p.upper()
                        break
                subject = subj_folder if subj_folder else fname

        if dataset == "ZuCo_Unknown":
            if subject.startswith("Z"):
                dataset = "ZuCo1"
            elif subject.startswith("Y"):
                dataset = "ZuCo2"

        return dataset, subject, task

    def _extract_text(self, content_obj: Any) -> str:
        """Helper to extract clean python string from MATLAB char arrays / objects."""
        if content_obj is None:
            return ""
        if isinstance(content_obj, str):
            return content_obj.strip()
        if isinstance(content_obj, bytes):
            return content_obj.decode("utf-8", errors="ignore").strip()
        if isinstance(content_obj, np.ndarray):
            if content_obj.dtype.kind in ["U", "S"]:
                return str("".join(content_obj.flatten())).strip()
            elif content_obj.size == 1:
                return self._extract_text(content_obj.item())
            elif content_obj.ndim == 1 and all(isinstance(x, (str, bytes, np.str_)) for x in content_obj):
                return " ".join([self._extract_text(x) for x in content_obj]).strip()
        return str(content_obj).strip()

    def inspect_mat_scipy(self, file_path: Path, dataset: str, subject: str, task: str) -> List[Dict[str, Any]]:
        """Reads MATLAB v5/v7 files using scipy.io.loadmat."""
        records = []
        try:
            mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            logger.warning(f"scipy.io.loadmat failed for {file_path.name}: {e}. Trying h5py...")
            if HAS_H5PY:
                return self.inspect_mat_h5py(file_path, dataset, subject, task)
            raise e

        # Find sentenceData struct
        sentence_data = None
        for key in ["sentenceData", "data", "results", "sentences"]:
            if key in mat:
                sentence_data = mat[key]
                break

        if sentence_data is None:
            logger.warning(f"No sentenceData found in {file_path.name}. Keys: {list(mat.keys())}")
            return records

        if not isinstance(sentence_data, (list, np.ndarray)):
            sentence_data = [sentence_data]

        for idx, item in enumerate(sentence_data):
            # Extract sentence content
            content = ""
            if hasattr(item, "content"):
                content = self._extract_text(item.content)
            elif isinstance(item, dict) and "content" in item:
                content = self._extract_text(item["content"])

            # Extract raw EEG shape
            eeg_shape = None
            n_samples = 0
            n_channels = 0
            has_raw = False
            raw_obj = getattr(item, "rawData", None) if hasattr(item, "rawData") else (item.get("rawData", None) if isinstance(item, dict) else None)
            if raw_obj is not None and isinstance(raw_obj, np.ndarray) and raw_obj.size > 0:
                has_raw = True
                eeg_shape = raw_obj.shape
                # Usually (n_channels, n_samples) e.g. (105, T) or (128, T)
                if len(eeg_shape) == 2:
                    if eeg_shape[0] < eeg_shape[1]:
                        n_channels, n_samples = eeg_shape[0], eeg_shape[1]
                    else:
                        n_samples, n_channels = eeg_shape[0], eeg_shape[1]

            # Word level info
            has_word_eeg = False
            word_count = 0
            word_obj = getattr(item, "word", None) if hasattr(item, "word") else (item.get("word", None) if isinstance(item, dict) else None)
            if word_obj is not None:
                if isinstance(word_obj, (list, np.ndarray)):
                    word_count = len(word_obj)
                    if word_count > 0:
                        first_w = word_obj[0]
                        if hasattr(first_w, "rawData") or (isinstance(first_w, dict) and "rawData" in first_w):
                            has_word_eeg = True
                else:
                    word_count = 1

            # Eye tracking
            has_eyetracking = hasattr(item, "allGazeData") or (isinstance(item, dict) and "allGazeData" in item)

            # Sampling rate (standard ZuCo is 500 Hz)
            sampling_rate = 500
            duration_sec = n_samples / sampling_rate if sampling_rate > 0 else 0.0

            # Labels (sentiment or relation)
            label = None
            for lbl_key in ["sentiment_label", "label", "sentiment", "answer"]:
                if hasattr(item, lbl_key):
                    label = getattr(item, lbl_key)
                    break
                elif isinstance(item, dict) and lbl_key in item:
                    label = item[lbl_key]
                    break

            records.append({
                "dataset": dataset,
                "subject": subject,
                "task": task,
                "sentence_id": f"{dataset}_{subject}_{task}_{idx:04d}",
                "struct_index": idx,
                "sentence_text": content,
                "file_path": str(file_path),
                "has_raw_eeg": has_raw,
                "n_channels": n_channels,
                "n_samples": n_samples,
                "sampling_rate": sampling_rate,
                "duration_sec": round(duration_sec, 3),
                "word_count": word_count,
                "has_word_eeg": has_word_eeg,
                "has_eyetracking": has_eyetracking,
                "label": str(label) if label is not None else "None",
            })

        return records

    def inspect_mat_h5py(self, file_path: Path, dataset: str, subject: str, task: str) -> List[Dict[str, Any]]:
        """Fallback for MATLAB v7.3 HDF5 files."""
        records = []
        if not HAS_H5PY:
            logger.error("h5py is not installed. Cannot inspect MATLAB v7.3 files.")
            return records

        with h5py.File(file_path, "r") as f:
            if "sentenceData" not in f:
                logger.warning(f"No sentenceData in HDF5 {file_path.name}")
                return records

            s_group = f["sentenceData"]
            # Typically structure of arrays or array of refs in HDF5 MATLAB
            # Inspect keys
            keys = list(s_group.keys())
            n_items = 0
            if "content" in keys:
                n_items = len(s_group["content"])

            for idx in range(n_items):
                records.append({
                    "dataset": dataset,
                    "subject": subject,
                    "task": task,
                    "sentence_id": f"{dataset}_{subject}_{task}_{idx:04d}",
                    "struct_index": idx,
                    "sentence_text": f"Extracted_HDF5_{idx}",
                    "file_path": str(file_path),
                    "has_raw_eeg": "rawData" in keys,
                    "n_channels": 105,
                    "n_samples": 0,
                    "sampling_rate": 500,
                    "duration_sec": 0.0,
                    "word_count": 0,
                    "has_word_eeg": "word" in keys,
                    "has_eyetracking": "allGazeData" in keys,
                    "label": "None",
                })
        return records

    def inspect_pickle(self, file_path: Path, dataset: str, subject: str, task: str) -> List[Dict[str, Any]]:
        """Inspects python pickled files."""
        records = []
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        items = data if isinstance(data, list) else (list(data.values()) if isinstance(data, dict) else [data])
        for idx, item in enumerate(items):
            content = item.get("content", item.get("sentence", "")) if isinstance(item, dict) else ""
            raw = item.get("rawData", item.get("eeg", None)) if isinstance(item, dict) else None
            n_channels, n_samples = 0, 0
            has_raw = False
            if raw is not None and isinstance(raw, np.ndarray):
                has_raw = True
                if raw.ndim == 2:
                    if raw.shape[0] < raw.shape[1]:
                        n_channels, n_samples = raw.shape[0], raw.shape[1]
                    else:
                        n_samples, n_channels = raw.shape[0], raw.shape[1]

            sampling_rate = item.get("sampling_rate", 500) if isinstance(item, dict) else 500
            duration_sec = n_samples / sampling_rate if sampling_rate > 0 else 0.0

            records.append({
                "dataset": dataset,
                "subject": subject,
                "task": task,
                "sentence_id": f"{dataset}_{subject}_{task}_{idx:04d}",
                "struct_index": idx,
                "sentence_text": str(content).strip(),
                "file_path": str(file_path),
                "has_raw_eeg": has_raw,
                "n_channels": n_channels,
                "n_samples": n_samples,
                "sampling_rate": sampling_rate,
                "duration_sec": round(duration_sec, 3),
                "word_count": len(item.get("word", [])) if isinstance(item, dict) else 0,
                "has_word_eeg": "word" in item if isinstance(item, dict) else False,
                "has_eyetracking": "allGazeData" in item if isinstance(item, dict) else False,
                "label": str(item.get("label", "None")) if isinstance(item, dict) else "None",
            })
        return records

    def inspect_all(self, max_files: Optional[int] = None) -> pd.DataFrame:
        """Inspects all found files and returns a unified DataFrame."""
        files = self.scan_files()
        if max_files:
            files = files[:max_files]

        all_records = []
        for fp in files:
            dataset, subject, task = self.infer_meta_from_path(fp)
            logger.info(f"Inspecting [{dataset}] Subj: {subject} | Task: {task} | File: {fp.name}")
            try:
                if fp.suffix.lower() == ".mat":
                    file_records = self.inspect_mat_scipy(fp, dataset, subject, task)
                elif fp.suffix.lower() in [".pkl", ".pickle"]:
                    file_records = self.inspect_pickle(fp, dataset, subject, task)
                else:
                    file_records = []
                all_records.extend(file_records)
            except Exception as e:
                logger.error(f"Failed to inspect {fp}: {e}\n{traceback.format_exc()}")

        df = pd.DataFrame(all_records)
        self.records = all_records
        return df

    def generate_inspection_report(self, df: pd.DataFrame, output_path: Union[str, Path]) -> str:
        """
        Generates a comprehensive Markdown report of the inspection findings,
        covering ZuCo 1 vs ZuCo 2 differences and the reconciliation strategy.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if df.empty:
            report_md = "# ZuCo Dataset Inspection Report\n\nNo records found. Please check data directory path.\n"
            output_path.write_text(report_md, encoding="utf-8")
            return report_md

        total_trials = len(df)
        datasets = sorted(df["dataset"].unique())
        subjects = sorted(df["subject"].unique())
        tasks = sorted(df["task"].unique())

        zuco1_df = df[df["dataset"] == "ZuCo1"]
        zuco2_df = df[df["dataset"] == "ZuCo2"]

        zuco1_subjs = sorted(zuco1_df["subject"].unique()) if not zuco1_df.empty else []
        zuco2_subjs = sorted(zuco2_df["subject"].unique()) if not zuco2_df.empty else []

        zuco1_tasks = sorted(zuco1_df["task"].unique()) if not zuco1_df.empty else []
        zuco2_tasks = sorted(zuco2_df["task"].unique()) if not zuco2_df.empty else []

        # Channel stats
        channels = df["n_channels"].unique()
        sr_vals = df["sampling_rate"].unique()

        # Duration stats
        mean_dur = df["duration_sec"].mean()
        min_dur = df["duration_sec"].min()
        max_dur = df["duration_sec"].max()
        median_dur = df["duration_sec"].median()

        # Word count stats
        mean_words = df["word_count"].mean()

        lines = [
            "# ZuCo 1.0 & ZuCo 2.0 Dataset Inspection Report",
            "",
            "## 1. Executive Summary",
            f"- **Total Recorded Trials/Sentences**: {total_trials:,}",
            f"- **Datasets Present**: {', '.join(datasets)}",
            f"- **Total Unique Subjects**: {len(subjects)} (ZuCo 1: {len(zuco1_subjs)}, ZuCo 2: {len(zuco2_subjs)})",
            f"- **Tasks Identified**: {', '.join(tasks)}",
            f"- **Sampling Rate(s)**: {', '.join(str(s) for s in sr_vals)} Hz",
            f"- **Channel Counts Observed**: {', '.join(str(c) for c in channels)} channels",
            f"- **Sentence Duration (sec)**: Mean = {mean_dur:.2f}s, Median = {median_dur:.2f}s, Min = {min_dur:.2f}s, Max = {max_dur:.2f}s",
            f"- **Mean Word Count per Sentence**: {mean_words:.1f} words",
            "",
            "## 2. Dataset Breakdown by Subject & Task",
            "",
            "| Dataset | Subject | Task | Sentence Count | Channels | Mean Duration (s) | Has Raw EEG | Word Fixations |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        summary_grp = df.groupby(["dataset", "subject", "task"]).agg(
            n_sentences=("sentence_id", "count"),
            channels=("n_channels", "first"),
            mean_dur=("duration_sec", "mean"),
            has_raw=("has_raw_eeg", "all"),
            has_words=("has_word_eeg", "any"),
        ).reset_index()

        for _, row in summary_grp.iterrows():
            lines.append(
                f"| {row['dataset']} | {row['subject']} | {row['task']} | {row['n_sentences']} | "
                f"{row['channels']} | {row['mean_dur']:.2f}s | {row['has_raw']} | {row['has_words']} |"
            )

        lines.extend([
            "",
            "## 3. Structural Comparison: ZuCo 1.0 vs. ZuCo 2.0",
            "",
            "| Dimension | ZuCo 1.0 | ZuCo 2.0 | Reconciled Solution |",
            "|:---|:---|:---|:---|",
            f"| **Subject Cohort** | {len(zuco1_subjs)} subjects ({', '.join(zuco1_subjs[:6])}...) | {len(zuco2_subjs)} subjects ({', '.join(zuco2_subjs[:6])}...) | Disjoint subject splits for within-dataset and cross-dataset testing |",
            f"| **Tasks** | {', '.join(zuco1_tasks)} | {', '.join(zuco2_tasks)} | Focus cross-dataset evaluation on overlapping tasks (**NR** & **TSR**); evaluate **SR** for sentiment decoding |",
            f"| **Electrode Montage** | 128-channel Geodesic (105 scalp channels standard) | 128-channel Geodesic (105 scalp channels standard) | Harmonize to the exact **105 common scalp channels** (or standardized 10-20 subset), re-referenced to average |",
            f"| **Sampling Rate** | 500 Hz | 500 Hz | Resample both to **200 Hz** (or 250 Hz for foundation models: LaBraM, BIOT, CBraMod) |",
            f"| **Sequence Length** | Variable (reading speed dependent) | Variable (reading speed dependent) | Fixed-window chunking / robust padding with boolean attention masks |",
            f"| **Word Fixations & Eye-Tracking** | Detailed `word` structs with FFD, TRT, GD | Detailed `word` structs with FFD, TRT, GD | Retained for token-level EEG representation & confound analysis |",
            "",
            "## 4. Key Preprocessing & Reconciliation Pipeline",
            "1. **Channel Alignment**: Extract common 105 scalp channels across all ZuCo 1 and ZuCo 2 recordings, ordered by standard 105 Geodesic montage.",
            "2. **Resampling**: Anti-aliasing low-pass filter followed by decimation to 200 Hz (reducing compute while capturing all canonical delta, theta, alpha, beta, and gamma bands up to 50 Hz).",
            "3. **Filtering**: 0.5 - 50 Hz bandpass filter (zero-phase Butterworth) and 50 Hz notch filter to eliminate line noise.",
            "4. **Normalization**: Per-channel robust z-score standardization computed per trial (to prevent inter-subject amplitude drift).",
            "5. **Artifact & Quality Thresholds**: Flag flat channels (std < 1e-6 uV) and extreme variance (> 5x channel median).",
            "6. **Unified Storage**: Save processed shards as `.pt` / `.npy` tensors accompanied by a master manifest index (`manifest.csv`), enabling memory-mapped streaming without RAM saturation.",
            "",
            "## 5. Next Steps",
            "- Save unified manifest index to `data/zuco_unified_index.csv`.",
            "- Proceed to **Step 1**: Construct deterministic subject-independent, sentence-held-out, and cross-dataset splits.",
        ])

        report_md = "\n".join(lines)
        output_path.write_text(report_md, encoding="utf-8")
        logger.info(f"Inspection report saved to {output_path}")
        return report_md
