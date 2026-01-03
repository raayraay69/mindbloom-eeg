"""
ASZED EEG Classification Pipeline (v2.3.0)

Authoritative Reference:
  Mosaku et al. (2025). "An open-access EEG dataset from indigenous African
  populations for schizophrenia research." Data in Brief, 62, 111934.
  DOI: 10.1016/j.dib.2025.111934

Methodological corrections from earlier versions:
  - v2.0: Fixed data leakage (scaler fit only on train folds), implemented
    subject-level CV to prevent identity leakage, corrected metric aggregation
  - v2.1: Subject ID normalization, label consistency checks, clarified OOF methodology
  - v2.2: Accurate CV strategy reporting, pre-specified primary model to avoid
    post-hoc selection bias
  - v2.2.1: Fixed subject ID normalization bug when CSV sn is numeric/float
  - v2.2.2: CSV column-name normalization, channel order enforcement
  - v2.2.3: Fixed channel enforcement to use IN-PLACE zero-padding (prevents
    index shift when channels missing), BDF/EDF reader selection, notch filter
    logic (skip if above lowpass cutoff)
  - v2.2.4: BDF search in find_files() to match reader, minimum channel threshold
    guard (reject files with <10 matched channels), pooled OOF recording metrics,
    reduced entropy sample size for consistency
  - v2.2.5: Rejection tracking by label (detect QC selection bias), deterministic
    channel collision handling (keep first match), None safety for parallel workers,
    subjects-dropped-entirely tracking
  - v2.2.6:
    * Fixed CV to use TRUE subject-level stratification (StratifiedKFold on subjects,
      then expand to recordings) - prevents sample-weighted stratification bias
    * Fixed channel canonicalization bug where "Cz" -> "" (now requires delimiter
      before suffix, and never returns empty string)
    * Extended QC analysis: file-level AND subject-level rejection rates by label,
      Fisher exact test for selection bias significance
    * Subjects-dropped-entirely now excludes unlabeled subjects (eligible only)
    * Removed double parallelism (RF n_jobs=1 during CV to prevent cluster crashes)
    * Notch filter backward-compatible with older scipy versions
    * Added sanity check guards that fail loudly
    * Improved paper-facing reporting clarity
  - v2.2.7:
    * Safe pick_types: only use if it keeps channels (prevents false rejections
      on EDF/BDF files where MNE labels channels as 'misc' instead of 'eeg')
    * Full QC dict saved in JSON output (qc_analysis_full) for reviewer defensibility
    * Per-fold subject class balance printed (proves stratification worked)
  - v2.2.8:
    * Explicit class count validation before StratifiedKFold (fail loudly if
      min class < n_splits, no silent fallback to non-stratified KFold)
    * Optimized per-fold class balance: O(1) dict lookup instead of O(N) np.where
    * Final trained model saved as .pkl file (trained on ALL data after CV)
    * Channel canonicalization now strips [number] suffixes (e.g., "Fp1[1]" -> "Fp1")
  - v2.3.0:
    * CRITICAL FIX: Channel montage corrected per Data in Brief paper
      OLD: Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T3,T4,T5,T6 (O1,O2 NOT in dataset!)
      NEW: Fp1,Fp2,F3,F4,F7,F8,C3,C4,Cz,T3,T4,T5,T6,P3,P4,Pz (matches paper exactly)
    * Coherence/PLI pairs updated for correct interhemispheric connections
    * Added paper DOI (10.1016/j.dib.2025.111934) as authoritative reference

Dataset: ASZED-153 (African Schizophrenia EEG Dataset)
  Data DOI: 10.5281/zenodo.14178398
  Paper DOI: 10.1016/j.dib.2025.111934
  Specifications per paper:
    - 76 schizophrenia patients (Age: 40±12; 45F, 31M)
    - 77 healthy controls (Age: 38±13; 28F, 49M)
    - 16 channels: Fp1, Fp2, F3, F4, F7, F8, C3, C4, Cz, T3, T4, T5, T6, P3, P4, Pz
    - Devices: Contec-KT2400 (200 Hz) and BrainMaster Discovery24-E (256 Hz)
    - Paradigms: resting state, arithmetic task, auditory oddball (MMN), 40Hz ASSR
    - Two Nigerian sites (OAUTHC Ile-Ife, Wesley Guild Ilesa), 50 Hz power grid

Author: Eric Raymond
Affiliation: Purdue University Indianapolis / Indiana University South Bend
Date: January 2026
"""

import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp"
os.environ["NUMBA_DISABLE_CACHING"] = "1"

import re
import sys
import json
import time
import warnings
import argparse
import traceback
import contextlib
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import signal, stats

# Handle scipy/numpy API changes across versions
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson

try:
    from numpy import trapezoid as np_trapz
except ImportError:
    from numpy import trapz as np_trapz

# Fisher exact test for QC bias analysis
try:
    from scipy.stats import fisher_exact
    FISHER_AVAILABLE = True
except ImportError:
    FISHER_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)

from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Optional: limit BLAS/OpenMP threadpools (prevents oversubscription when combined
# with joblib preprocessing parallelism). If threadpoolctl is unavailable, the
# guard becomes a no-op.
try:
    from threadpoolctl import threadpool_limits  # type: ignore
    THREADPOOLCTL_AVAILABLE = True
except Exception:
    threadpool_limits = None  # type: ignore
    THREADPOOLCTL_AVAILABLE = False

# Default threadpool limit can be overridden via CLI (--threadpool-limit) or env.
# Set to 0 to disable limiting.
THREADPOOL_LIMIT = int(os.environ.get("ASZED_THREADPOOL_LIMIT", "1"))


def threadpool_guard():
    """Context manager that limits native threadpools when enabled."""
    if (not THREADPOOLCTL_AVAILABLE) or (THREADPOOL_LIMIT is None) or (THREADPOOL_LIMIT <= 0):
        return contextlib.nullcontext()
    try:
        return threadpool_limits(limits=int(THREADPOOL_LIMIT))
    except Exception:
        # Fail open: never crash the pipeline due to optional thread limiting.
        return contextlib.nullcontext()

# Use SLURM allocation if available, otherwise default to local CPU count
# This is for PREPROCESSING parallelism only; CV runs single-threaded
N_JOBS = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 4))

# Pre-specified to avoid model selection bias (see Cawley & Talbot, 2010)
PRIMARY_MODEL_NAME = "Random Forest"

# Standard 10-20 system 16-channel montage for ASZED-153
# Per Data in Brief paper (DOI: 10.1016/j.dib.2025.111934):
# "Both systems used identical electrode placements following the standard
# 10–20 system at sixteen sites: Fp1, Fp2, F3, F4, F7, F8, C3, C4, Cz,
# T3, T4, T5, T6, P3, P4, and Pz."
# Feature index i always corresponds to EXPECTED_CHANNELS[i % 16]
EXPECTED_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "C3", "C4",
    "Cz", "T3", "T4", "T5", "T6", "P3", "P4", "Pz"
]

# Alternative naming conventions we might encounter
CHANNEL_ALIASES = {
    "T7": "T3", "T8": "T4",  # Modern vs classic 10-20 naming
    "P7": "T5", "P8": "T6",
    "FP1": "Fp1", "FP2": "Fp2",  # Case variations
}

# Minimum channels required to consider a file valid (prevents mostly-zero inputs)
MIN_CHANNELS_REQUIRED = 10

# Entropy sample size (reduced from 500 for consistency across machines/time limits)
ENTROPY_SAMPLE_SIZE = 250


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def normalize_subject_id(sid) -> str:
    """
    Normalize subject identifiers to handle inconsistent formatting in ASZED.

    Handles:
      - "subject_001" -> "1"
      - "Subject_001" -> "1"
      - "001" -> "1"
      - 1 -> "1"
      - 1.0 -> "1"
      - NaN / None -> ""
    """
    try:
        if pd.isna(sid):
            return ""
    except Exception:
        pass

    if isinstance(sid, (int, np.integer)):
        return str(int(sid))
    if isinstance(sid, (float, np.floating)):
        if float(sid).is_integer():
            return str(int(sid))

    s = str(sid).strip().lower()
    if s in ("", "nan", "none"):
        return ""

    s = re.sub(r"^subject_", "", s)

    if re.fullmatch(r"\d+(\.0+)?", s):
        return str(int(float(s)))

    groups = re.findall(r"\d+", s)
    if groups:
        return str(int(groups[-1]))

    return s


def normalize_column_name(col: str) -> str:
    """Normalize CSV column names: strip whitespace, lowercase."""
    return str(col).strip().lower()


def canonicalize_channel_name(ch: str) -> str:
    """
    Strip common EDF adornments from channel names.
    "EEG Fp1-REF" -> "Fp1", "EEG F3-A2" -> "F3", "Fp1[1]" -> "Fp1", etc.

    v2.2.6 FIX: Now REQUIRES a delimiter (-, _, space) before reference suffixes.
    This prevents "Cz" from being incorrectly stripped to "".

    v2.2.8 FIX: Also strips [number] suffixes common in some EDF exports.

    Does NOT handle bipolar montages like "Fp1-F7" (those would need
    different processing entirely).
    """
    original = str(ch).strip()
    result = original

    # Remove "EEG " or "EEG-" prefix
    result = re.sub(r"^EEG[-_ ]?", "", result, flags=re.IGNORECASE)

    # v2.2.8 FIX: Strip [number] suffix (e.g., "Fp1[1]" -> "Fp1")
    result = re.sub(r"\[\d+\]$", "", result)

    # v2.2.6 FIX: Only strip reference suffixes if preceded by actual delimiter
    # This prevents "Cz" -> "" (the old regex r"[-_ ]?(REF|...)$" made delimiter optional)
    result = re.sub(r"[-_ ]+(REF|A1|A2|M1|M2|LE|AVG|CZ)$", "", result, flags=re.IGNORECASE)

    # Clean up any trailing separators
    result = result.strip("-_ ")

    # v2.2.6 SAFETY: Never return empty string - fall back to cleaned original
    if not result:
        result = original.strip()

    return result


class Config:
    """
    Central configuration for paths, sampling parameters, and feature extraction
    settings. Frequency bands follow standard clinical EEG conventions.

    Per Data in Brief paper (DOI: 10.1016/j.dib.2025.111934):
      - Contec-KT2400: 200 Hz, 50 Hz notch, 0-100 Hz bandpass
      - BrainMaster Discovery24-E: 256 Hz, no filtering (z-score QC)
      - We resample both to 250 Hz for uniform feature extraction
    """
    def __init__(self, data_path=None, output_path=None):
        self.DATA_ROOT = Path(data_path) if data_path else Path(".")
        self.ASZED_DIR = self.DATA_ROOT / "data" / "ASZED" / "version_1.1"
        self.CSV_PATH = self.DATA_ROOT / "data" / "ASZED_SpreadSheet.csv"
        self.OUTPUT_PATH = Path(output_path) if output_path else self.DATA_ROOT / "results"

        # Original device sampling rates (per paper):
        #   Contec-KT2400: 200 Hz
        #   BrainMaster Discovery24-E: 256 Hz
        # Target rate chosen to be between both for minimal interpolation artifacts
        self.SAMPLING_RATE = 250
        self.ORIGINAL_RATES = {"Contec-KT2400": 200, "BrainMaster-Discovery24E": 256}
        self.N_CHANNELS = 16

        # Standard clinical frequency bands
        self.BANDS = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }

        # ERP windows in sample indices at 250 Hz
        # MMN paradigm per paper: standard 1KHz/100ms, deviants 1KHz/250ms and 3KHz/100ms
        # Typical MMN latency: 100-250ms post-stimulus
        # At 250 Hz: 100ms = 25 samples, 250ms = 62.5 samples
        self.ERP_WINDOWS = {
            "N100": (20, 30),    # 80-120ms - early sensory response
            "P200": (37, 62),   # 148-248ms - attention modulation
            "MMN": (25, 62),    # 100-248ms - mismatch negativity window
            "P300": (62, 125)   # 248-500ms - cognitive processing
        }

        self.FILTER_LOW = 0.5
        self.FILTER_HIGH = 45
        self.NOTCH_FREQ = 50  # Nigeria uses 50 Hz power grid


def _import_available(mod: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def check_dependencies() -> bool:
    """Verify required packages are installed."""
    required = {
        "mne": "mne",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "numpy": "numpy",
        "pandas": "pandas",
        "joblib": "joblib",
        "tqdm": "tqdm",
    }
    missing = [pkg for mod, pkg in required.items() if not _import_available(mod)]
    if missing:
        print(f"Missing packages: {missing}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


# -----------------------------------------------------------------------------
# Feature extraction
#
# 264 features total:
#   - Spectral power (5 bands x 16 channels = 80)
#   - ERP components (4 components x 5 metrics = 20)
#   - Inter-channel coherence (6 pairs x 5 bands = 30)
#   - Phase-lag index (6 pairs = 6)
#   - Statistical moments (6 stats x 16 channels = 96)
#   - Sample entropy (16 channels = 16)
#   - Higuchi fractal dimension (16 channels = 16)
# -----------------------------------------------------------------------------

def extract_spectral_power(data, fs, bands):
    """Compute band power using Welch's method."""
    features = []
    n_ch = data.shape[0]

    for ch in range(n_ch):
        freqs, psd = signal.welch(data[ch], fs=fs, nperseg=min(256, data.shape[1]))
        for low, high in bands.values():
            idx = (freqs >= low) & (freqs <= high)
            try:
                features.append(simpson(psd[idx], x=freqs[idx]) if idx.any() else 0)
            except Exception:
                features.append(np_trapz(psd[idx], freqs[idx]) if idx.any() else 0)

    return np.array(features[:80])


def extract_erp_components(data, fs, windows):
    """
    Extract ERP-like features from averaged signal. Not true stimulus-locked
    ERPs, but temporal dynamics that may correlate with ERP abnormalities.
    """
    features = []
    avg = np.mean(data, axis=0)

    for comp, (s, e) in windows.items():
        if e < len(avg):
            w = avg[s:e]
            if comp in ["N100", "MMN"]:
                pa, pi = (np.min(w), int(np.argmin(w))) if len(w) else (0, 0)
            else:
                pa, pi = (np.max(w), int(np.argmax(w))) if len(w) else (0, 0)
            features.extend([pa, (s + pi) / fs * 1000 if fs else 0])
        else:
            features.extend([0, 0])

    for s, e in windows.values():
        if e < len(avg):
            seg = avg[s:e]
            features.extend([float(np.mean(seg)), float(np.std(seg)), float(np_trapz(seg))])
        else:
            features.extend([0, 0, 0])

    return np.array(features[:20])


def compute_coherence(data, fs, bands):
    """
    Magnitude-squared coherence between electrode pairs. Focus on
    interhemispheric and anterior-posterior connections commonly
    disrupted in schizophrenia.

    Pairs based on ASZED-153 channel layout (per Data in Brief paper):
    Fp1(0), Fp2(1), F3(2), F4(3), F7(4), F8(5), C3(6), C4(7),
    Cz(8), T3(9), T4(10), T5(11), T6(12), P3(13), P4(14), Pz(15)
    """
    features = []
    # Interhemispheric pairs commonly disrupted in SZ:
    # (0,1): Fp1-Fp2 prefrontal, (6,7): C3-C4 central, (9,10): T3-T4 temporal
    # (13,14): P3-P4 parietal, (2,3): F3-F4 frontal, (11,12): T5-T6 posterior temporal
    pairs = [(0, 1), (2, 3), (6, 7), (9, 10), (13, 14), (11, 12)]

    for c1, c2 in pairs:
        if np.allclose(data[c1], 0) or np.allclose(data[c2], 0):
            features.extend([0.0] * len(bands))
            continue
        try:
            f, coh = signal.coherence(data[c1], data[c2], fs=fs, nperseg=min(256, data.shape[1]))
            for low, high in bands.values():
                idx = (f >= low) & (f <= high)
                features.append(float(np.mean(coh[idx])) if idx.any() else 0.0)
        except Exception:
            features.extend([0.0] * len(bands))

    return np.array(features[:30])


def compute_pli(data):
    """
    Phase-lag index (Stam et al., 2007). Quantifies phase synchronization
    while being relatively insensitive to volume conduction artifacts.

    Same interhemispheric pairs as coherence, based on ASZED-153 channels.
    """
    features = []
    # Match coherence pairs: Fp1-Fp2, F3-F4, C3-C4, T3-T4, P3-P4, T5-T6
    pairs = [(0, 1), (2, 3), (6, 7), (9, 10), (13, 14), (11, 12)]

    for c1, c2 in pairs:
        if np.allclose(data[c1], 0) or np.allclose(data[c2], 0):
            features.append(0.0)
            continue
        try:
            a1, a2 = signal.hilbert(data[c1]), signal.hilbert(data[c2])
            phase_diff = np.angle(a1) - np.angle(a2)
            features.append(float(np.abs(np.mean(np.sign(np.sin(phase_diff))))))
        except Exception:
            features.append(0.0)

    return np.array(features[:6])


def extract_stats(data):
    """Basic statistical features per channel."""
    features = []
    for ch in range(data.shape[0]):
        d = data[ch]
        features.extend([
            float(np.mean(d)),
            float(np.std(d)),
            float(stats.skew(d)) if np.std(d) > 0 else 0.0,
            float(stats.kurtosis(d)) if np.std(d) > 0 else 0.0,
            float(np.sqrt(np.mean(d ** 2))),
            float(np.ptp(d)),
        ])

    return np.array(features[:96])


def compute_entropy(data, m=2, r=0.2):
    """
    Approximate sample entropy. Reduced complexity has been reported
    in schizophrenia. Uses fixed sample size for consistency.
    """
    features = []
    for ch in range(data.shape[0]):
        d = data[ch][:ENTROPY_SAMPLE_SIZE] if len(data[ch]) > ENTROPY_SAMPLE_SIZE else data[ch]

        if np.std(d) > 0:
            d = (d - np.mean(d)) / np.std(d)
            N = len(d)

            def count_matches(template_len):
                count = 0
                templates = [d[i:i + template_len] for i in range(N - template_len)]
                for i in range(len(templates)):
                    for j in range(i + 1, len(templates)):
                        if np.max(np.abs(templates[i] - templates[j])) < r:
                            count += 1
                return count

            try:
                B, A = count_matches(m), count_matches(m + 1)
                features.append(float(-np.log(A / B)) if A > 0 and B > 0 else 0.0)
            except Exception:
                features.append(0.0)
        else:
            features.append(0.0)

    return np.array(features[:16])


def compute_fd(data, kmax=10):
    """Higuchi fractal dimension. Values typically 1.4-1.8 for EEG."""
    features = []
    for ch in range(data.shape[0]):
        d = data[ch]
        N = len(d)

        if np.allclose(d, 0):
            features.append(0.0)
            continue

        if N > kmax * 2:
            L, x = [], []
            for k in range(1, min(kmax + 1, N // 2)):
                Lk = 0.0
                for m in range(k):
                    mx = int(np.floor((N - m - 1) / k))
                    if mx > 0:
                        Lmk = 0.0
                        for i in range(1, mx + 1):
                            i1, i2 = m + i * k, m + (i - 1) * k
                            if i1 < N and i2 < N:
                                Lmk += np.abs(d[i1] - d[i2])
                        Lmk = Lmk * (N - 1) / (mx * k * k)
                        Lk += Lmk
                if Lk > 0:
                    L.append(np.log(Lk / k))
                    x.append(np.log(1.0 / k))

            if len(x) > 1:
                try:
                    features.append(float(np.polyfit(x, L, 1)[0]))
                except Exception:
                    features.append(0.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)

    return np.array(features[:16])


def extract_all_features(data, fs, config):
    """Concatenate all feature groups into 264-dimensional vector."""
    f = []
    f.extend(extract_spectral_power(data, fs, config.BANDS))
    f.extend(extract_erp_components(data, fs, config.ERP_WINDOWS))
    f.extend(compute_coherence(data, fs, config.BANDS))
    f.extend(compute_pli(data))
    f.extend(extract_stats(data))
    f.extend(compute_entropy(data))
    f.extend(compute_fd(data))
    return np.array(f, dtype=float)


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------

def load_labels(csv_path: Path):
    """
    Parse ASZED metadata spreadsheet. Returns dict mapping normalized
    subject IDs to binary labels (0=control, 1=patient).
    """
    print(f"\nLoading labels from: {csv_path}")

    for encoding in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(csv_path, encoding=encoding, dtype=str)
            break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not read CSV with any encoding: {csv_path}")

    df.columns = [normalize_column_name(c) for c in df.columns]
    print(f"  Columns found: {list(df.columns)}")
    print(f"  Found {len(df)} rows")

    sn_col = None
    for candidate in ["sn", "subject", "subject_id", "id", "subj"]:
        if candidate in df.columns:
            sn_col = candidate
            break
    if sn_col is None:
        raise ValueError(f"Could not find subject ID column. Available: {list(df.columns)}")

    cat_col = None
    for candidate in ["category", "group", "diagnosis", "label", "class"]:
        if candidate in df.columns:
            cat_col = candidate
            break
    if cat_col is None:
        raise ValueError(f"Could not find category column. Available: {list(df.columns)}")

    print(f"  Using columns: sn='{sn_col}', category='{cat_col}'")

    label_map = {}
    for _, row in df.iterrows():
        sid = normalize_subject_id(row.get(sn_col, ""))
        if not sid:
            continue

        cat = str(row.get(cat_col, "")).lower().strip()
        if "control" in cat or "hc" in cat or "healthy" in cat:
            label_map[sid] = 0
        elif "patient" in cat or "schiz" in cat or "sz" in cat:
            label_map[sid] = 1

    n_ctrl = sum(1 for v in label_map.values() if v == 0)
    n_pat = sum(1 for v in label_map.values() if v == 1)
    print(f"  Mapped {len(label_map)} subjects ({n_ctrl} controls, {n_pat} patients)")

    return label_map


def find_files(aszed_dir: Path):
    """Locate EEG files and extract subject IDs from directory structure."""
    print(f"\nScanning: {aszed_dir}")
    
    files = []
    for ext in ["*.edf", "*.EDF", "*.bdf", "*.BDF"]:
        files.extend(aszed_dir.rglob(ext))
    
    ext_counts = Counter(f.suffix.lower() for f in files)
    print(f"  Found {len(files)} EEG files: {dict(ext_counts)}")

    pairs = []
    for f in files:
        sid = None
        for part in f.parts:
            part_lower = str(part).lower()
            if part_lower.startswith("subject_") or part_lower.startswith("sub_"):
                sid = normalize_subject_id(part)
                break
        if sid:
            pairs.append((f, sid))

    unique = set(s for _, s in pairs)
    print(f"  Unique subjects: {len(unique)}")

    return pairs


def scan_channel_names(pairs, n_samples=20):
    """
    Scan a sample of files to report channel naming patterns.
    """
    import mne
    mne.set_log_level("ERROR")
    
    all_channels = []
    sample_files = pairs[:min(n_samples, len(pairs))]
    
    for fp, _ in sample_files:
        try:
            ext = fp.suffix.lower()
            if ext == ".bdf":
                raw = mne.io.read_raw_bdf(str(fp), preload=False, verbose="ERROR")
            else:
                raw = mne.io.read_raw_edf(str(fp), preload=False, verbose="ERROR")
            all_channels.extend(raw.ch_names)
        except Exception:
            continue
    
    if all_channels:
        counts = Counter(all_channels)
        unique_names = sorted(counts.keys())
        print(f"\n  Channel names found (sample of {len(sample_files)} files):")
        print(f"    {unique_names[:20]}")
        if len(unique_names) > 20:
            print(f"    ... and {len(unique_names) - 20} more unique names")
        
        canonical = set()
        for ch in unique_names:
            canon = canonicalize_channel_name(ch)
            canonical.add(canon.lower())
        
        missing = [exp for exp in EXPECTED_CHANNELS if exp.lower() not in canonical]
        if missing:
            print(f"    Warning: Expected channels possibly not found: {missing}")
            print(f"    (May still match after full canonicalization)")


def standardize_to_16ch_matrix(raw, expected_channels, aliases):
    """
    Build a fixed 16×T matrix in EXPECTED_CHANNELS order.
    Missing channels are zero-padded IN-PLACE to prevent index shift.
    """
    raw_to_canonical = {}
    seen_canonical = set()
    collisions = []
    
    for ch in raw.ch_names:
        base = canonicalize_channel_name(ch)
        canonical = None
        
        for alias, canon in aliases.items():
            if base.upper() == alias.upper():
                canonical = canon
                break
        
        if canonical is None:
            for exp in expected_channels:
                if base.lower() == exp.lower():
                    canonical = exp
                    break
        
        if canonical is not None:
            if canonical in seen_canonical:
                collisions.append(f"{ch} -> {canonical} (duplicate, skipped)")
            else:
                raw_to_canonical[ch] = canonical
                seen_canonical.add(canonical)
    
    canonical_to_raw = {v: k for k, v in raw_to_canonical.items()}
    
    n_times = raw.n_times
    data = []
    channels_found = 0
    
    for exp_ch in expected_channels:
        if exp_ch in canonical_to_raw:
            raw_ch = canonical_to_raw[exp_ch]
            data.append(raw.get_data(picks=[raw_ch])[0])
            channels_found += 1
        else:
            data.append(np.zeros(n_times, dtype=float))
    
    return np.vstack(data), channels_found, collisions


def load_eeg(file_path: Path, target_fs=250):
    """Load EEG file, standardize to 16 channels, and resample if needed."""
    import mne
    mne.set_log_level("ERROR")

    ext = file_path.suffix.lower()
    if ext == ".bdf":
        raw = mne.io.read_raw_bdf(str(file_path), preload=True, verbose="ERROR")
    else:
        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose="ERROR")

    # v2.2.7: Safe pick_types - only use if it keeps channels
    # On some EDF/BDFs, MNE doesn't label channels as EEG reliably (may come in as 'misc')
    # This prevents false rejections that inflate QC drops and invite reviewer suspicion
    try:
        raw_eeg = raw.copy()
        raw_eeg.pick_types(eeg=True, exclude="bads")
        if len(raw_eeg.ch_names) > 0:
            raw = raw_eeg
        # If pick_types left nothing, keep original raw (name-based matching will handle it)
    except Exception:
        pass

    if len(raw.ch_names) == 0:
        raise ValueError("No EEG channels found")

    fs = float(raw.info["sfreq"])
    if fs != target_fs:
        raw.resample(target_fs)

    data, n_channels, collisions = standardize_to_16ch_matrix(raw, EXPECTED_CHANNELS, CHANNEL_ALIASES)
    
    return data, target_fs, n_channels, collisions


def preprocess(data, fs, config):
    """DC removal, bandpass, notch filter."""
    out = []
    for ch in data:
        if np.allclose(ch, 0):
            out.append(ch)
            continue
            
        ch = ch - np.mean(ch)

        try:
            nyq = fs / 2.0
            low = config.FILTER_LOW / nyq
            high = min(config.FILTER_HIGH / nyq, 0.99)
            b, a = signal.butter(4, [low, high], "band")
            ch = signal.filtfilt(b, a, ch)
        except Exception:
            pass

        # v2.2.6: Notch filter with backward-compatible scipy API
        if config.NOTCH_FREQ < min(config.FILTER_HIGH, fs / 2.0):
            try:
                # Try new scipy API first (fs parameter)
                b, a = signal.iirnotch(config.NOTCH_FREQ, 30, fs=fs)
                ch = signal.filtfilt(b, a, ch)
            except TypeError:
                # Fall back to old API (normalized frequency)
                try:
                    w0_normalized = config.NOTCH_FREQ / (fs / 2.0)
                    b, a = signal.iirnotch(w0_normalized, 30)
                    ch = signal.filtfilt(b, a, ch)
                except Exception:
                    pass
            except Exception:
                pass

        out.append(ch)

    return np.array(out)


def process_single_file(fp: Path, sid: str, label_map: dict, config: Config):
    """Process one EEG file: load, preprocess, extract features."""
    import mne
    mne.set_log_level("ERROR")
    warnings.filterwarnings("ignore")

    label = label_map.get(sid, -1)

    try:
        if sid not in label_map:
            return {"status": "no_label", "file": str(fp), "subject_id": sid, "label": -1}

        # Optional: prevent hidden BLAS/OpenMP oversubscription inside each joblib worker.
        # This is particularly helpful on shared clusters where native libraries may default
        # to using all cores.
        with threadpool_guard():
            data, fs, n_ch, collisions = load_eeg(fp, config.SAMPLING_RATE)

            if data.shape[1] < 500:
                return {"status": "too_short", "file": str(fp), "subject_id": sid, "label": label}

            if n_ch < MIN_CHANNELS_REQUIRED:
                return {"status": "low_channels", "file": str(fp), "subject_id": sid,
                        "label": label, "n_channels": n_ch}

            data = preprocess(data, fs, config)
            feat = extract_all_features(data, fs, config)
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "status": "ok",
            "features": feat,
            "label": label,
            "subject_id": sid,
            "n_channels": n_ch,
            "collisions": collisions
        }

    except Exception as e:
        return {"status": "error", "file": str(fp), "subject_id": sid, 
                "label": label, "error": str(e)}


def verify_label_consistency(subject_ids, y):
    """Check each subject has consistent label across recordings."""
    print("\nVerifying label consistency...")
    df = pd.DataFrame({"subject": subject_ids, "label": y})
    nunique = df.groupby("subject")["label"].nunique()
    inconsistent = nunique[nunique > 1]

    if len(inconsistent) > 0:
        print(f"  ERROR: {len(inconsistent)} subjects have inconsistent labels!")
        for subj in list(inconsistent.index)[:5]:
            labels = df[df["subject"] == subj]["label"].unique()
            print(f"    Subject {subj}: labels = {labels}")
        raise ValueError(f"Label inconsistency for {len(inconsistent)} subjects")
    else:
        print(f"  All {len(nunique)} subjects have consistent labels")

    return True


def analyze_rejections(results, label_map, file_subjects):
    """
    Comprehensive QC analysis to detect selection bias.
    
    v2.2.6: Extended with file-level rates, subject-level rates, and Fisher exact test.
    Only considers "eligible" subjects (in label_map AND discovered in files).
    """
    results = [r for r in results if isinstance(r, dict)]
    
    # Define eligible subjects: have labels AND appear in file list
    eligible_subjects = set(label_map.keys()) & set(file_subjects)
    eligible_ctrl = {s for s in eligible_subjects if label_map.get(s) == 0}
    eligible_patient = {s for s in eligible_subjects if label_map.get(s) == 1}
    
    # File-level tracking
    files_by_label = {"ctrl": 0, "patient": 0}
    rejected_files_by_label = {"ctrl": 0, "patient": 0}
    
    # Subject-level tracking
    subjects_with_any_accepted = set()
    subjects_with_any_rejection = set()
    subjects_with_any_file = defaultdict(set)
    
    for r in results:
        status = r.get("status", "unknown")
        label = r.get("label", -1)
        sid = r.get("subject_id", "")
        
        # Only count labeled files
        if label == 0:
            label_name = "ctrl"
        elif label == 1:
            label_name = "patient"
        else:
            continue  # Skip unlabeled
        
        files_by_label[label_name] += 1
        
        if status == "ok":
            subjects_with_any_accepted.add(sid)
        else:
            rejected_files_by_label[label_name] += 1
            subjects_with_any_rejection.add(sid)
        
        if sid:
            subjects_with_any_file[sid].add(status)
    
    # Compute subjects dropped entirely (among eligible only)
    subjects_dropped = {sid for sid in eligible_subjects 
                        if sid in subjects_with_any_file and "ok" not in subjects_with_any_file[sid]}
    
    dropped_ctrl = len(subjects_dropped & eligible_ctrl)
    dropped_patient = len(subjects_dropped & eligible_patient)
    
    # Compute rates
    file_rejection_rate_ctrl = (rejected_files_by_label["ctrl"] / files_by_label["ctrl"] 
                                 if files_by_label["ctrl"] > 0 else 0.0)
    file_rejection_rate_patient = (rejected_files_by_label["patient"] / files_by_label["patient"] 
                                    if files_by_label["patient"] > 0 else 0.0)
    
    subject_drop_rate_ctrl = dropped_ctrl / len(eligible_ctrl) if eligible_ctrl else 0.0
    subject_drop_rate_patient = dropped_patient / len(eligible_patient) if eligible_patient else 0.0
    
    # Fisher exact test for selection bias in subject drops
    # Contingency table: [[kept_ctrl, dropped_ctrl], [kept_patient, dropped_patient]]
    fisher_pvalue = None
    if FISHER_AVAILABLE and len(eligible_ctrl) > 0 and len(eligible_patient) > 0:
        kept_ctrl = len(eligible_ctrl) - dropped_ctrl
        kept_patient = len(eligible_patient) - dropped_patient
        try:
            _, fisher_pvalue = fisher_exact([[kept_ctrl, dropped_ctrl], 
                                              [kept_patient, dropped_patient]])
        except Exception:
            pass
    
    return {
        # File-level stats
        "files_total_ctrl": files_by_label["ctrl"],
        "files_total_patient": files_by_label["patient"],
        "files_rejected_ctrl": rejected_files_by_label["ctrl"],
        "files_rejected_patient": rejected_files_by_label["patient"],
        "file_rejection_rate_ctrl": float(file_rejection_rate_ctrl),
        "file_rejection_rate_patient": float(file_rejection_rate_patient),
        
        # Subject-level stats
        "subjects_eligible_ctrl": len(eligible_ctrl),
        "subjects_eligible_patient": len(eligible_patient),
        "subjects_with_any_rejection_ctrl": len(subjects_with_any_rejection & eligible_ctrl),
        "subjects_with_any_rejection_patient": len(subjects_with_any_rejection & eligible_patient),
        "subjects_dropped_entirely_ctrl": dropped_ctrl,
        "subjects_dropped_entirely_patient": dropped_patient,
        "subject_drop_rate_ctrl": float(subject_drop_rate_ctrl),
        "subject_drop_rate_patient": float(subject_drop_rate_patient),
        
        # Statistical test
        "fisher_pvalue_dropped_subjects": float(fisher_pvalue) if fisher_pvalue is not None else None,
        
        # Legacy format for backward compatibility
        "subjects_dropped_entirely": {"ctrl": dropped_ctrl, "patient": dropped_patient, "unknown": 0},
        "n_subjects_dropped": dropped_ctrl + dropped_patient,
    }


def process_dataset(config: Config, max_files=None):
    """Main preprocessing routine."""
    print("\n" + "=" * 70)
    print("ASZED-153 PREPROCESSING (v2.3.0)")
    print("=" * 70)

    label_map = load_labels(config.CSV_PATH)
    pairs = find_files(config.ASZED_DIR)

    if not pairs:
        raise RuntimeError("No EEG files found")

    scan_channel_names(pairs)

    file_subj = set(s for _, s in pairs)
    label_subj = set(label_map.keys())
    match = file_subj & label_subj

    print(f"\nID alignment check:")
    print(f"  Subjects in files: {len(file_subj)}")
    print(f"  Subjects in CSV:   {len(label_subj)}")
    print(f"  Matching:          {len(match)}")

    if len(match) < max(1, int(len(file_subj) * 0.9)):
        print(f"  WARNING: Only {len(match)}/{len(file_subj)} subjects matched")
        print(f"  File IDs: {sorted(list(file_subj))[:8]}...")
        print(f"  CSV IDs:  {sorted(list(label_subj))[:8]}...")

    if not match:
        raise RuntimeError("No subjects matched between files and CSV")

    if max_files:
        pairs = pairs[:max_files]

    # Extract file subjects for QC analysis
    file_subjects_processed = [sid for _, sid in pairs]

    print(f"\nProcessing {len(pairs)} files with {N_JOBS} workers...")
    print(f"  Channel order: {EXPECTED_CHANNELS}")
    print(f"  Missing channels: zero-padded in-place (no index shift)")
    print(f"  Minimum channels required: {MIN_CHANNELS_REQUIRED}")
    start_time = time.time()

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(process_single_file)(fp, sid, label_map, config)
        for fp, sid in tqdm(pairs, desc="Processing", unit="file", ncols=80)
    )

    results = [r for r in results if isinstance(r, dict)]

    valid = [r for r in results if r.get("status") == "ok"]
    rejected_counts = {
        "no_label": sum(1 for r in results if r.get("status") == "no_label"),
        "too_short": sum(1 for r in results if r.get("status") == "too_short"),
        "low_channels": sum(1 for r in results if r.get("status") == "low_channels"),
        "error": sum(1 for r in results if r.get("status") == "error"),
    }
    
    print(f"\n  File processing summary:")
    print(f"    Accepted: {len(valid)}")
    print(f"    Rejected: {sum(rejected_counts.values())} "
          f"(no_label={rejected_counts['no_label']}, too_short={rejected_counts['too_short']}, "
          f"low_channels={rejected_counts['low_channels']}, error={rejected_counts['error']})")

    # v2.2.6: Comprehensive QC analysis
    qc_analysis = analyze_rejections(results, label_map, file_subjects_processed)
    
    print(f"\n  QC Analysis (selection bias check):")
    print(f"    File rejection rate:    ctrl={qc_analysis['file_rejection_rate_ctrl']*100:.1f}%, "
          f"patient={qc_analysis['file_rejection_rate_patient']*100:.1f}%")
    print(f"    Subjects dropped:       ctrl={qc_analysis['subjects_dropped_entirely_ctrl']}/{qc_analysis['subjects_eligible_ctrl']} "
          f"({qc_analysis['subject_drop_rate_ctrl']*100:.1f}%), "
          f"patient={qc_analysis['subjects_dropped_entirely_patient']}/{qc_analysis['subjects_eligible_patient']} "
          f"({qc_analysis['subject_drop_rate_patient']*100:.1f}%)")
    if qc_analysis['fisher_pvalue_dropped_subjects'] is not None:
        print(f"    Fisher exact p-value:   {qc_analysis['fisher_pvalue_dropped_subjects']:.4f}")

    if not valid:
        raise RuntimeError("No samples processed successfully")

    all_collisions = []
    for r in valid:
        all_collisions.extend(r.get("collisions", []))
    if all_collisions:
        print(f"\n  Channel name collisions (first match kept): {len(all_collisions)}")
        for c in all_collisions[:5]:
            print(f"    {c}")
        if len(all_collisions) > 5:
            print(f"    ... and {len(all_collisions) - 5} more")

    X = np.array([r["features"] for r in valid], dtype=float)
    y = np.array([r["label"] for r in valid], dtype=int)
    subject_ids = np.array([r["subject_id"] for r in valid], dtype=str)
    
    ch_counts = [r["n_channels"] for r in valid]
    min_ch, max_ch, mean_ch = min(ch_counts), max(ch_counts), np.mean(ch_counts)
    print(f"\n  Channels matched per file: min={min_ch}, max={max_ch}, mean={mean_ch:.1f}")

    verify_label_consistency(subject_ids, y)

    elapsed = time.time() - start_time
    unique_subjects = set(subject_ids)
    n_subjects = len(unique_subjects)
    n_ctrl_subj = len(set(subject_ids[y == 0]))
    n_pat_subj = len(set(subject_ids[y == 1]))

    print(f"\nCompleted in {elapsed / 60:.1f} minutes")
    print(f"  Recordings: {len(y)} ({(y == 0).sum()} ctrl, {(y == 1).sum()} patient)")
    print(f"  Subjects:   {n_subjects} ({n_ctrl_subj} ctrl, {n_pat_subj} patient)")
    print(f"  Features:   {X.shape[1]}")

    # v2.2.6: Sanity check guards
    assert X.shape[0] == len(y) == len(subject_ids), "Array length mismatch"
    assert X.shape[1] == 264, f"Expected 264 features, got {X.shape[1]}"
    assert "" not in set(subject_ids), "Empty subject ID found"

    metadata = {
        "n_recordings": int(len(y)),
        "n_subjects": int(n_subjects),
        "n_features": int(X.shape[1]),
        "n_controls_recordings": int((y == 0).sum()),
        "n_patients_recordings": int((y == 1).sum()),
        "n_controls_subjects": int(n_ctrl_subj),
        "n_patients_subjects": int(n_pat_subj),
        "recordings_per_subject_mean": float(len(y) / n_subjects),
        "channels_matched_range": [int(min_ch), int(max_ch)],
        "channels_matched_mean": float(mean_ch),
        "files_rejected": rejected_counts,
        "qc_analysis": qc_analysis,
    }

    # v2.2.6: Sanity guard
    assert metadata["n_subjects"] == len(set(subject_ids)), "Subject count mismatch in metadata"

    if metadata["n_subjects"] != 153:
        print(f"\nNote: Expected 153 subjects, got {n_subjects}")

    return X, y, subject_ids, metadata


# -----------------------------------------------------------------------------
# Subject-level aggregation
# -----------------------------------------------------------------------------

def aggregate_predictions_by_subject(subjects, y_true, y_pred, y_prob, threshold=0.5):
    """Average predictions across recordings from same subject."""
    df = pd.DataFrame({
        "subject": subjects,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob
    })

    agg = df.groupby("subject").agg(
        y_true=("y_true", "first"),
        y_prob_mean=("y_prob", "mean"),
        y_pred_vote=("y_pred", "mean"),
        n_recordings=("y_true", "count"),
    ).reset_index()

    agg["y_pred_prob"] = (agg["y_prob_mean"] >= threshold).astype(int)
    agg["y_pred_majority"] = (agg["y_pred_vote"] >= 0.5).astype(int)

    return agg


# -----------------------------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------------------------

def subject_stratified_cv_splits(X, y, groups, n_splits=5, random_state=42):
    """
    v2.2.8: TRUE subject-level stratification with explicit class count validation.

    1. Build subject table: unique subjects + their label (first recording's label)
    2. Validate min class count >= n_splits (fail loudly, no silent fallback)
    3. Run StratifiedKFold on SUBJECTS to get balanced subject folds
    4. Convert subject folds -> recording indices by membership

    This prevents sample-weighted stratification bias where subjects with
    more recordings get more weight in determining fold composition.
    """
    # Build subject table
    unique_subjects = sorted(set(groups))
    subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}

    # Get label for each subject (first recording's label)
    subject_labels = {}
    for i, subj in enumerate(groups):
        if subj not in subject_labels:
            subject_labels[subj] = y[i]

    subject_y = np.array([subject_labels[s] for s in unique_subjects])

    # v2.2.8: Explicit class count check - fail loudly, no silent fallback
    # This prevents silently running non-stratified folds if QC drops reduce a class
    counts = np.bincount(subject_y)
    if counts.min() < n_splits:
        raise ValueError(
            f"Not enough subjects per class for {n_splits}-fold stratification: "
            f"class counts={counts.tolist()} (need at least {n_splits} per class). "
            f"Consider reducing n_splits or checking QC drops for class imbalance."
        )

    # Stratified split on subjects (now guaranteed to have enough samples)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_name = "SubjectStratifiedKFold"
    splitter = skf.split(unique_subjects, subject_y)
    
    # Convert subject folds to recording indices
    recording_folds = []
    for train_subj_idx, test_subj_idx in splitter:
        train_subjects = set(unique_subjects[i] for i in train_subj_idx)
        test_subjects = set(unique_subjects[i] for i in test_subj_idx)

        train_idx = np.array([i for i, g in enumerate(groups) if g in train_subjects])
        test_idx = np.array([i for i, g in enumerate(groups) if g in test_subjects])

        recording_folds.append((train_idx, test_idx))

    # v2.2.8: Return subject_labels dict for efficient per-fold class balance calculation
    return recording_folds, cv_name, subject_labels


def subject_level_cv(X, y, groups, n_splits=5, random_state=42):
    """
    Subject-level CV with TRUE subject stratification.

    v2.2.8: Uses subject_stratified_cv_splits() to ensure folds are
    balanced by subject labels, not sample-weighted by recording counts.
    Now also receives subject_labels dict for efficient per-fold class balance.
    """
    print("\n" + "=" * 70)
    print("SUBJECT-LEVEL CROSS-VALIDATION")
    print("=" * 70)

    folds, cv_name, subject_label = subject_stratified_cv_splits(X, y, groups, n_splits, random_state)

    print(f"  Strategy: {cv_name} (k={n_splits})")
    print(f"  Note: Folds constructed by stratifying on SUBJECTS, then expanded to recordings")
    print(f"  Grouping: Subject ID (no subject appears in both train and test)")
    print(f"  Scaling:  Fit on train fold only (Pipeline)")

    # v2.2.6: RF uses n_jobs=1 to avoid double parallelism with preprocessing
    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=1,  # v2.2.6: Prevent cluster oversubscription
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=random_state)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=random_state)),
        ]),
    }

    results = {}

    for model_name, pipeline in models.items():
        print(f"\nEvaluating {model_name}...")

        fold_acc, fold_auc, fold_f1 = [], [], []
        all_y_true, all_y_pred, all_y_prob, all_subjects = [], [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            train_subj = set(groups[train_idx])
            test_subj = set(groups[test_idx])
            overlap = train_subj & test_subj
            if overlap:
                raise ValueError(f"Subject leakage detected: {sorted(list(overlap))[:10]}")

            # v2.2.8: Per-fold subject class balance using cached dict (O(1) per subject)
            train_ctrl = sum(1 for s in train_subj if subject_label[s] == 0)
            train_sz = sum(1 for s in train_subj if subject_label[s] == 1)
            test_ctrl = sum(1 for s in test_subj if subject_label[s] == 0)
            test_sz = sum(1 for s in test_subj if subject_label[s] == 1)

            # Optional: limit native BLAS/OpenMP threads during fit/predict.
            # Even though RF uses n_jobs=1, some linear algebra backends can still
            # oversubscribe cores unless constrained.
            with threadpool_guard():
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                try:
                    y_prob = pipeline.predict_proba(X_test)[:, 1]
                except Exception:
                    y_prob = y_pred.astype(float)

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_y_prob.extend(np.asarray(y_prob).tolist())
            all_subjects.extend(groups[test_idx].tolist())

            acc = accuracy_score(y_test, y_pred)
            fold_acc.append(acc)

            try:
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                auc = np.nan
            fold_auc.append(auc)

            f1 = f1_score(y_test, y_pred, zero_division=0)
            fold_f1.append(f1)

            print(f"  Fold {fold_idx + 1}: acc={acc:.3f}, auc={auc:.3f}, "
                  f"train={len(train_subj)} subj ({train_ctrl}C/{train_sz}S), "
                  f"test={len(test_subj)} subj ({test_ctrl}C/{test_sz}S)")

        all_y_true = np.array(all_y_true, dtype=int)
        all_y_pred = np.array(all_y_pred, dtype=int)
        all_y_prob = np.array(all_y_prob, dtype=float)
        all_subjects = np.array(all_subjects, dtype=str)

        oof_rec_acc = accuracy_score(all_y_true, all_y_pred)
        oof_rec_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
        try:
            oof_rec_auc = roc_auc_score(all_y_true, all_y_prob)
        except Exception:
            oof_rec_auc = np.nan

        subject_agg = aggregate_predictions_by_subject(all_subjects, all_y_true, all_y_pred, all_y_prob)

        # v2.2.6: Sanity guards
        assert subject_agg["subject"].is_unique, "Duplicate subjects in aggregation"
        n_unique_subjects = len(np.unique(all_subjects))
        assert len(subject_agg) == n_unique_subjects, \
            f"Subject aggregation length {len(subject_agg)} != unique subjects {n_unique_subjects}"

        subj_acc = accuracy_score(subject_agg["y_true"], subject_agg["y_pred_prob"])
        try:
            subj_auc = roc_auc_score(subject_agg["y_true"], subject_agg["y_prob_mean"])
        except Exception:
            subj_auc = np.nan
        subj_f1 = f1_score(subject_agg["y_true"], subject_agg["y_pred_prob"], zero_division=0)

        results[model_name] = {
            "recording_accuracy_fold_mean": float(np.mean(fold_acc)),
            "recording_accuracy_fold_std": float(np.std(fold_acc)),
            "fold_accuracies": [float(x) for x in fold_acc],
            "recording_accuracy_oof": float(oof_rec_acc),
            "recording_auc_oof": float(oof_rec_auc) if np.isfinite(oof_rec_auc) else float("nan"),
            "recording_f1_oof": float(oof_rec_f1),
            "subject_accuracy": float(subj_acc),
            "subject_auc": float(subj_auc) if np.isfinite(subj_auc) else float("nan"),
            "subject_f1": float(subj_f1),
            "all_y_true": all_y_true,
            "all_y_pred": all_y_pred,
            "all_y_prob": all_y_prob,
            "all_subjects": all_subjects,
            "subject_aggregation": subject_agg,
        }

        print(f"  Recording-level (pooled OOF): {oof_rec_acc*100:.1f}%")
        print(f"  Subject-level: {subj_acc*100:.1f}%")

    return results, cv_name


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=2000, confidence=0.95, random_state=42):
    """Bootstrap confidence interval."""
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            scores.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception:
            continue

    if not scores:
        return np.nan, np.nan

    alpha = (1 - confidence) / 2
    return (
        float(np.percentile(scores, alpha * 100)),
        float(np.percentile(scores, (1 - alpha) * 100))
    )


# -----------------------------------------------------------------------------
# Results reporting
# -----------------------------------------------------------------------------

def print_final_summary(cv_results, metadata, cv_name, n_folds):
    """Print formatted results with v2.2.6 clarity improvements."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nDataset: ASZED-153 (DOI: 10.5281/zenodo.14178398)")
    print(f"  Recordings: {metadata['n_recordings']}")
    print(f"  Subjects:   {metadata['n_subjects']}")
    print(f"  Controls:   {metadata['n_controls_subjects']} subj "
          f"({metadata['n_controls_recordings']} rec)")
    print(f"  Patients:   {metadata['n_patients_subjects']} subj "
          f"({metadata['n_patients_recordings']} rec)")

    # v2.2.6: QC summary
    qc = metadata.get("qc_analysis", {})
    if qc:
        print(f"\nQC Summary:")
        print(f"  File rejection rate:  ctrl={qc.get('file_rejection_rate_ctrl', 0)*100:.1f}%, "
              f"patient={qc.get('file_rejection_rate_patient', 0)*100:.1f}%")
        print(f"  Subject drop rate:    ctrl={qc.get('subject_drop_rate_ctrl', 0)*100:.1f}%, "
              f"patient={qc.get('subject_drop_rate_patient', 0)*100:.1f}%")
        if qc.get('fisher_pvalue_dropped_subjects') is not None:
            print(f"  Selection bias test:  Fisher p={qc['fisher_pvalue_dropped_subjects']:.4f}")

    print(f"\nMethodology:")
    print(f"  CV: {n_folds}-fold {cv_name}")
    print(f"      (Folds constructed by subject-level stratification, then expanded to recordings)")
    print(f"  Scaling: StandardScaler fit on train folds only")
    print(f"  Aggregation: Mean probability per subject, threshold=0.5")
    print(f"  Primary model: {PRIMARY_MODEL_NAME} (pre-specified)")
    print(f"  Primary metrics: Subject-level; recording-level is pooled OOF")

    paper_model = PRIMARY_MODEL_NAME if PRIMARY_MODEL_NAME in cv_results else max(
        cv_results, key=lambda k: cv_results[k]["subject_accuracy"]
    )

    print(f"\nRecording-level accuracy (pooled OOF, N={metadata['n_recordings']}):")
    for name, res in cv_results.items():
        marker = " *" if name == paper_model else ""
        print(f"  {name}: {res['recording_accuracy_oof']*100:.1f}%{marker}")

    print(f"\nSubject-level accuracy (N={metadata['n_subjects']}):")
    for name, res in cv_results.items():
        marker = " *" if name == paper_model else ""
        subj_agg = res["subject_aggregation"]
        lo, hi = bootstrap_ci(subj_agg["y_true"].values, subj_agg["y_pred_prob"].values, accuracy_score)
        print(f"  {name}: {res['subject_accuracy']*100:.1f}% "
              f"(95% CI: {lo*100:.1f}-{hi*100:.1f}%){marker}")

    paper_res = cv_results[paper_model]
    subj_agg = paper_res["subject_aggregation"]

    cm_rec = confusion_matrix(paper_res["all_y_true"], paper_res["all_y_pred"])
    cm_subj = confusion_matrix(subj_agg["y_true"], subj_agg["y_pred_prob"])

    # v2.2.6: Sanity guard - confusion matrix sums
    cm_rec_sum = cm_rec.sum()
    cm_subj_sum = cm_subj.sum()
    assert cm_rec_sum == metadata['n_recordings'], \
        f"Recording CM sum {cm_rec_sum} != n_recordings {metadata['n_recordings']}"
    assert cm_subj_sum == metadata['n_subjects'], \
        f"Subject CM sum {cm_subj_sum} != n_subjects {metadata['n_subjects']}"

    print(f"\nConfusion matrix ({paper_model}):")
    print(f"\n  Recording-level (N={cm_rec_sum}):")
    print(f"                 Pred HC  Pred SZ")
    print(f"    Actual HC    {cm_rec[0,0]:5d}    {cm_rec[0,1]:5d}")
    print(f"    Actual SZ    {cm_rec[1,0]:5d}    {cm_rec[1,1]:5d}")

    print(f"\n  Subject-level (N={cm_subj_sum}):")
    print(f"                 Pred HC  Pred SZ")
    print(f"    Actual HC    {cm_subj[0,0]:5d}    {cm_subj[0,1]:5d}")
    print(f"    Actual SZ    {cm_subj[1,0]:5d}    {cm_subj[1,1]:5d}")

    print(f"\nClassification report (subject-level):")
    print(classification_report(subj_agg["y_true"], subj_agg["y_pred_prob"], target_names=["Control", "Patient"]))

    return paper_model, float(cv_results[paper_model]["subject_accuracy"])


def save_results(config, X, y, subject_ids, cv_results, metadata, cv_name, n_folds, random_state=42):
    """Save features, results, and trained model with v2.2.8 QC metadata."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    output_dir = config.OUTPUT_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    feat_path = output_dir / f"aszed_features_{timestamp}.npz"
    np.savez_compressed(feat_path, X=X, y=y, subject_ids=subject_ids)
    print(f"  Features: {feat_path}")

    # v2.2.8: Train final model on ALL data and save as .pkl
    print(f"\n  Training final {PRIMARY_MODEL_NAME} on all data...")
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1,  # Use all cores for final training (no CV parallelism conflict)
        )),
    ])
    final_pipeline.fit(X, y)

    model_path = output_dir / f"aszed_model_{timestamp}.pkl"
    from joblib import dump
    dump(final_pipeline, model_path)
    print(f"  Model:    {model_path}")

    # Extract QC for top-level inclusion
    qc = metadata.get("qc_analysis", {})

    summary = {
        "version": "2.3.0",
        "timestamp": timestamp,
        "dataset": {
            "name": "ASZED-153",
            "data_doi": "10.5281/zenodo.14178398",
            "paper_doi": "10.1016/j.dib.2025.111934",
            "paper_ref": "Mosaku et al. (2025) Data in Brief 62:111934",
            "description": "African Schizophrenia EEG Dataset - Indigenous African populations"
        },
        "metadata": {
            "n_recordings": metadata["n_recordings"],
            "n_subjects": metadata["n_subjects"],
            "n_features": metadata["n_features"],
            "n_controls_recordings": metadata["n_controls_recordings"],
            "n_patients_recordings": metadata["n_patients_recordings"],
            "n_controls_subjects": metadata["n_controls_subjects"],
            "n_patients_subjects": metadata["n_patients_subjects"],
            "recordings_per_subject_mean": metadata["recordings_per_subject_mean"],
            "channels_matched_range": metadata["channels_matched_range"],
            "channels_matched_mean": metadata["channels_matched_mean"],
            "n_folds": int(n_folds),
        },
        "methodology": {
            "cv_strategy": cv_name,
            "cv_description": "Folds constructed by subject-level stratification, then expanded to recordings",
            "n_folds": int(n_folds),
            "grouping": "subject_id",
            "normalization": "StandardScaler inside CV (Pipeline)",
            "subject_aggregation": "mean probability, threshold=0.5",
            "primary_model": PRIMARY_MODEL_NAME,
            "channel_order": EXPECTED_CHANNELS,
            "missing_channel_handling": "zero-padded in-place (no index shift)",
            "channel_collision_handling": "keep first match (deterministic)",
            "min_channels_required": MIN_CHANNELS_REQUIRED,
            "entropy_sample_size": ENTROPY_SAMPLE_SIZE,
        },
        "qc": {
            "files_rejected": metadata.get("files_rejected", {}),
            "file_rejection_rate_ctrl": qc.get("file_rejection_rate_ctrl"),
            "file_rejection_rate_patient": qc.get("file_rejection_rate_patient"),
            "subjects_eligible_ctrl": qc.get("subjects_eligible_ctrl"),
            "subjects_eligible_patient": qc.get("subjects_eligible_patient"),
            "subjects_dropped_entirely_ctrl": qc.get("subjects_dropped_entirely_ctrl"),
            "subjects_dropped_entirely_patient": qc.get("subjects_dropped_entirely_patient"),
            "subject_drop_rate_ctrl": qc.get("subject_drop_rate_ctrl"),
            "subject_drop_rate_patient": qc.get("subject_drop_rate_patient"),
            "fisher_pvalue_dropped_subjects": qc.get("fisher_pvalue_dropped_subjects"),
            # v2.2.7: Full QC dict for reviewer defensibility
            "qc_analysis_full": qc,
        },
        "results": {},
    }

    for model_name, res in cv_results.items():
        summary["results"][model_name] = {
            "recording_accuracy_oof": float(res["recording_accuracy_oof"]),
            "recording_auc_oof": float(res["recording_auc_oof"]),
            "recording_f1_oof": float(res["recording_f1_oof"]),
            "recording_accuracy_fold_mean": float(res["recording_accuracy_fold_mean"]),
            "recording_accuracy_fold_std": float(res["recording_accuracy_fold_std"]),
            "subject_accuracy": float(res["subject_accuracy"]),
            "subject_auc": float(res["subject_auc"]),
            "subject_f1": float(res["subject_f1"]),
            "fold_accuracies": [float(x) for x in res["fold_accuracies"]],
        }

    # Add model path to JSON for traceability
    summary["model_path"] = str(model_path)

    json_path = output_dir / f"cv_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results:  {json_path}")

    return output_dir


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    global N_JOBS, THREADPOOL_LIMIT
    parser = argparse.ArgumentParser(description="ASZED EEG Analysis Pipeline v2.3.0")
    parser.add_argument("--data-path", default=".", help="Data directory")
    parser.add_argument("--output-path", default=None, help="Output directory")
    parser.add_argument("--max-files", type=int, default=None, help="Limit files processed")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds")
    parser.add_argument("-j", "--jobs", type=int, default=None, help="Parallel workers (preprocessing only)")
    parser.add_argument(
        "--threadpool-limit",
        type=int,
        default=THREADPOOL_LIMIT,
        help=(
            "Limit BLAS/OpenMP native threads via threadpoolctl (0 disables). "
            "Useful to prevent oversubscription on shared clusters."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.jobs:
        N_JOBS = args.jobs

    THREADPOOL_LIMIT = int(args.threadpool_limit)
    if THREADPOOL_LIMIT > 0 and not THREADPOOLCTL_AVAILABLE:
        print("WARNING: --threadpool-limit requested but threadpoolctl is not installed; continuing without limits")

    np.random.seed(args.seed)

    print("=" * 70)
    print("ASZED EEG Classification Pipeline v2.3.0")
    print("Dataset: African Schizophrenia EEG Dataset")
    print("  Data DOI: 10.5281/zenodo.14178398")
    print("  Paper DOI: 10.1016/j.dib.2025.111934")
    print("=" * 70)
    print(f"\nDate: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Workers: {N_JOBS} (preprocessing only; CV is single-threaded)")
    if THREADPOOL_LIMIT > 0 and THREADPOOLCTL_AVAILABLE:
        print(f"Threadpool limit: {THREADPOOL_LIMIT} (via threadpoolctl)")
    elif THREADPOOL_LIMIT > 0 and not THREADPOOLCTL_AVAILABLE:
        print(f"Threadpool limit: requested={THREADPOOL_LIMIT} (threadpoolctl missing; not applied)")
    else:
        print("Threadpool limit: disabled")
    print(f"Seed: {args.seed}")
    print(f"Primary model: {PRIMARY_MODEL_NAME}")

    if not check_dependencies():
        sys.exit(1)

    import mne
    mne.set_log_level("ERROR")

    config = Config(args.data_path, args.output_path)

    print(f"\nPaths:")
    print(f"  Data:   {config.DATA_ROOT}")
    print(f"  EDF:    {config.ASZED_DIR}")
    print(f"  Labels: {config.CSV_PATH}")
    print(f"  Output: {config.OUTPUT_PATH}")

    if not config.CSV_PATH.exists():
        print(f"Error: Labels not found at {config.CSV_PATH}")
        sys.exit(1)

    try:
        X, y, subject_ids, metadata = process_dataset(config, args.max_files)
        metadata["n_folds"] = int(args.n_folds)
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(np.unique(y)) < 2:
        print("Error: Need both classes")
        sys.exit(1)

    try:
        cv_results, cv_name = subject_level_cv(X, y, subject_ids, n_splits=args.n_folds, random_state=args.seed)
    except Exception as e:
        print(f"CV failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    paper_model, paper_acc = print_final_summary(cv_results, metadata, cv_name=cv_name, n_folds=args.n_folds)
    save_results(config, X, y, subject_ids, cv_results, metadata, cv_name=cv_name, n_folds=args.n_folds, random_state=args.seed)

    print("\n" + "=" * 70)
    print("FOR PAPER")
    print("=" * 70)
    print(f"""
Dataset: ASZED-153 ({metadata['n_subjects']} subjects, {metadata['n_recordings']} recordings)
  DOI: 10.5281/zenodo.14178398
Evaluation: {args.n_folds}-fold {cv_name}
  - Folds constructed by subject-level stratification, then expanded to recordings
  - No subject appears in both train and test within any fold
Model: {paper_model} (pre-specified)
Subject-level accuracy: {paper_acc*100:.1f}%
Note: Confusion matrix sums to N={metadata['n_subjects']} (subjects, not recordings)
""")


if __name__ == "__main__":
    main()