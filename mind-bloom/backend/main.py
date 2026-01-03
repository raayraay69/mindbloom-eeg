"""
EEG Schizophrenia Prediction API
FastAPI backend for the mind-bloom website
"""

import os
import tempfile
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eeg_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Feature extraction imports
from scipy import signal, stats
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson

try:
    from numpy import trapezoid as np_trapz
except ImportError:
    from numpy import trapz as np_trapz

app = FastAPI(
    title="EEG Schizophrenia Screening API",
    description="Upload EEG files for schizophrenia risk assessment",
    version="1.0.0",
    root_path="/api"
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
MODEL_PATH = Path(__file__).parent / "schizophrenia_backend_model.pkl"
METADATA_PATH = Path(__file__).parent / "model_metadata.json"
model = None
model_metadata = None

# Constants matching training pipeline (v2.3.0)
# Per Data in Brief paper (DOI: 10.1016/j.dib.2025.111934):
# "Both systems used identical electrode placements following the standard
# 10-20 system at sixteen sites: Fp1, Fp2, F3, F4, F7, F8, C3, C4, Cz,
# T3, T4, T5, T6, P3, P4, and Pz."
EXPECTED_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "C3", "C4",
    "Cz", "T3", "T4", "T5", "T6", "P3", "P4", "Pz"
]
CHANNEL_ALIASES = {
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
    "FP1": "Fp1", "FP2": "Fp2",
}
SAMPLING_RATE = 250
BANDS = {
    "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "gamma": (30, 45),
}
ERP_WINDOWS = {"N100": (20, 30), "P200": (37, 62), "MMN": (25, 62), "P300": (62, 125)}
ENTROPY_SAMPLE_SIZE = 250


# ============================================================================
# Response Models
# ============================================================================

class ChannelStatus(BaseModel):
    """Status of individual EEG channel"""
    name: str
    found: bool
    quality_score: Optional[float] = None  # 0-1, based on signal characteristics
    is_zero: bool = False
    is_noisy: bool = False
    snr_db: Optional[float] = None

class SignalQuality(BaseModel):
    """Overall signal quality metrics"""
    overall_score: float  # 0-1, weighted average
    channels_found: int
    channels_expected: int
    zero_channels: int
    noisy_channels: int
    average_snr_db: Optional[float] = None
    preprocessing_status: Dict[str, bool]  # Which filters succeeded

class ValidationDetails(BaseModel):
    """Detailed validation information"""
    file_format: str
    original_sampling_rate: float
    resampled: bool
    duration_seconds: float
    total_samples: int
    channels_in_file: List[str]
    channel_statuses: List[ChannelStatus]
    signal_quality: SignalQuality
    validation_passed: bool
    validation_errors: List[str]
    validation_warnings: List[str]

class PreviewData(BaseModel):
    """EEG preview data for visualization"""
    channel_names: List[str]
    sample_data: List[List[float]]  # First few seconds of data
    sampling_rate: float
    time_points: List[float]


@app.on_event("startup")
async def load_model():
    global model, model_metadata
    if MODEL_PATH.exists():
        model = load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}")

    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            model_metadata = json.load(f)
        print(f"Model metadata loaded from {METADATA_PATH}")
    else:
        print(f"WARNING: Model metadata not found at {METADATA_PATH}")
        # Create default metadata if file doesn't exist
        model_metadata = {
            "version": "1.0.0",
            "dataset": "ASZED-153",
            "algorithm": "RandomForestClassifier"
        }


# ============================================================================
# Signal Quality Assessment Functions
# ============================================================================

def calculate_snr(channel_data: np.ndarray, fs: float) -> Optional[float]:
    """Calculate Signal-to-Noise Ratio in dB."""
    try:
        if np.allclose(channel_data, 0) or len(channel_data) < 100:
            return None

        # Use power spectral density to estimate SNR
        freqs, psd = signal.welch(channel_data, fs=fs, nperseg=min(256, len(channel_data)))

        # Signal: 0.5-45 Hz (EEG range)
        signal_idx = (freqs >= 0.5) & (freqs <= 45)
        # Noise: >45 Hz
        noise_idx = freqs > 45

        if not signal_idx.any() or not noise_idx.any():
            return None

        signal_power = np.mean(psd[signal_idx])
        noise_power = np.mean(psd[noise_idx])

        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return float(snr)
        return None
    except Exception as e:
        logger.warning(f"SNR calculation failed: {e}")
        return None


def assess_channel_quality(channel_data: np.ndarray, fs: float, channel_name: str) -> ChannelStatus:
    """Assess quality of a single channel."""
    is_zero = np.allclose(channel_data, 0)

    if is_zero:
        return ChannelStatus(
            name=channel_name,
            found=True,
            quality_score=0.0,
            is_zero=True,
            is_noisy=False,
            snr_db=None
        )

    # Calculate metrics
    snr = calculate_snr(channel_data, fs)
    std = float(np.std(channel_data))

    # Check for excessive noise (very high std or very low SNR)
    is_noisy = (std > 200) or (snr is not None and snr < -5)

    # Calculate quality score (0-1)
    quality_score = 0.5  # default

    if snr is not None:
        # Good SNR: >10dB = 1.0, Poor SNR: <-10dB = 0.0
        quality_score = np.clip((snr + 10) / 20, 0, 1)

    # Penalize extreme variance
    if std > 100:
        quality_score *= 0.7

    return ChannelStatus(
        name=channel_name,
        found=True,
        quality_score=float(quality_score),
        is_zero=False,
        is_noisy=is_noisy,
        snr_db=snr
    )


def assess_signal_quality(data: np.ndarray, fs: float, channels_found: int,
                         preprocessing_status: Dict[str, bool]) -> tuple:
    """Assess overall signal quality and create channel statuses."""
    channel_statuses = []
    zero_count = 0
    noisy_count = 0
    snr_values = []
    quality_scores = []

    for i, channel_name in enumerate(EXPECTED_CHANNELS):
        channel_data = data[i]

        # Check if channel was found (non-zero)
        if np.allclose(channel_data, 0):
            channel_statuses.append(ChannelStatus(
                name=channel_name,
                found=False,
                quality_score=0.0,
                is_zero=True,
                is_noisy=False,
                snr_db=None
            ))
            zero_count += 1
        else:
            status = assess_channel_quality(channel_data, fs, channel_name)
            channel_statuses.append(status)

            if status.is_noisy:
                noisy_count += 1
            if status.snr_db is not None:
                snr_values.append(status.snr_db)
            if status.quality_score is not None:
                quality_scores.append(status.quality_score)

    # Calculate overall quality score
    overall_score = np.mean(quality_scores) if quality_scores else 0.0

    # Penalize for missing channels
    channel_penalty = channels_found / len(EXPECTED_CHANNELS)
    overall_score *= channel_penalty

    signal_quality = SignalQuality(
        overall_score=float(overall_score),
        channels_found=channels_found,
        channels_expected=len(EXPECTED_CHANNELS),
        zero_channels=zero_count,
        noisy_channels=noisy_count,
        average_snr_db=float(np.mean(snr_values)) if snr_values else None,
        preprocessing_status=preprocessing_status
    )

    return signal_quality, channel_statuses


# ============================================================================
# Feature extraction functions (matching training pipeline)
# ============================================================================

def canonicalize_channel_name(ch: str) -> str:
    """Strip EDF adornments from channel names."""
    import re
    result = str(ch).strip()
    result = re.sub(r"^EEG[-_ ]?", "", result, flags=re.IGNORECASE)
    result = re.sub(r"\[\d+\]$", "", result)
    result = re.sub(r"[-_ ]+(REF|A1|A2|M1|M2|LE|AVG|CZ)$", "", result, flags=re.IGNORECASE)
    result = result.strip("-_ ")
    return result if result else str(ch).strip()


def standardize_to_16ch_matrix(raw, expected_channels, aliases):
    """Build fixed 16xT matrix in expected channel order."""
    raw_to_canonical = {}
    seen_canonical = set()

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

        if canonical and canonical not in seen_canonical:
            raw_to_canonical[ch] = canonical
            seen_canonical.add(canonical)

    canonical_to_raw = {v: k for k, v in raw_to_canonical.items()}

    data = []
    channels_found = 0

    for exp_ch in expected_channels:
        if exp_ch in canonical_to_raw:
            raw_ch = canonical_to_raw[exp_ch]
            data.append(raw.get_data(picks=[raw_ch])[0])
            channels_found += 1
        else:
            data.append(np.zeros(raw.n_times, dtype=float))

    return np.vstack(data), channels_found


def preprocess(data, fs):
    """DC removal, bandpass, notch filter. Returns processed data and filter status."""
    out = []
    bandpass_success = True
    notch_success = True

    for ch in data:
        if np.allclose(ch, 0):
            out.append(ch)
            continue
        ch = ch - np.mean(ch)

        # Bandpass filter
        try:
            nyq = fs / 2.0
            low, high = 0.5 / nyq, min(45 / nyq, 0.99)
            b, a = signal.butter(4, [low, high], "band")
            ch = signal.filtfilt(b, a, ch)
        except Exception as e:
            bandpass_success = False
            logger.warning(f"Bandpass filter failed: {e}")

        # Notch filter
        try:
            b, a = signal.iirnotch(50, 30, fs=fs)
            ch = signal.filtfilt(b, a, ch)
        except Exception as e:
            notch_success = False
            logger.warning(f"Notch filter failed: {e}")

        out.append(ch)

    preprocessing_status = {
        "dc_removal": True,
        "bandpass_filter": bandpass_success,
        "notch_filter": notch_success
    }

    return np.array(out), preprocessing_status


def extract_spectral_power(data, fs, bands):
    """Compute band power using Welch's method."""
    features = []
    for ch in range(data.shape[0]):
        freqs, psd = signal.welch(data[ch], fs=fs, nperseg=min(256, data.shape[1]))
        for low, high in bands.values():
            idx = (freqs >= low) & (freqs <= high)
            try:
                features.append(simpson(psd[idx], x=freqs[idx]) if idx.any() else 0)
            except:
                features.append(np_trapz(psd[idx], freqs[idx]) if idx.any() else 0)
    return np.array(features[:80])


def extract_erp_components(data, fs, windows):
    """Extract ERP-like features from averaged signal."""
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
    Magnitude-squared coherence between electrode pairs.

    Pairs based on ASZED-153 channel layout (per Data in Brief paper):
    Fp1(0), Fp2(1), F3(2), F4(3), F7(4), F8(5), C3(6), C4(7),
    Cz(8), T3(9), T4(10), T5(11), T6(12), P3(13), P4(14), Pz(15)

    Interhemispheric pairs commonly disrupted in SZ:
    (0,1): Fp1-Fp2 prefrontal, (2,3): F3-F4 frontal, (6,7): C3-C4 central,
    (9,10): T3-T4 temporal, (13,14): P3-P4 parietal, (11,12): T5-T6 posterior temporal
    """
    features = []
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
        except:
            features.extend([0.0] * len(bands))

    return np.array(features[:30])


def compute_pli(data):
    """
    Phase-lag index (Stam et al., 2007). Quantifies phase synchronization
    while being relatively insensitive to volume conduction artifacts.

    Same interhemispheric pairs as coherence, based on ASZED-153 channels:
    Fp1-Fp2, F3-F4, C3-C4, T3-T4, P3-P4, T5-T6
    """
    features = []
    pairs = [(0, 1), (2, 3), (6, 7), (9, 10), (13, 14), (11, 12)]

    for c1, c2 in pairs:
        if np.allclose(data[c1], 0) or np.allclose(data[c2], 0):
            features.append(0.0)
            continue
        try:
            a1, a2 = signal.hilbert(data[c1]), signal.hilbert(data[c2])
            phase_diff = np.angle(a1) - np.angle(a2)
            features.append(float(np.abs(np.mean(np.sign(np.sin(phase_diff))))))
        except:
            features.append(0.0)

    return np.array(features[:6])


def extract_stats(data):
    """Basic statistical features per channel."""
    features = []
    for ch in range(data.shape[0]):
        d = data[ch]
        features.extend([
            float(np.mean(d)), float(np.std(d)),
            float(stats.skew(d)) if np.std(d) > 0 else 0.0,
            float(stats.kurtosis(d)) if np.std(d) > 0 else 0.0,
            float(np.sqrt(np.mean(d ** 2))), float(np.ptp(d)),
        ])
    return np.array(features[:96])


def compute_entropy(data, m=2, r=0.2):
    """Sample entropy."""
    features = []
    for ch in range(data.shape[0]):
        d = data[ch][:ENTROPY_SAMPLE_SIZE] if len(data[ch]) > ENTROPY_SAMPLE_SIZE else data[ch]
        if np.std(d) > 0:
            d = (d - np.mean(d)) / np.std(d)
            N = len(d)
            def count_matches(tlen):
                count = 0
                templates = [d[i:i + tlen] for i in range(N - tlen)]
                for i in range(len(templates)):
                    for j in range(i + 1, len(templates)):
                        if np.max(np.abs(templates[i] - templates[j])) < r:
                            count += 1
                return count
            try:
                B, A = count_matches(m), count_matches(m + 1)
                features.append(float(-np.log(A / B)) if A > 0 and B > 0 else 0.0)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
    return np.array(features[:16])


def compute_fd(data, kmax=10):
    """Higuchi fractal dimension."""
    features = []
    for ch in range(data.shape[0]):
        d = data[ch]
        N = len(d)
        if np.allclose(d, 0) or N <= kmax * 2:
            features.append(0.0)
            continue
        L, x = [], []
        for k in range(1, min(kmax + 1, N // 2)):
            Lk = 0.0
            for m in range(k):
                mx = int(np.floor((N - m - 1) / k))
                if mx > 0:
                    Lmk = sum(np.abs(d[m + i * k] - d[m + (i - 1) * k])
                              for i in range(1, mx + 1) if m + i * k < N and m + (i - 1) * k < N)
                    Lmk = Lmk * (N - 1) / (mx * k * k)
                    Lk += Lmk
            if Lk > 0:
                L.append(np.log(Lk / k))
                x.append(np.log(1.0 / k))
        if len(x) > 1:
            try:
                features.append(float(np.polyfit(x, L, 1)[0]))
            except:
                features.append(0.0)
        else:
            features.append(0.0)
    return np.array(features[:16])


def extract_all_features(data, fs):
    """Extract all 264 features."""
    f = []
    f.extend(extract_spectral_power(data, fs, BANDS))
    f.extend(extract_erp_components(data, fs, ERP_WINDOWS))
    f.extend(compute_coherence(data, fs, BANDS))
    f.extend(compute_pli(data))
    f.extend(extract_stats(data))
    f.extend(compute_entropy(data))
    f.extend(compute_fd(data))
    return np.array(f, dtype=float)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "EEG Schizophrenia Screening API",
        "status": "ready" if model is not None else "model not loaded",
        "version": "1.0.0",
        "model_info": model_metadata if model_metadata else None
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/validate")
async def validate_eeg(file: UploadFile = File(...)):
    """
    Validate and preview an EEG file without making predictions.
    Returns detailed validation information and preview data.
    """
    filename = file.filename.lower()
    validation_errors = []
    validation_warnings = []

    # Basic file type check
    if not (filename.endswith('.edf') or filename.endswith('.bdf')):
        validation_errors.append("Invalid file format. Please upload an EDF or BDF file.")
        return {
            "error": "INVALID_INPUT",
            "code": "UNSUPPORTED_FORMAT",
            "message": "Invalid file format. Please upload an EDF or BDF file.",
            "accepted_formats": [".edf", ".bdf"],
            "validation_passed": False,
            "validation_errors": validation_errors,
            "validation_warnings": validation_warnings
        }

    try:
        import mne
        mne.set_log_level("ERROR")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Load EEG file
            logger.info(f"Validating file: {filename}")
            file_format = "BDF" if filename.endswith('.bdf') else "EDF"

            if filename.endswith('.bdf'):
                raw = mne.io.read_raw_bdf(tmp_path, preload=True, verbose="ERROR")
            else:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose="ERROR")

            # Get file info
            original_fs = float(raw.info["sfreq"])
            duration = raw.n_times / original_fs
            channels_in_file = raw.ch_names.copy()

            logger.info(f"File loaded: {len(channels_in_file)} channels, {duration:.2f}s duration, {original_fs}Hz")

            # Resample if needed
            resampled = False
            if original_fs != SAMPLING_RATE:
                raw.resample(SAMPLING_RATE)
                resampled = True
                validation_warnings.append(f"Resampled from {original_fs}Hz to {SAMPLING_RATE}Hz")

            # Standardize channels
            data, n_channels = standardize_to_16ch_matrix(raw, EXPECTED_CHANNELS, CHANNEL_ALIASES)

            # Validation checks
            min_duration_seconds = 2.0
            min_samples = int(min_duration_seconds * SAMPLING_RATE)
            if data.shape[1] < min_samples:
                validation_errors.append({
                    "code": "INSUFFICIENT_DURATION",
                    "message": f"Recording too short: {duration:.2f}s (minimum {min_duration_seconds}s required)",
                    "current_duration": round(duration, 2),
                    "minimum_duration": min_duration_seconds
                })

            min_channels_required = 10
            if n_channels < min_channels_required:
                validation_errors.append({
                    "code": "INSUFFICIENT_CHANNELS",
                    "message": f"Too few channels matched: {n_channels}/16 found (minimum {min_channels_required} required)",
                    "channels_found": n_channels,
                    "channels_required": min_channels_required,
                    "channels_expected": len(EXPECTED_CHANNELS)
                })

            # Missing channels
            missing_channels = []
            for i, ch in enumerate(EXPECTED_CHANNELS):
                if np.allclose(data[i], 0):
                    missing_channels.append(ch)

            if missing_channels:
                validation_warnings.append(
                    f"Missing channels ({len(missing_channels)}): {', '.join(missing_channels)}"
                )

            # Preprocess and assess quality
            preprocessed_data, preprocessing_status = preprocess(data, SAMPLING_RATE)

            # Track filter failures
            if not preprocessing_status["bandpass_filter"]:
                validation_warnings.append("Bandpass filter failed - using DC-removed data only")
            if not preprocessing_status["notch_filter"]:
                validation_warnings.append("Notch filter (50Hz) failed - line noise may be present")

            # Assess signal quality
            signal_quality, channel_statuses = assess_signal_quality(
                preprocessed_data, SAMPLING_RATE, n_channels, preprocessing_status
            )

            # Quality warnings
            if signal_quality.noisy_channels > 0:
                validation_warnings.append(
                    f"{signal_quality.noisy_channels} channels appear noisy (high variance or low SNR)"
                )

            if signal_quality.overall_score < 0.3:
                validation_warnings.append(
                    f"Low overall signal quality (score: {signal_quality.overall_score:.2f})"
                )

            # Create preview data (first 2 seconds)
            preview_samples = min(500, preprocessed_data.shape[1])  # 2 seconds at 250Hz
            preview_data = PreviewData(
                channel_names=[ch.name for ch in channel_statuses if ch.found],
                sample_data=[
                    preprocessed_data[i, :preview_samples].tolist()
                    for i, ch in enumerate(channel_statuses) if ch.found
                ],
                sampling_rate=SAMPLING_RATE,
                time_points=[i / SAMPLING_RATE for i in range(preview_samples)]
            )

            validation_details = ValidationDetails(
                file_format=file_format,
                original_sampling_rate=original_fs,
                resampled=resampled,
                duration_seconds=float(duration),
                total_samples=int(raw.n_times),
                channels_in_file=channels_in_file,
                channel_statuses=channel_statuses,
                signal_quality=signal_quality,
                validation_passed=len(validation_errors) == 0,
                validation_errors=validation_errors,
                validation_warnings=validation_warnings
            )

            logger.info(
                f"Validation complete: {'PASSED' if validation_details.validation_passed else 'FAILED'}, "
                f"Quality: {signal_quality.overall_score:.2f}, "
                f"Channels: {n_channels}/16"
            )

            return {
                "validation": validation_details,
                "preview": preview_data
            }

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Validation error for {filename}: {str(e)}", exc_info=True)
        validation_errors.append(f"Error processing file: {str(e)}")
        return {
            "validation_passed": False,
            "validation_errors": validation_errors,
            "validation_warnings": validation_warnings
        }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an EEG file (EDF/BDF format) for schizophrenia screening prediction.
    Returns probability score, risk classification, and detailed validation info.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    filename = file.filename.lower()
    validation_errors = []
    validation_warnings = []

    if not (filename.endswith('.edf') or filename.endswith('.bdf')):
        logger.warning(f"Invalid file type attempted: {filename}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_INPUT",
                "code": "UNSUPPORTED_FORMAT",
                "message": "Invalid file format. Please upload an EDF or BDF file.",
                "accepted_formats": [".edf", ".bdf"]
            }
        )

    try:
        import mne
        mne.set_log_level("ERROR")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Load EEG file
            logger.info(f"Processing prediction for: {filename}")
            file_format = "BDF" if filename.endswith('.bdf') else "EDF"

            if filename.endswith('.bdf'):
                raw = mne.io.read_raw_bdf(tmp_path, preload=True, verbose="ERROR")
            else:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose="ERROR")

            # Get file info
            original_fs = float(raw.info["sfreq"])
            duration = raw.n_times / original_fs
            channels_in_file = raw.ch_names.copy()

            # Resample if needed
            resampled = False
            if original_fs != SAMPLING_RATE:
                raw.resample(SAMPLING_RATE)
                resampled = True
                validation_warnings.append(f"Resampled from {original_fs}Hz to {SAMPLING_RATE}Hz")

            # Standardize channels
            data, n_channels = standardize_to_16ch_matrix(raw, EXPECTED_CHANNELS, CHANNEL_ALIASES)

            # Validation checks
            min_duration_seconds = 2.0
            min_samples = int(min_duration_seconds * SAMPLING_RATE)
            if data.shape[1] < min_samples:
                error_detail = {
                    "error": "INVALID_INPUT",
                    "code": "INSUFFICIENT_DURATION",
                    "message": f"Recording too short: {duration:.2f}s (minimum {min_duration_seconds}s required)",
                    "current_duration": round(duration, 2),
                    "minimum_duration": min_duration_seconds
                }
                logger.error(f"Validation failed for {filename}: {error_detail['message']}")
                raise HTTPException(status_code=400, detail=error_detail)

            min_channels_required = 10
            if n_channels < min_channels_required:
                # Find which channels were matched
                found_channels = []
                missing_channels = []
                for i, ch in enumerate(EXPECTED_CHANNELS):
                    if np.allclose(data[i], 0):
                        missing_channels.append(ch)
                    else:
                        found_channels.append(ch)

                error_detail = {
                    "error": "INVALID_INPUT",
                    "code": "INSUFFICIENT_CHANNELS",
                    "message": f"Too few channels matched: {n_channels}/16 found (minimum {min_channels_required} required)",
                    "channels_found": n_channels,
                    "channels_required": min_channels_required,
                    "channels_expected": len(EXPECTED_CHANNELS),
                    "found_channel_names": found_channels,
                    "missing_channel_names": missing_channels
                }
                logger.error(f"Validation failed for {filename}: {error_detail['message']}")
                raise HTTPException(status_code=400, detail=error_detail)

            # Check for missing channels (for warnings)
            missing_channels = []
            for i, ch in enumerate(EXPECTED_CHANNELS):
                if np.allclose(data[i], 0):
                    missing_channels.append(ch)

            if missing_channels:
                validation_warnings.append(
                    f"Missing {len(missing_channels)} channels: {', '.join(missing_channels)}"
                )

            # Preprocess
            preprocessed_data, preprocessing_status = preprocess(data, SAMPLING_RATE)

            # Track filter failures
            if not preprocessing_status["bandpass_filter"]:
                validation_warnings.append("Bandpass filter failed - using DC-removed data only")
            if not preprocessing_status["notch_filter"]:
                validation_warnings.append("Notch filter (50Hz) failed - line noise may be present")

            # Assess signal quality
            signal_quality, channel_statuses = assess_signal_quality(
                preprocessed_data, SAMPLING_RATE, n_channels, preprocessing_status
            )

            # Quality warnings
            if signal_quality.noisy_channels > 0:
                validation_warnings.append(
                    f"{signal_quality.noisy_channels} channels appear noisy"
                )

            if signal_quality.overall_score < 0.3:
                validation_warnings.append(
                    f"Low signal quality (score: {signal_quality.overall_score:.2f}) - results may be less reliable"
                )

            # Extract features
            features = extract_all_features(preprocessed_data, SAMPLING_RATE)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Predict
            X = features.reshape(1, -1)
            probability = model.predict_proba(X)[0][1]
            prediction = int(probability >= 0.5)

            # Determine risk level and uncertainty status
            confidence_status = "CONFIDENT"
            clinical_recommendation = None

            # Flag uncertain predictions (probability between 0.4 and 0.6)
            if 0.4 <= probability <= 0.6:
                confidence_status = "UNCERTAIN"
                clinical_recommendation = "Recommend clinical follow-up; prediction confidence below threshold"

            # Determine risk level
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.5:
                risk_level = "Low-Moderate"
            elif probability < 0.7:
                risk_level = "Moderate-High"
            else:
                risk_level = "High"

            logger.info(
                f"Prediction complete for {filename}: "
                f"Risk={risk_level}, Prob={probability:.4f}, "
                f"Quality={signal_quality.overall_score:.2f}, "
                f"Channels={n_channels}/16"
            )

            validation_details = ValidationDetails(
                file_format=file_format,
                original_sampling_rate=original_fs,
                resampled=resampled,
                duration_seconds=float(duration),
                total_samples=int(data.shape[1]),
                channels_in_file=channels_in_file,
                channel_statuses=channel_statuses,
                signal_quality=signal_quality,
                validation_passed=True,
                validation_errors=[],
                validation_warnings=validation_warnings
            )

            return {
                "success": True,
                "prediction": "Schizophrenia Indicators Detected" if prediction == 1 else "No Schizophrenia Indicators",
                "probability": round(float(probability), 4),
                "risk_level": risk_level,
                "confidence": round(abs(probability - 0.5) * 2, 4),
                "confidence_status": confidence_status,
                "clinical_recommendation": clinical_recommendation,
                "channels_matched": n_channels,
                "recording_length_seconds": round(preprocessed_data.shape[1] / SAMPLING_RATE, 2),
                "validation": validation_details,
                "model_info": {
                    "version": model_metadata.get("version", "1.0.0"),
                    "training_date": model_metadata.get("training_date", "Unknown"),
                    "dataset": model_metadata.get("dataset", {}).get("name", "ASZED-153"),
                    "algorithm": model_metadata.get("model", {}).get("algorithm", "RandomForestClassifier"),
                    "feature_count": model_metadata.get("model", {}).get("feature_count", 264),
                    "accuracy_reported": model_metadata.get("performance", {}).get("production_accuracy", 0.837)
                } if model_metadata else None,
                "disclaimer": "This is a screening tool only. Results should be interpreted by a qualified healthcare professional."
            }

        finally:
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
