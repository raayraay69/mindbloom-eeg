"""
EEG Schizophrenia Prediction API
FastAPI backend for the mind-bloom website
"""

import os
import tempfile
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import warnings

warnings.filterwarnings("ignore")

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
model = None

# Constants matching training pipeline
EXPECTED_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"
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


@app.on_event("startup")
async def load_model():
    global model
    if MODEL_PATH.exists():
        model = load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}")


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
    """DC removal, bandpass, notch filter."""
    out = []
    for ch in data:
        if np.allclose(ch, 0):
            out.append(ch)
            continue
        ch = ch - np.mean(ch)
        try:
            nyq = fs / 2.0
            low, high = 0.5 / nyq, min(45 / nyq, 0.99)
            b, a = signal.butter(4, [low, high], "band")
            ch = signal.filtfilt(b, a, ch)
        except:
            pass
        try:
            b, a = signal.iirnotch(50, 30, fs=fs)
            ch = signal.filtfilt(b, a, ch)
        except:
            pass
        out.append(ch)
    return np.array(out)


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
    """Magnitude-squared coherence between electrode pairs."""
    features = []
    pairs = [(0, 8), (1, 9), (0, 1), (8, 9), (4, 12), (5, 13)]

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
    """Phase-lag index."""
    features = []
    pairs = [(0, 8), (1, 9), (0, 1), (8, 9), (4, 12), (5, 13)]

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
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an EEG file (EDF/BDF format) for schizophrenia screening prediction.
    Returns probability score and risk classification.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    filename = file.filename.lower()
    if not (filename.endswith('.edf') or filename.endswith('.bdf')):
        raise HTTPException(status_code=400, detail="Please upload an EDF or BDF file")

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
            if filename.endswith('.bdf'):
                raw = mne.io.read_raw_bdf(tmp_path, preload=True, verbose="ERROR")
            else:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose="ERROR")

            # Resample if needed
            fs = float(raw.info["sfreq"])
            if fs != SAMPLING_RATE:
                raw.resample(SAMPLING_RATE)

            # Standardize channels
            data, n_channels = standardize_to_16ch_matrix(raw, EXPECTED_CHANNELS, CHANNEL_ALIASES)

            if data.shape[1] < 500:
                raise HTTPException(status_code=400, detail="Recording too short (need at least 2 seconds)")

            if n_channels < 10:
                raise HTTPException(status_code=400, detail=f"Too few channels matched ({n_channels}/16)")

            # Preprocess
            data = preprocess(data, SAMPLING_RATE)

            # Extract features
            features = extract_all_features(data, SAMPLING_RATE)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Predict
            X = features.reshape(1, -1)
            probability = model.predict_proba(X)[0][1]
            prediction = int(probability >= 0.5)

            # Determine risk level
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.5:
                risk_level = "Low-Moderate"
            elif probability < 0.7:
                risk_level = "Moderate-High"
            else:
                risk_level = "High"

            return {
                "success": True,
                "prediction": "Schizophrenia Indicators Detected" if prediction == 1 else "No Schizophrenia Indicators",
                "probability": round(float(probability), 4),
                "risk_level": risk_level,
                "confidence": round(abs(probability - 0.5) * 2, 4),  # 0-1 scale
                "channels_matched": n_channels,
                "recording_length_seconds": round(data.shape[1] / SAMPLING_RATE, 2),
                "disclaimer": "This is a screening tool only. Results should be interpreted by a qualified healthcare professional."
            }

        finally:
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
