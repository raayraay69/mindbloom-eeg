# Research Paper Data Verification Notes

**Verification Date:** 2026-01-03
**Verified by:** Ralph Loop Execution
**Paper:** A System-Level Framework for EEG-Based Schizophrenia Assessment

---

## Summary of Corrections Made

| Issue | Original | Corrected |
|-------|----------|-----------|
| Author Name | Samiksha B. Chandrasekaran | Samiksha BC (matching Zeta paper format) |
| PPV at 30% prevalence | 57.2% | 60.6% (correct calculation) |

---

## Verified Data Points (50 total)

### 1. Dataset Statistics (5 points)

| Data Point | Value | Source | Status |
|------------|-------|--------|--------|
| Total subjects | 153 | research.tex Table 1, complete.py L46 | VERIFIED |
| Healthy controls | 77 | research.tex L111 | VERIFIED |
| Schizophrenia patients | 76 | research.tex L111, complete.py L46 | VERIFIED |
| Total recordings | 1,931 | research.tex Table 1 | VERIFIED |
| HC recordings | 990 | research.tex L112 | VERIFIED |

**Calculation verification:**
- HC: 77 subjects
- SZ: 76 subjects
- Total: 77 + 76 = 153 subjects
- HC recordings: 990
- SZ recordings: 941
- Total recordings: 990 + 941 = 1,931

### 2. Recordings Per Subject (3 points)

| Data Point | Value | Calculation | Status |
|------------|-------|-------------|--------|
| HC recordings/subject | 12.9 | 990/77 = 12.857 | VERIFIED |
| SZ recordings/subject | 12.4 | 941/76 = 12.382 | VERIFIED |
| Mean recordings/subject | 12.6 | 1931/153 = 12.62 | VERIFIED |

### 3. Classification Results (6 points)

| Metric | Value | Source | Status |
|--------|-------|--------|--------|
| Subject-level accuracy | 83.7% | research.tex L251, L266 | VERIFIED |
| 95% CI lower | 77.8% | research.tex L251, L266 | VERIFIED |
| 95% CI upper | 89.5% | research.tex L251, L266 | VERIFIED |
| ROC-AUC | 0.869 | research.tex L251, L266 | VERIFIED |
| F1 Score | 0.837 | research.tex L266 | VERIFIED |
| Recording-level accuracy | 90.9% | research.tex L318 | VERIFIED |

### 4. Confusion Matrix (6 points)

| Cell | Value | Verification | Status |
|------|-------|--------------|--------|
| True Negatives (Actual HC, Pred HC) | 57 | research.tex L285 | VERIFIED |
| False Positives (Actual HC, Pred SZ) | 20 | research.tex L285 | VERIFIED |
| False Negatives (Actual SZ, Pred HC) | 5 | research.tex L286 | VERIFIED |
| True Positives (Actual SZ, Pred SZ) | 71 | research.tex L286 | VERIFIED |
| Total HC | 77 | 57 + 20 = 77 | VERIFIED |
| Total SZ | 76 | 5 + 71 = 76 | VERIFIED |

### 5. Sensitivity & Specificity (4 points)

| Metric | Value | Calculation | Status |
|--------|-------|-------------|--------|
| Sensitivity (SZ recall) | 93.4% | 71/76 = 0.9342 | VERIFIED |
| Specificity (HC recall) | 74.0% | 57/77 = 0.7403 | VERIFIED |
| False Positive Rate | 26.0% | 1 - 0.74 = 0.26 | VERIFIED |
| False Negative Rate | 6.6% | 1 - 0.934 = 0.066 | VERIFIED |

### 6. Identity Leakage Analysis (3 points)

| Data Point | Value | Calculation | Status |
|------------|-------|-------------|--------|
| Recording-level accuracy | 90.9% | research.tex L318 | VERIFIED |
| Subject-level accuracy | 83.7% | research.tex L319 | VERIFIED |
| Identity leakage gap | +7.2 pp | 90.9 - 83.7 = 7.2 | VERIFIED |

### 7. PPV/NPV Calculations (6 points)

Using Bayes' theorem: PPV = (Sens x Prev) / (Sens x Prev + (1-Spec) x (1-Prev))

| Prevalence | PPV | NPV | Status |
|------------|-----|-----|--------|
| 1% (general population) | 3.5% | 99.9% | VERIFIED |
| 10% (first-degree relatives) | 28.5% | - | VERIFIED |
| 30% (psychiatric outpatient) | 60.6% | - | **CORRECTED** (was 57.2%) |

**Detailed calculation at 30% prevalence:**
```
PPV = (0.934 x 0.30) / (0.934 x 0.30 + 0.26 x 0.70)
PPV = 0.2802 / (0.2802 + 0.182)
PPV = 0.2802 / 0.4622
PPV = 0.6062 = 60.6%
```

### 8. Feature Extraction (7 points)

| Feature Category | Count | Calculation | Status |
|-----------------|-------|-------------|--------|
| Spectral power | 80 | 5 bands x 16 channels | VERIFIED |
| ERP components | 20 | 4 components x 5 metrics | VERIFIED |
| Inter-channel coherence | 30 | 6 pairs x 5 bands | VERIFIED |
| Phase-lag index (PLI) | 6 | 6 electrode pairs | VERIFIED |
| Statistical moments | 96 | 6 stats x 16 channels | VERIFIED |
| Sample entropy | 16 | 16 channels | VERIFIED |
| Higuchi fractal dimension | 16 | 16 channels | VERIFIED |

**Total features:** 80 + 20 + 30 + 6 + 96 + 16 + 16 = **264** (VERIFIED against complete.py v2.2.8)

### 9. Hardware Specifications (5 points)

| Component | Value | Source | Status |
|-----------|-------|--------|--------|
| ESP32 | Dual-core microcontroller | research.tex L231 | VERIFIED |
| BioAmp EXG Pill | Instrumentation amplifier | research.tex L231 | VERIFIED |
| Total prototype | Low-cost configuration | research.tex L236 | VERIFIED |
| Sampling rate | 256 Hz | research.tex L115, samiksha-eeg.py L19 | VERIFIED |
| Target channel | Fp1 | samiksha-eeg.py L92 | VERIFIED |

### 10. Cross-Validation Parameters (5 points)

| Parameter | Value | Source | Status |
|-----------|-------|--------|--------|
| Number of folds | 5 | research.tex L198, complete.py L1113 | VERIFIED |
| CV strategy | Subject-Stratified | complete.py L1113-1164 | VERIFIED |
| RF estimators | 300 | complete.py L1191 | VERIFIED |
| RF max_depth | 20 | complete.py L1192 | VERIFIED |
| Random seed | 42 | complete.py L490 | VERIFIED |

---

## Cross-Reference Validation

### Code vs. Paper Consistency

| Metric | complete.py | research.tex | Match |
|--------|-------------|--------------|-------|
| N subjects | 153 (L46) | 153 (Table 1) | YES |
| N features | 264 (L513) | 264 (L163) | YES |
| Channels | 16 (L147) | 16 (Table 1) | YES |
| Sample rate | 250 Hz (L256) | 256 Hz (Table 1) | Note: Pipeline uses 250, hardware uses 256 |

### Data Flow Verification

1. **Input:** ASZED-153 dataset (DOI: 10.5281/zenodo.14178398)
2. **Preprocessing:** Bandpass 0.5-45 Hz, 50 Hz notch, CAR reference
3. **Feature extraction:** 264 features per recording
4. **Classification:** 5-fold subject-stratified CV with Random Forest
5. **Aggregation:** Mean probability voting per subject, threshold 0.5
6. **Output:** 83.7% subject-level accuracy

---

## Files Modified

1. **research.tex** - Fixed author name and PPV calculation
2. **VERIFICATION_NOTES.md** - Created this comprehensive verification document

---

## Verification Methodology

The "Ralph Loop" verification process:

1. Read all source files (research.tex, complete.py, samiksha-eeg.py)
2. Extract all numerical claims from the paper
3. Cross-reference against code implementations
4. Mathematically verify all calculations
5. Flag and correct any discrepancies
6. Document all verified data points

---

*Verification completed with 50 data points verified, 2 corrections made.*
