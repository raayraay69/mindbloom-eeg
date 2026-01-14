# MindBloom: EEG-Based Schizophrenia Screening System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)

## Abstract

MindBloom is a machine learning-based diagnostic system for schizophrenia screening using electroencephalography (EEG) signals. The system implements a Random Forest classification pipeline trained on the ASZED-153 dataset—a novel African Schizophrenia EEG Dataset comprising 153 subjects from Nigerian clinical sites. Our approach achieves **83.7% subject-level accuracy** (95% CI: 77.8–89.5%) with an ROC-AUC of **0.869**, demonstrating clinically relevant sensitivity (93.4%) for schizophrenia detection while maintaining methodological rigor through subject-stratified cross-validation to prevent identity leakage.

The platform integrates a trauma-informed web application built with Next.js and Firebase, featuring adaptive severity disclosure powered by large language models. This repository contains the complete classification pipeline, web application, hardware integration code for low-cost EEG acquisition, and a multi-phase validation framework (RALPH) for clinical deployment readiness.

---

## Table of Contents

1. [Research Background](#research-background)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [System Architecture](#system-architecture)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Validation Framework](#validation-framework)
9. [Hardware Integration](#hardware-integration)
10. [Contributing](#contributing)
11. [Citation](#citation)
12. [License](#license)

---

## Research Background

Schizophrenia affects approximately 1% of the global population and presents significant diagnostic challenges, particularly in resource-limited settings where access to psychiatric specialists is constrained. Traditional diagnosis relies on clinical interviews and behavioral observation, which require specialized training and can delay intervention during critical early illness phases.

EEG-based biomarkers offer a promising avenue for objective, accessible schizophrenia screening. Prior research has identified several neurophysiological signatures associated with schizophrenia:

- **Gamma-band abnormalities**: Reduced 40 Hz auditory steady-state response (ASSR) power and phase-locking
- **Mismatch negativity (MMN) deficits**: Attenuated ERP responses to deviant auditory stimuli
- **Altered spectral power distributions**: Increased theta/delta activity, reduced alpha peak frequency
- **Disrupted functional connectivity**: Aberrant coherence patterns across cortical regions

This work addresses a critical gap in the literature: the underrepresentation of African populations in neuroscience research. By training and validating our models on the ASZED-153 dataset, we contribute to more equitable global mental health diagnostics.

---

## Dataset

### ASZED-153 (African Schizophrenia EEG Dataset)

| Attribute | Specification |
|-----------|---------------|
| **Total Subjects** | 153 (76 SZ, 77 HC) |
| **Recordings** | 1,931 (941 SZ, 990 HC) |
| **Demographics** | Ages 38–40 years; mixed gender |
| **Clinical Sites** | OAUTHC Ile-Ife, Wesley Guild Ilesa (Nigeria) |
| **Paradigms** | Resting state, Arithmetic task, Auditory oddball (MMN), 40 Hz ASSR |
| **Hardware** | Contec-KT2400 (200 Hz), BrainMaster Discovery24-E (256 Hz) |
| **Montage** | 16-channel 10-20 system |
| **Publication** | DOI: [10.1016/j.dib.2025.111934](https://doi.org/10.1016/j.dib.2025.111934) |

### Channel Configuration

```
Fp1, Fp2, F3, F4, F7, F8, C3, C4, Cz, T3, T4, T5, T6, P3, P4, Pz
```

---

## Methodology

### Preprocessing Pipeline

1. **DC Offset Removal**: Zero-mean centering per channel
2. **Bandpass Filtering**: 4th-order Butterworth, 0.5–45 Hz passband
3. **Notch Filtering**: 50 Hz power line interference rejection
4. **Re-referencing**: Common Average Reference (CAR)
5. **Quality Control**: Automated artifact detection with Fisher exact tests for rejection bias

### Feature Extraction

The pipeline extracts **264 features** per recording across multiple signal processing domains:

| Feature Category | Count | Description |
|-----------------|-------|-------------|
| Spectral Power | 80 | 5 frequency bands × 16 channels (δ, θ, α, β, γ) |
| ERP Components | 20 | P50, N100, P200, P300 metrics |
| Inter-channel Coherence | 30 | 6 electrode pairs × 5 bands |
| Phase-Lag Index (PLI) | 6 | Functional connectivity measures |
| Statistical Moments | 96 | Mean, variance, skewness, kurtosis, min, max |
| Sample Entropy | 16 | Nonlinear complexity per channel |
| Higuchi Fractal Dimension | 16 | Signal self-similarity metrics |

### Classification Model

- **Algorithm**: Random Forest Classifier
- **Hyperparameters**: 300 estimators, max_depth=20, class_weight='balanced'
- **Validation**: 5-fold subject-stratified cross-validation
- **Aggregation**: Recording-level predictions aggregated via majority voting to subject-level diagnosis

### Methodological Safeguards

To prevent **identity leakage**—a common pitfall in biomedical ML where within-subject correlations inflate accuracy estimates—all recordings from a given subject are confined to a single fold during cross-validation. This ensures that test set predictions reflect true generalization to unseen individuals.

---

## System Architecture

```
mindbloom-eeg/
├── Working-code/
│   └── Classification Pipeline/
│       └── complete.py          # Core ML pipeline (v2.3.0, 1,691 LOC)
├── mind-bloom/                  # Next.js web application
│   ├── src/
│   │   ├── app/                 # App Router pages
│   │   ├── components/          # React components
│   │   └── ai/                  # Genkit AI flows
│   └── public/
├── ralph/                       # Multi-phase validation framework
├── training/                    # Training notebooks (Colab/Kaggle)
├── Research-Paper/              # LaTeX manuscript and documentation
└── hardware/                    # ESP32 + BioAmp integration
```

### Technology Stack

**Backend & Data Science**
- Python 3.8+ with NumPy, SciPy, scikit-learn, pandas
- MNE-Python for neuroimaging data structures
- joblib for parallel feature extraction

**Frontend & Application**
- Next.js 15.5.9 with React 19
- Radix UI + Tailwind CSS + shadcn/ui
- Firebase Authentication & Hosting
- Google Genkit with Gemini 2.5 Flash for adaptive disclosure

**Hardware**
- ESP32 dual-core microcontroller
- BioAmp EXG Pill instrumentation amplifier
- AgAgCl electrodes with 256 Hz sampling

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 18+ and npm/pnpm
- Git

### Classification Pipeline

```bash
# Clone the repository
git clone https://github.com/raayraay69/mindbloom-eeg.git
cd mindbloom-eeg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy scipy scikit-learn pandas mne pyedflib joblib tqdm
```

### Web Application

```bash
cd mind-bloom

# Install dependencies
npm install

# Configure environment variables
cp .env.example .env.local
# Edit .env.local with your Firebase and Genkit credentials

# Run development server
npm run dev
```

---

## Usage

### Running the Classification Pipeline

```python
from complete import EEGSchizophreniaClassifier

# Initialize classifier
classifier = EEGSchizophreniaClassifier(
    data_dir='/path/to/ASZED-153',
    n_folds=5,
    random_state=42
)

# Run full pipeline
results = classifier.run_pipeline()

# Access metrics
print(f"Subject-level Accuracy: {results['accuracy']:.1%}")
print(f"ROC-AUC: {results['roc_auc']:.3f}")
print(f"Sensitivity: {results['sensitivity']:.1%}")
print(f"Specificity: {results['specificity']:.1%}")
```

### Web Application

1. Navigate to `http://localhost:3000`
2. Create an account or sign in
3. Upload EEG files (.EDF/.BDF format, max 500 MB)
4. View analysis results with adaptive severity disclosure
5. Access therapeutic resources and session history

---

## Results

### Classification Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| **Subject-level Accuracy** | 83.7% | 77.8–89.5% |
| **ROC-AUC** | 0.869 | — |
| **Sensitivity (SZ Recall)** | 93.4% | — |
| **Specificity (HC Recall)** | 74.0% | — |
| **Recording-level Accuracy** | 90.9% | — |

### Feature Importance Analysis

Top discriminative features identified through Random Forest importance scores:

1. Gamma-band (30–45 Hz) spectral power in temporal regions (T3, T4)
2. Alpha-band (8–13 Hz) coherence between frontal and parietal sites
3. P300 amplitude and latency metrics
4. Sample entropy in frontal channels (Fp1, Fp2)
5. Theta/alpha ratio in central electrodes

---

## Validation Framework

### RALPH: Rigorous Adaptive Loop for Phased Hypothesis-testing

The RALPH framework orchestrates a multi-phase validation protocol ensuring clinical deployment readiness:

| Phase | Objective | Success Criterion | Dependencies |
|-------|-----------|-------------------|--------------|
| **Phase 1** | Internal Validation | ≥80% accuracy on held-out ASZED data | None |
| **Phase 2** | Hardware Validation | SNR acceptable, latency <50ms | None |
| **Phase 3** | External Validation | ≥70% accuracy OR documented domain shift | Phase 1 |
| **Phase 4** | Prospective Pilot | IRB approval, DSM-5 correlation | Phases 2, 3 |

Phases 1 and 2 execute in parallel; Phase 3 depends on Phase 1 completion; Phase 4 requires both Phases 2 and 3.

---

## Hardware Integration

### Low-Cost EEG Acquisition System

For resource-limited deployment scenarios, we provide integration code for a minimal viable EEG acquisition system:

```
Components:
- ESP32 DevKit (dual-core, WiFi/BLE)
- BioAmp EXG Pill (instrumentation amplifier)
- AgAgCl electrodes (single-channel Fp1)
- Serial communication @ 115200 baud
```

See `Working-code/samiksha-eeg.py` for real-time signal acquisition with:
- 256 Hz sampling rate
- Live signal quality monitoring
- Automatic EDF file generation
- Visual feedback via LED indicators

---

## Contributing

We welcome contributions from the research community. Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes with clear messages
4. Push to your fork
5. Open a Pull Request with detailed description

### Development Standards

- Follow PEP 8 for Python code
- Use TypeScript strict mode for web components
- Include docstrings and type annotations
- Add unit tests for new functionality
- Update documentation as needed

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{mindbloom2025,
  title={MindBloom: EEG-Based Schizophrenia Screening System},
  author={MindBloom Contributors},
  year={2025},
  url={https://github.com/raayraay69/mindbloom-eeg}
}
```

### Related Publications

```bibtex
@article{aszed2025,
  title={ASZED-153: African Schizophrenia EEG Dataset},
  journal={Data in Brief},
  year={2025},
  doi={10.1016/j.dib.2025.111934}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- The clinical teams at OAUTHC Ile-Ife and Wesley Guild Ilesa for data collection
- The ASZED-153 dataset authors for enabling research on underrepresented populations
- The open-source neuroimaging community (MNE-Python, scikit-learn)

---

## Contact

For questions, collaborations, or clinical partnership inquiries, please open an issue or contact the maintainers through the repository.

---

*MindBloom: A gentle space for understanding and well-being.*
