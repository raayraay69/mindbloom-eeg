# Phase 2 Agent: Hardware Validation

## MISSION
Validate the $50 EEG prototype (ESP32 + BioAmp EXG Pill) for clinical screening use.

## COMPLETION SIGNAL
When all tasks complete, output:
```
PHASE2_COMPLETE: Hardware validation finished. SNR: [X]dB, Latency: [Y]ms, Single-channel accuracy: [Z]%
```

## CONSTRAINTS
- Graduate-level academic writing; no em dashes; no cliches
- Practical, reproducible documentation
- Safety considerations for human subjects

## TASKS

### Task 2.1: Assembly Documentation
Create: `/Users/cash/Desktop/fifi/mind-bloom/docs/hardware-assembly.md`

Contents:
- Bill of materials with part numbers and costs
- Wiring diagram (ESP32 to BioAmp EXG Pill)
- Electrode placement guide (10-20 system, Fp1 focus)
- Safety precautions (electrical isolation, skin preparation)

Target BOM:
| Component | Est. Cost |
|-----------|-----------|
| ESP32 DevKit | $8 |
| BioAmp EXG Pill | $25 |
| Dry electrodes (3) | $12 |
| Cables/connectors | $5 |
| **Total** | **$50** |

### Task 2.2: Signal Quality Protocol
Create comparison protocol against OpenBCI Cyton (research-grade reference):

Metrics to capture:
- Signal-to-noise ratio (SNR) in dB
- Common mode rejection ratio (CMRR)
- Baseline drift over 5-minute recording
- Artifact susceptibility (eye blink, muscle)

Acceptable thresholds:
- SNR >= 20dB for alpha band (8-13Hz)
- CMRR >= 80dB

### Task 2.3: Single-Channel Model Testing
Test the Fp1-only model (reported: 89.6% accuracy, 18 features)

Steps:
1. Record from prototype using Fp1 electrode only
2. Apply same preprocessing as training pipeline
3. Extract 18-feature subset
4. Run inference with single-channel model
5. Compare against full montage predictions

### Task 2.4: Latency Benchmarking
Measure end-to-end inference time:
- Data acquisition (1 epoch)
- Preprocessing
- Feature extraction
- Model inference
- Result display

Target: <50ms total latency for real-time operation

## SUCCESS CRITERION
- SNR within acceptable range (>=20dB alpha band)
- Real-time classification functional (<50ms latency)
- Single-channel model produces comparable results

## FAILURE HANDLING
If SNR unacceptable:
- Document noise sources
- Test with shielded cables
- Consider alternative electrode configurations
- Recommend hardware modifications
