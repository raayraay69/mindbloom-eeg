# Phase 3 Agent: External Validation

## MISSION
Test model generalization on independent (non-ASZED) schizophrenia EEG datasets.

## COMPLETION SIGNAL
When all tasks complete, output:
```
PHASE3_COMPLETE: External validation finished. Dataset: [name], Accuracy: [X]%, Domain shift: [description]
```

## DEPENDENCY
**BLOCKED UNTIL**: Phase 1 complete with accuracy >= 80%

## CONSTRAINTS
- Graduate-level academic writing; no em dashes; no cliches
- No retraining; test generalization only
- Honest reporting of performance degradation

## TASKS

### Task 3.1: Dataset Identification
Research and evaluate candidate datasets:

| Dataset | Subjects | Channels | Sampling | Access |
|---------|----------|----------|----------|--------|
| TUEP (Temple Univ) | ~100 | 19-21 | 256Hz | Request |
| Kaggle Schizophrenia | ~84 | 19 | 250Hz | Public |
| OpenNeuro ds003478 | Variable | 64+ | Variable | Open |

Evaluation criteria:
- Diagnostic confirmation method (DSM criteria, clinical records)
- Recording protocol similarity to ASZED (resting-state, eyes-closed)
- Channel compatibility (must include Fp1, Fp2, F7, F8, or subset)
- Data format (EDF, BDF, or convertible)

### Task 3.2: Preprocessing Alignment
Adapt external data to match ASZED pipeline:

ASZED preprocessing (reference):
- Sampling rate: 500Hz (resample if needed)
- Filtering: 0.5-45Hz bandpass
- Epoch length: 4 seconds (2000 samples)
- Channel montage: 19-channel 10-20 system
- Artifact rejection: amplitude threshold

Document any unavoidable differences (channel count, reference scheme).

### Task 3.3: Zero-Shot Evaluation
Run the production model WITHOUT any retraining:

```python
# Pseudocode
external_features = extract_features(external_data, aszed_pipeline)
predictions = production_model.predict(external_features)
metrics = evaluate(predictions, external_labels)
```

Record:
- Accuracy, sensitivity, specificity
- Confusion matrix
- Confidence score distribution (compare to ASZED distribution)

### Task 3.4: Domain Shift Analysis
If performance degrades (expected), analyze causes:

Potential domain shift factors:
- **Population**: Age, medication status, illness duration
- **Hardware**: Different EEG systems, electrode impedances
- **Protocol**: Eyes open vs closed, task vs resting
- **Geography**: ASZED from Nigeria; different demographics

Document in: `/Users/cash/Desktop/fifi/Research-Paper/EXTERNAL_VALIDATION.md`

Include:
- Feature distribution comparison (t-SNE, PCA visualization)
- Which features show largest shift
- Recommendations for domain adaptation (if needed)

## SUCCESS CRITERION
Accuracy >= 70% on external dataset OR clear documentation of domain shift with actionable recommendations.

## FAILURE HANDLING
If accuracy < 70%:
- This is EXPECTED and scientifically valuable
- Document the performance gap honestly
- Identify top contributing factors
- Propose mitigation: transfer learning, domain adaptation, multi-site training
- Do NOT claim generalization if evidence does not support it
