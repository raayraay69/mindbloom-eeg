# Phase 1 Agent: Internal Validation

## MISSION
Validate the deployed MindBloom model against held-out ASZED data. Report results precisely.

## COMPLETION SIGNAL
When all tasks complete, output:
```
PHASE1_COMPLETE: Internal validation finished. Accuracy: [X]%, Sensitivity: [Y]%, Specificity: [Z]%
```

## CONSTRAINTS
- Graduate-level academic writing; no em dashes; no cliches
- Report metrics honestly; do not inflate results
- Document all discrepancies between training and production

## TASKS

### Task 1.1: Locate Held-Out Data
```
Search for ASZED test/validation split in:
- /Users/cash/Desktop/fifi/training/
- /Users/cash/Desktop/fifi/Working-code/
```
If no held-out set exists, document this and suggest creating one with proper subject-level separation.

### Task 1.2: API Prediction Testing
```
Endpoint: Cloud Run deployment
Method: POST predictions for each held-out recording
Capture: prediction, confidence score, response time
```

### Task 1.3: Metric Comparison
Compare against reported values:
- Training accuracy: 93.75% (subject-level, Random Forest)
- Production accuracy: 83.7% (reported)
- Expected range: within 5% of reported

Calculate:
- Accuracy
- Sensitivity (true positive rate)
- Specificity (true negative rate)
- Confusion matrix

### Task 1.4: Documentation
Write results to: `/Users/cash/Desktop/fifi/Research-Paper/VALIDATION_RESULTS.md`

Include:
- Sample sizes (n per class)
- Performance metrics with 95% confidence intervals
- Any discrepancies and hypothesized causes
- Recommendation for proceeding to Phase 3

## SUCCESS CRITERION
Accuracy >= 80% on held-out set; metrics within 5% of reported values.

## FAILURE HANDLING
If accuracy < 80%, document:
- Potential causes (data drift, preprocessing differences, model versioning)
- Recommended remediation steps
- Do NOT proceed to Phase 3 until resolved
