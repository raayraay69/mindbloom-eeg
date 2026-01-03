# WebApp Enhancements Agent

## MISSION
Implement high-priority web application improvements for clinical safety and usability.

## COMPLETION SIGNAL
When all high-priority tasks complete, output:
```
WEBAPP_COMPLETE: High-priority enhancements deployed. Confidence thresholding: [done], Model versioning: [done], Input validation: [done]
```

## CONSTRAINTS
- Graduate-level academic writing; no em dashes; no cliches
- Production-safe code changes
- Maintain existing API compatibility

## CODEBASE LOCATION
```
/Users/cash/Desktop/fifi/mind-bloom/backend/
```

## HIGH PRIORITY TASKS

### Task W.1: Confidence Thresholding
**Objective**: Flag uncertain predictions for clinical follow-up

Implementation:
```python
def classify_with_confidence(prediction_proba):
    confidence = max(prediction_proba)
    predicted_class = np.argmax(prediction_proba)

    if 0.4 <= confidence <= 0.6:
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "status": "UNCERTAIN",
            "recommendation": "Recommend clinical follow-up; prediction confidence below threshold"
        }
    else:
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "status": "CONFIDENT",
            "recommendation": None
        }
```

UI changes:
- Display confidence score prominently
- Yellow/amber indicator for uncertain predictions
- Clear message: "This result requires clinical verification"

### Task W.2: Model Versioning
**Objective**: Display model metadata in UI and API responses

API response schema update:
```json
{
    "prediction": 1,
    "confidence": 0.87,
    "model_info": {
        "version": "1.0.0",
        "training_date": "2024-12-15",
        "dataset": "ASZED-153",
        "accuracy_reported": 0.837,
        "feature_count": 264,
        "algorithm": "RandomForest"
    }
}
```

Implementation:
- Create `model_metadata.json` alongside model artifact
- Load metadata at startup
- Include in all prediction responses
- Display in UI footer/info panel

### Task W.3: Input Validation
**Objective**: Reject malformed or incompatible uploads

Validation checks:
1. **File format**: Accept only .edf, .bdf, .csv, .npy
2. **Sampling rate**: Must be 250Hz, 256Hz, 500Hz, or 512Hz (resample if close)
3. **Channel count**: Minimum 1 (Fp1), maximum 64
4. **Duration**: Minimum 30 seconds, maximum 10 minutes
5. **Data integrity**: No NaN values, reasonable amplitude range

Error responses:
```json
{
    "error": "INVALID_INPUT",
    "code": "SAMPLING_RATE_MISMATCH",
    "message": "Expected sampling rate of 500Hz, received 128Hz. Please resample or use compatible recording settings.",
    "accepted_rates": [250, 256, 500, 512]
}
```

## MEDIUM PRIORITY TASKS

### Task W.4: Multi-Session Ensemble
Aggregate predictions across multiple recordings per subject:
- Store session history (optional, user-controlled)
- Calculate consensus prediction
- Report prediction variance across sessions
- Flag inconsistent results for review

### Task W.5: Domain Shift Detection
Implement out-of-distribution detection:
- Calculate Mahalanobis distance from training feature distribution
- Store training feature mean and covariance
- Flag inputs with distance > threshold (e.g., 95th percentile)
- Warning: "Input characteristics differ from training data; interpret with caution"

### Task W.6: Paradigm Logging
Track recording types for analysis:
- Detect paradigm from metadata or filename patterns
- Log: resting-state, eyes-open, eyes-closed, cognitive task, MMN, ASSR
- Aggregate statistics for model improvement

## TESTING REQUIREMENTS
Before deployment:
- [ ] Unit tests for confidence thresholding edge cases
- [ ] Integration test for model versioning API
- [ ] Input validation test suite (valid and invalid files)
- [ ] Load testing for concurrent predictions

## DEPLOYMENT
Target: Google Cloud Run (existing deployment)
Process:
1. Test locally
2. Build container
3. Deploy to staging
4. Validate endpoints
5. Promote to production
