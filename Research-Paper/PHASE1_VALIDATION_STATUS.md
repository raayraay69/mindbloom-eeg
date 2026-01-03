# Phase 1: Internal Validation Status

**Date:** January 3, 2026
**Status:** READY FOR EXECUTION (pending held-out data preparation)

## Current Findings

### Test Data Availability

**Status:** No dedicated held-out test set found in repository.

The training pipeline (documented in `training/ASZED_Training_Colab.ipynb`) uses 5-fold stratified cross-validation at the subject level rather than maintaining a separate held-out test set. This approach has benefits (all data used for training and validation) but creates challenges for Phase 1 internal validation.

**Evidence:**
- Training notebook implements `subject_level_cv()` function with `StratifiedKFold(n_splits=5)`
- No separate test files (`.edf`, `.bdf`, `.csv`) found in repository
- Model evaluation was performed through cross-validation, not hold-out testing

### Implications for Validation Roadmap

The absence of a dedicated held-out test set means Phase 1 internal validation requires one of the following approaches:

#### Option 1: Create Held-Out Set from ASZED-153 (Recommended)
1. Access the full ASZED-153 dataset (currently not in repository)
2. Perform subject-level stratified split (e.g., 80/20 train/test)
3. Retrain model on 80% subset
4. Validate on 20% held-out subjects
5. Compare performance metrics with cross-validation results

**Advantages:**
- True generalization estimate on unseen subjects
- Maintains methodological rigor (subject-level separation)
- Provides baseline for external validation comparison

**Requirements:**
- Access to full ASZED-153 dataset
- Retraining time (estimated 30-60 minutes)
- Update deployed model with retrained version

#### Option 2: Cross-Validation Verification (Interim)
1. Reproduce cross-validation procedure from training notebook
2. Verify that reported metrics (83.7% accuracy, 93.4% sensitivity, 74.0% specificity) can be replicated
3. Document consistency between original training and current reproduction

**Advantages:**
- Feasible with current repository contents
- Validates reproducibility of training pipeline
- No need for external data access

**Limitations:**
- Does not test generalization to truly unseen subjects
- Cannot detect overfitting beyond what cross-validation reveals
- Less rigorous than dedicated hold-out validation

#### Option 3: Deploy-and-Monitor Approach
1. Deploy current model with comprehensive logging
2. Collect predictions on real-world uploads
3. Retrospectively analyze performance when ground truth becomes available

**Advantages:**
- Tests model in realistic deployment conditions
- Captures domain shift effects not present in training data

**Limitations:**
- Requires clinical partnerships for ground truth labels
- Long timeline to collect sufficient data
- Ethical considerations for using unvalidated model

## Recommended Next Steps

### Immediate Actions (This Session)
1. ✅ **Implement high-priority web app enhancements:**
   - ✅ Confidence thresholding (flag predictions 0.4-0.6 as uncertain)
   - ✅ Model versioning in API responses
   - ✅ Enhanced input validation with detailed error codes

2. ✅ **Documentation improvements:**
   - ✅ Remove em dashes from research paper
   - ✅ Create model metadata file

3. 📝 **Create internal validation plan:**
   - Document test data gap
   - Propose Option 1 (held-out set creation) as primary path
   - Identify who has access to ASZED-153 raw data

### Medium-Term Actions (Next 2-4 Weeks)
1. **Obtain ASZED-153 dataset access:**
   - Contact dataset authors or institutional repository
   - Verify data use permissions and ethics approval

2. **Implement Option 1 (if dataset accessible):**
   - Create subject-level 80/20 split
   - Retrain model on training subset
   - Validate on held-out 20%
   - Document results in `VALIDATION_RESULTS.md`

3. **Alternative: Implement Option 2 (if dataset inaccessible):**
   - Reproduce cross-validation from training notebook
   - Verify metric consistency
   - Document limitations in validation report

### Long-Term Actions (Phase 2-4)
1. **Phase 2: Hardware Validation**
   - Order ESP32 + BioAmp components
   - Validate signal quality against research-grade equipment
   - Test single-channel Fp1 model

2. **Phase 3: External Validation**
   - Identify independent EEG schizophrenia datasets
   - Test model generalization without retraining
   - Document domain shift effects

3. **Phase 4: Prospective Pilot**
   - Partner with clinical site
   - Design IRB-approved blinded comparison study
   - Collect usability feedback

## Success Criteria for Phase 1 Completion

Phase 1 will be considered **COMPLETE** when:
- ✅ High-priority web app enhancements deployed
- ⏳ Held-out test set created (Option 1) OR cross-validation reproduced (Option 2)
- ⏳ Validation metrics documented in `VALIDATION_RESULTS.md`
- ⏳ Accuracy ≥ 80% achieved on validation set
- ⏳ Metrics within 5% of reported cross-validation results
- ⏳ Discrepancies (if any) analyzed and explained

## Current Status Summary

| Task | Status | Blocker |
|------|--------|---------|
| Web app enhancements | ✅ COMPLETE | None |
| Research paper review | ✅ COMPLETE | None |
| Test data identification | ✅ COMPLETE | None (gap documented) |
| Held-out set creation | ⏳ PENDING | ASZED-153 dataset access required |
| Validation execution | ⏳ BLOCKED | Awaiting test data |
| Results documentation | ⏳ BLOCKED | Awaiting validation execution |

**Overall Phase 1 Status:** IN PROGRESS (40% complete)

**Critical Blocker:** Access to full ASZED-153 dataset for held-out test set creation.

**Recommended Action:** Team lead (Samiksha BC) to coordinate with dataset authors or check institutional data repository access.
