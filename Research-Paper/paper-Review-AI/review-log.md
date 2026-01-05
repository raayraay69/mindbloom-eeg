# Research Paper Review Log

**Paper Title:** A System-Level Framework for EEG-Based Schizophrenia Assessment: Methodological Rigor, Uncertainty Quantification, and Hardware Feasibility

**Authors:** Samiksha B. Chandrasekaran, Eric Raymond

**Review Date:** 2025-12-29

---

## Summary

This paper presents an EEG-based schizophrenia classification pipeline emphasizing methodological integrity over inflated metrics. Uses ASZED-153 dataset (N=153 subjects, 1,931 recordings) with strict subject-level cross-validation. Achieves 83.7% accuracy with ROC-AUC of 0.869.

---

## Strengths

### Methodology
| Item | Status | Notes |
|------|--------|-------|
| Subject-level cross-validation | STRONG | Correctly prevents identity leakage |
| Pre-specified primary model (RF) | STRONG | Prevents p-hacking/model shopping |
| Quality control bias analysis | STRONG | Fisher exact test confirms no differential rejection |
| Transparent accuracy comparison | STRONG | 7.2-point gap quantifies inflation from naive methods |
| Confidence intervals reported | STRONG | 95% CI: 77.8-89.5% for primary metric |

### Writing & Presentation
| Item | Status | Notes |
|------|--------|-------|
| Paper organization | STRONG | Clear, logical structure |
| Appropriate hedging | STRONG | Honest about limitations |
| Clinical interpretation | STRONG | Confusion matrix analysis is grounded |
| Limitations section | STRONG | Comprehensive and honest |

### Scientific Contribution
| Item | Status | Notes |
|------|--------|-------|
| Addresses real problem (identity leakage) | STRONG | Important contribution to field |
| Reproducibility focus | STRONG | Public dataset, clear methods |
| Feature importance validation | STRONG | Frontal channels match neuroscience literature |

---

## Weaknesses / Concerns

### Major Issues

| Issue | Severity | Status | Recommendation |
|-------|----------|--------|----------------|
| Hardware section disconnected | HIGH | NEEDS REVISION | Abstract + Contributions + Methods + Results (research.tex:56-62, 83-90, 188-199, 299-310, 344-357) still imply the prototype was validated on ASZED data even though no hardware data exists; rewrite as feasibility demonstration and confine validation claims to Limitations 383-389. |
| Recording vs Subject gap interpretation | MEDIUM | NEEDS REVISION | Discussion 318-323 still frames the 7.2-pt delta purely as "identity leakage"; needs nuance that part of the drop comes from harder cross-subject generalization and paradigm heterogeneity. |
| Feature dimensionality concerns | MEDIUM | NEEDS REVISION | Feature extraction lists 264 features (research.tex:142-156) for 153 subjects with no feature selection, regularization, or control for overfitting; add rationale or dimensionality reduction. |
| Referenced figures missing | HIGH | NEEDS FIX | Figure callouts for the CV diagram and feature-importance plot (research.tex:161, 299) have no corresponding figure environments; either add the figures or remove the references. |

### Minor Issues

| Issue | Severity | Status | Recommendation |
|-------|----------|--------|----------------|
| Missing CIs for AUC/F1 in Table 2 | LOW | NEEDS REVISION | Table 2 (research.tex:216-230) still lists only point estimates; add 95% CI columns once code outputs them. |
| High per-fold variance (74.2%-90.0%) | LOW | NEEDS DISCUSSION | Table 4 (research.tex:275-295) shows a 16-pt swing yet Discussion never explains the weak folds; add narrative (e.g., paradigm mix, subject count). |
| No model comparison statistics | LOW | NEEDS REVISION | Provide statistical tests or overlapping-CI commentary to justify RF beating LR/SVM in Table 2 (research.tex:216-230). |
| Missing majority-class baseline | LOW | NEEDS REVISION | Table 2 lacks the ~50.3% majority-class baseline or other chance comparator for context. |
| ERP feature description lacks Global Field Power detail | LOW | NEEDS CLARIFICATION | Feature extraction bullet (research.tex:147-148) still reads like per-channel ERPs; explicitly say the windows operate on spatially averaged (GFP) signals. |
| PLI frequency band unspecified | LOW | NEEDS CLARIFICATION | Phase-lag index bullet (research.tex:149-151) omits which band the PLI was computed over. |
| Electrode pair selection arbitrary | LOW | NEEDS JUSTIFICATION | Need rationale for the six electrode pairs listed for coherence/PLI (research.tex:149-151). |
| Minor file count discrepancy | LOW | TRIVIAL | Dataset section (research.tex:104) says 1,931 recordings but QC (research.tex:208) references 1,932 files; reconcile. |
| Medication confound discussion | LOW | RESOLVED | Limitations now explicitly mention medication effects (research.tex:379-389); no further action unless we run stratified analyses. |

---

## Open Questions for Authors

1. How was the 0.5 classification threshold chosen? Was threshold optimization considered?
2. Why 5-fold CV specifically? Pre-specified or post-hoc?
3. "ERP-like components" on resting-state EEG—what's the biological interpretation?
4. Were paradigm types (eyes open, MMN, ASSR) used as features or analyzed separately?

---

## Technical Accuracy Verification

| Claim | Verified | Notes |
|-------|----------|-------|
| Subject-level CV prevents identity leakage | YES | Methodology is correct |
| Fisher exact test for selection bias | YES | Appropriate statistical test |
| 264 features breakdown (80+20+30+6+96+32) | YES | Math checks out |
| Sensitivity calculation (71/76=93.4%) | YES | Correct |
| Specificity calculation (57/77=74.0%) | YES | Correct |
| CI interpretation | YES | Reasonable for N=153 |

---

## Revision Tracking

### Changes Made
| Date | Section | Change Description | Status |
|------|---------|-------------------|--------|
| 2025-12-29 | Paper review | Re-read `research.tex`; logged remaining edits (hardware framing, figures, metrics) | IN PROGRESS |

### Changes Rejected
| Date | Suggested Change | Reason for Rejection |
|------|-----------------|---------------------|
| | | |

---

## Overall Verdict

**Recommendation:** Accept with minor revisions

**Rationale:** The paper makes a genuine contribution by demonstrating and quantifying the identity leakage problem in EEG-ML research. The core methodology is sound. This is the kind of "honest science" paper the field needs.

**Priority Revisions:**
1. Downgrade hardware section to "future work" or remove
2. Nuance the interpretation of the 7.2-point gap
3. Address feature dimensionality concerns
4. Add confidence intervals to all metrics
5. Clarify "Global Field Power" for ERP components (vs channel-specific)
6. Justify or revisit the 1-second (250 sample) limit for entropy calculation
7. Report a majority-class baseline, add model-comparison statistics, and explain the weak CV folds before finalizing Table 2
8. Add the missing CV diagram and feature-importance figure (or drop the callouts)

---

## Code Review Analysis (complete.py v2.2.8)

### Consistency with Paper
| Feature | Code Status | Paper Claim | Notes |
|---------|-------------|-------------|-------|
| Subject-level CV | **VERIFIED** | Matches | `subject_stratified_cv_splits` correctly isolates subjects |
| Model (Random Forest) | **VERIFIED** | Matches | `n_estimators=300, max_depth=20` matches text |
| Feature Count (264) | **VERIFIED** | Matches | Code generates exactly 264 features |
| Hardware Validation | **MISSING** | Claimed | Code calculates features but **does not output feature importance rankings** to verify the "Fp1 is top feature" claim. |

### Technical Findings & Bugs
1.  **Entropy Sample Size Truncation:**
    - *Code:* `d = data[ch][:ENTROPY_SAMPLE_SIZE]` (where `ENTROPY_SAMPLE_SIZE = 250`)
    - *Issue:* The code only uses the **first 1 second** of data for Sample Entropy and Fractal Dimension. This is likely too short to be robust and may introduce noise. The paper does not mention this truncation.
    - *Recommendation:* Increase sample size or process full recording (or use sliding windows).

2.  **ERP "Global" Averaging:**
    - *Code:* `avg = np.mean(data, axis=0)`
    - *Issue:* ERP features are extracted from the *average of all 16 channels*.
    - *Recommendation:* Explicitly state in the paper that these are "Global Field Power" or "spatial mean" logic events, as "ERP-like components" usually implies channel-specific analysis in absence of qualifiers.

3.  **Missing CI for AUC/F1:**
    - *Code:* `bootstrap_ci` is only called for Accuracy in `print_final_summary`.
    - *Issue:* Paper Table 2 lacks CIs for AUC/F1. Code supports calculating them but doesn't print them.
    - *Recommendation:* Update code to print CIs for all metrics to support Table 2 revision.

---

## Notes

- 2025-12-29: research.tex audit shows hardware is still portrayed as validated throughout Abstract/Contributions/Results/Discussion; rewrite those passages as feasibility demonstration and keep validation claims confined to Limitations.
- 2025-12-29: No figures exist for the referenced CV diagram or feature-importance plot; either generate the assets or remove the `Figure~\ref{fig:...}` callouts.
- 2025-12-29: Table 2 still lacks majority baseline, AUC/F1 CIs, and any statistical significance discussion; once code emits the extra CIs, update the table + text accordingly.
- 2025-12-29: Feature extraction text needs clarifications (GFP ERP description, PLI band, electrode-pair rationale) plus a short paragraph about the 264 vs 153 dimensionality.
- 2025-12-29: Discussion should mention that some of the 7.2-pt drop comes from harder cross-subject generalization, not only leakage, and fold-level variance deserves a short narrative.


## AI Solver Prompt

(Copy and paste this prompt to an AI assistant to apply the required fixes)

---

You are an expert AI researcher and engineer. I have a research paper (`research.tex`) and its accompanying code (`complete.py`). A detailed review has identified several discrepancies and areas for improvement. Please perform the following tasks to align the code and paper and improve quality:

**1. Code Fixes (`complete.py`):**
*   **Fix Entropy Calculation:** The current code truncates data to only 250 samples (1 second) for Sample Entropy and Fractal Dimension features (`ENTROPY_SAMPLE_SIZE = 250`). This is too short for robust analysis. Please update the code to either use the full recording length or implement a more robust windowing strategy (e.g., 4-second epochs) to improve reliability.
*   **Output Feature Importance:** The paper claims that feature importance analysis identified "Fp1" as a top predictor, validating the hardware design. However, the code currently **does not output validation metrics** to support this. Please update the `save_results` or `print_final_summary` function to extract and print the top 20 feature importances from the trained Random Forest model.
*   **Add Confidence Intervals:** The paper reports 95% Confidence Intervals for Accuracy, but incorrectly lacks them for AUC and F1 in Table 2. The code has a `bootstrap_ci` function but only uses it for Accuracy. Please update the code to calculate and print 95% CIs for **AUC and F1 scores** as well.

**2. Paper Revisions (`research.tex`):**
*   **Downgrade Hardware Claims:** The hardware prototype was not used to collect the data for the main classification results (which came from the research-grade ASZED dataset). Please revise the "Hardware" sections (Abstract, Introduction, Discussion) to clearly label this as a "feasibility study" or "proof of concept" rather than implying the results validate the hardware directly.
*   **Clarify ERP/Connectivity Methodology:** The code calculates "ERP-like components" by averaging **all 16 channels** (Global Field Power). Please revise the "Feature Extraction" section to spell this out, explain why those six electrode pairs were chosen for coherence/PLI, and note the frequency band(s) used for PLI.
*   **Refine "Accuracy Drop" Interpretation:** Ensure the Abstract and Discussion frame the drop between recording-level (90.9%) and subject-level (83.7%) accuracy as a positive demonstration of scientific rigor (quantifying identity leakage) rather than just a performance failure. The current text is mostly good but double-check the tone.
*   **Update Table 2 Context:** Once the code outputs CIs for AUC/F1, update the table to include them, add the ~50.3% majority-class baseline, and briefly discuss whether RF meaningfully outperforms LR/SVM as well as why Fold 2/3 underperform.
*   **Restore Referenced Figures:** Either add the CV diagram and feature-importance figure that are currently referenced (Figure~\ref{fig:cv_diagram}, Figure~\ref{fig:importance}) or remove the callouts to avoid LaTeX errors.

Please apply these changes to `complete.py` and `research.tex` now.
