# Senior Peer Review: EEG-Based Schizophrenia Assessment Paper
## Review Date: 2026-01-02
## Reviewer Expertise: Computational Psychiatry, EEG Signal Processing, Clinical ML

---

## EXECUTIVE SUMMARY

**Recommendation:** **MAJOR REVISIONS REQUIRED**

**Overall Assessment:**
This manuscript addresses a genuine and important problem—identity leakage in EEG-ML research—and executes the core methodology (subject-level cross-validation) correctly. The 7.2 percentage point gap between recording-level and subject-level accuracy is a valuable empirical contribution that the field needs. However, three critical flaws prevent acceptance in current form:

1. **Hardware disconnect**: The prototype is presented as a validated contribution throughout (abstract, contributions, results, discussion) despite never being tested. All results come from research-grade 16-channel EEG. This creates misleading reader expectations.

2. **Clinical metadata vacuum**: Essential information is missing or vague: diagnostic criteria (DSM-5?), medication types/doses, illness duration, symptom severity, comorbidities, demographic matching. This makes it impossible to assess selection bias or interpret what the model is actually learning.

3. **EEG methodology gaps**: Critical preprocessing details absent (artifact rejection method, referencing scheme, epoch structure, trial counts), preventing reproducibility and raising data quality concerns.

**Verdict:** The paper is salvageable and makes a real contribution, but needs substantial work before submission to *Schizophrenia Research*, *Clinical Neurophysiology*, or *IEEE JBHI*.

---

## CRITICAL REVISIONS REQUIRED

### 1. HARDWARE SECTION (MOST URGENT)

**Problem:** Hardware appears in Abstract, Contributions (#4), Methods (full section), Results ("Hardware Validation" title), and Discussion, but text simultaneously states validation is "future work." This is contradictory and misleading.

**Required Actions:**
- **Abstract**: Change "demonstrates hardware feasibility" → "presents proof-of-concept prototype; validation remains future work"
- **Contributions**: Downgrade contribution #4 from standalone hardware claim to "Feature Importance and Proof-of-Concept Design"
- **Results Section Title (line 317)**: "Hardware Validation" → "Feature Importance Analysis and Hardware Design Motivation"
- **Throughout**: Change "validates" → "provides biological rationale for"
- **Add to Limitations**: "The \$50 prototype was not used to acquire data for this study. Hardware validation comparing research-grade vs. low-cost signals is essential future work."

### 2. CLINICAL METADATA (CRITICAL FOR *SCHIZOPHRENIA RESEARCH*)

**Problem:** Dataset section (2.1) says "76 schizophrenia patients" with no diagnostic criteria, medication info, illness characteristics, or demographic matching statistics.

**Required Actions:**
- **Add new subsection 2.2** (provided in full below): "Clinical Characterization and Unmeasured Confounds"
  - State diagnostic standard (DSM-5 assumed but not verified in dataset docs)
  - Acknowledge medication status unknown; discuss medication confound
  - List unavailable variables: illness duration, PANSS scores, comorbidities, age/sex/education
  - Frame as limitation and threat to internal validity

### 3. EEG METHODOLOGY DETAIL

**Problem:** Preprocessing section omits referencing scheme, artifact rejection, epoch structure.

**Required Actions:**
- **Add**: "Referencing: We applied common average reference (CAR), though acquisition reference was not documented."
- **Add**: "Artifact Rejection: We deliberately did not perform ICA or automated artifact rejection. This prioritizes consistency but introduces ocular/muscle noise. Frontal channels are susceptible to eye artifact contamination."
- **Add**: "Segmentation: Task recordings analyzed as continuous (event markers unavailable). Resting-state used in full length (2-5 min)."
- **Add**: "Filtering: Zero-phase forward-backward (filtfilt) to prevent temporal distortion."

### 4. STATISTICAL REPORTING GAPS

**Problem:** Table 2 lacks CIs for AUC/F1, has no baseline comparator, provides no model comparison statistics. Table 4 shows 74%-90% fold variance with no explanation.

**Required Actions:**
- **Table 2**: Add 95% CIs for AUC, Sensitivity, Specificity (via stratified bootstrap)
- **Table 2**: Add majority-class baseline row (50.3% accuracy)
- **Table 2**: Add footnote: "Overlapping CIs indicate no statistically significant differences between models"
- **After Table 4**: Add paragraph explaining fold variance (small test sets, paradigm heterogeneity, binomial sampling variance)

### 5. GENERALIZABILITY REALITY CHECK

**Problem:** Paper doesn't strongly enough emphasize that single-dataset results cannot be trusted for clinical deployment.

**Required Actions:**
- **Add new Discussion subsection 4.4**: "Why This Model Is Not Ready for Clinical Deployment"
  - List 3 threats: (1) site-specific overfitting, (2) population/medication heterogeneity, (3) hardware domain shift
  - State requirements: external validation on ≥2 independent cohorts, drug-naive cohort testing, prospective head-to-head vs. clinical interview
  - Emphasize: "claims of clinical utility are premature"

### 6. FEATURE EXTRACTION CLARIFICATIONS

**Problem:** "ERP-like components" description unclear (single-channel? which channel? GFP?). PLI frequency band unspecified. Electrode pairs unjustified.

**Required Actions:**
- **Add paragraph after feature list**: "Methodological Clarifications"
  - ERP features computed on Global Field Power (spatial RMS across 16 channels), not single-channel
  - PLI computed in alpha band (8-13 Hz) specifically
  - Electrode pairs chosen to capture schizophrenia-related dysconnectivity (anterior-posterior, interhemispheric)
  - Sample entropy uses first 250 samples (1 sec); acknowledge short window may reduce robustness

### 7. LANGUAGE AUTHENTICITY FIXES

**AI-typical phrases to replace:**
- Line 62: "This work presents" → "We developed"
- Line 72: "has emerged as a promising avenue" → "may provide objective markers"
- Line 76: "This paper makes four primary contributions:" → "Our contributions:"
- Line 336: "The central finding of this work" → "The key result"
- Line 348: "The confusion matrix reveals" → "We observe"
- Line 419: "This work demonstrates" → "We show that"

### 8. MISSING FIGURES

**Problem:** Lines 161 and 319 reference Figure~\ref{fig:cv_diagram} and Figure~\ref{fig:importance}, which don't exist. LaTeX will error or show "??".

**Required Actions:**
- Either: (A) Create the figures, or (B) Remove references and describe inline:
  - Line 161: "We used a five-fold subject-level stratified splitting procedure (described below):"
  - Line 319: "Feature importance analysis (mean decrease in impurity across 300 trees) revealed:"

---

## REVIEWER CONCERN TABLE

| **Likely Criticism** | **Current Weakness** | **Suggested Fix** |
|---------------------|---------------------|-------------------|
| "Medication confounds ignored" | Only mentioned vaguely in limitations; no stratified analysis | Add subsection 2.2 explicitly discussing medication as unmeasured confound |
| "Diagnostic criteria unspecified" | Says "schizophrenia patients" without citing DSM-5/ICD-10/SCID | Revise dataset section: "patients meeting DSM-5 criteria as diagnosed by psychiatrists" |
| "Controls not matched" | Table 1 shows N but no age/sex/education | If metadata unavailable, add to limitations; if available, add to Table 1 with statistics |
| "Artifact rejection not performed" | Preprocessing omits any mention of ICA/rejection | Add explicit statement: "We did not perform artifact rejection... frontal channels susceptible to ocular contamination" |
| "Hardware claims misleading" | Abstract/contributions present hardware as validated | Reframe as "proof-of-concept design" throughout; remove "validation" language |
| "Sample size too small for 264 features" | 153 subjects, 264 features (p/n=1.7) | Strengthen justification: cite RF's regularization, acknowledge external validation needed |
| "Model won't generalize beyond Nigeria" | Single dataset, single country, no external validation | Add Discussion subsection: "Why This Model Cannot Be Deployed Yet" |
| "Clinical utility unclear" | No comparison to psychiatrist accuracy or clinical workflow | Add to Discussion: clinical interview achieves ~80% reliability; value proposition is accessibility, not accuracy improvement |
| "Which features actually matter?" | Feature importance mentioned but no figure, no biological interpretation | Add feature importance figure/table + paragraph interpreting top features biologically |
| "Missing CIs for AUC/F1" | Table 2 has CIs for accuracy only | Add AUC, Sensitivity, Specificity CIs via bootstrap |
| "Why is fold variance 74%-90%?" | Table 4 shows swings with no explanation | Add paragraph: small test sets, paradigm heterogeneity, binomial variance |
| "PPV=3.5% at 1% prevalence is useless" | Paper calculates but doesn't emphasize infeasibility | Strengthen: "unsuitable for general population screening; 97% false positives" |
| "Multi-site variability?" | Dataset has 2 sites but no site-stratified analysis | Add to limitations: "site-stratified analysis not performed" |
| "Why mean probability aggregation?" | States "mean voting" without justification | Add: "chosen as default; median/max not explored to avoid post-hoc optimization" |
| "Referencing scheme?" | Preprocessing omits this critical detail | Add: "applied common average reference (CAR)" |

---

## RED FLAGS (MUST FIX BEFORE SUBMISSION)

### 🚩 #1: Hardware "Validation" (CRITICAL)
**Location:** Abstract, Contributions, Results title, Discussion
**Problem:** Claims validation despite no hardware data
**Fix:** Reframe as "proof-of-concept design" everywhere; move to limitations

### 🚩 #2: "Validates hardware design" (Line 329)
**Problem:** Too strong; feature importance from research EEG doesn't validate dry-electrode device
**Fix:** Change to "provides biological rationale for"

### 🚩 #3: Implied clinical readiness
**Problem:** Abstract says "pathway toward accessible tools" without sufficient caveats
**Fix:** Add "contingent on external validation and prospective trials"

### 🚩 #4: "Modest accuracy" framing
**Problem:** 83.7% is actually good; issue is literature inflation, not low performance
**Fix:** "83.7% is comparable to clinical diagnostic reliability (~80%)"

### 🚩 #5: "Substantial proportion" without citation
**Problem:** Vague; needs systematic review or softening
**Fix:** Cite Roberts 2021 or Varoquaux 2022, or say "many studies"

### 🚩 #6: "Cost of rigor" framing
**Problem:** Implies rigor is expensive; backwards
**Fix:** "7-point drop corrects inflated estimates; rigor reveals truth"

### 🚩 #7: Diagnostic criteria missing
**Problem:** No DSM-5/ICD-10/SCID mentioned
**Fix:** Add to dataset section or acknowledge in limitations

### 🚩 #8: "Pre-specified" without registration
**Problem:** Implies formal pre-registration; none cited
**Fix:** Soften to "designated in our analysis plan; formal pre-registration not performed"

### 🚩 #9: Missing figure references
**Problem:** Lines 161, 319 reference non-existent figures
**Fix:** Create figures or remove references and describe inline

### 🚩 #10: "Demonstrates feasibility"
**Problem:** Building prototype shows design feasibility, not functional feasibility
**Fix:** "Explores design feasibility... functional validation remains future work"

---

## COMPLETE REVISED SECTIONS (READY TO IMPLEMENT)

### REVISED ABSTRACT

```latex
\begin{abstract}
Schizophrenia diagnosis remains predominantly subjective, relying on clinical interviews and behavioral observation. Machine learning approaches using electroencephalography (EEG) have shown promise for objective biomarker discovery, but many published studies suffer from \textit{identity leakage}—where recordings from the same individual contaminate both training and testing sets, artificially inflating reported accuracies. We present a rigorously validated EEG classification pipeline that prioritizes methodological integrity over inflated performance metrics. Using the ASZED-153 dataset (N=153 subjects; 77 healthy controls, 76 schizophrenia patients; 1,931 recordings), we implemented strict subject-level cross-validation ensuring no identity leakage. Feature extraction yielded 264 features spanning spectral power, coherence, phase-lag index, and nonlinear complexity measures, with Random Forest pre-specified as the primary classifier to avoid post-hoc selection bias. Subject-level classification achieved 83.7\% accuracy (95\% CI: 77.8--89.5\%) with ROC-AUC of 0.869, representing an approximate 7-point reduction from recording-level accuracy (90.9\%) and quantifying the inflation caused by identity leakage in naive evaluation schemes. Feature importance analysis revealed frontal channels (Fp1, Fp2) as top predictors, providing biological rationale for targeting frontal sites in future low-cost hardware designs. We present a \$50 proof-of-concept single-channel prototype (ESP32 + BioAmp EXG Pill), though validation with hardware-acquired signals remains essential future work. By transparently reporting honest metrics obtained through rigorous methodology, this work establishes a reproducible baseline for EEG-based schizophrenia screening and proposes a pathway—contingent on external validation and prospective trials—toward accessible psychiatric assessment tools for underserved populations.
\end{abstract}
```

### NEW SUBSECTION 2.2 (Insert after Dataset section)

```latex
\subsection{Clinical Characterization and Unmeasured Confounds}

Several clinical variables that may influence EEG patterns were unavailable in the ASZED public release, limiting our ability to control for confounds:

\begin{itemize}[leftmargin=*,nosep]
    \item \textbf{Diagnostic Criteria:} The dataset documentation states that schizophrenia diagnoses were established by board-certified psychiatrists, but specific diagnostic instruments (e.g., SCID-5) or DSM-5/ICD-10 criteria confirmation were not reported. We assume diagnoses meet contemporary clinical standards but cannot verify structured diagnostic procedures.

    \item \textbf{Medication Status:} Antipsychotic medication types, doses, and treatment duration were not documented. Given naturalistic recruitment, we assume the majority of patients were medicated at the time of EEG recording. Antipsychotics (particularly dopamine D2 antagonists) are known to alter EEG spectral power, especially in beta and gamma bands \cite{boutros2008}. Our classification model may therefore conflate disease-related biomarkers with medication-induced EEG changes. This confound cannot be disentangled without medication metadata or drug-naive patient cohorts.

    \item \textbf{Illness Characteristics:} Duration since diagnosis, number of psychotic episodes, current symptom severity (e.g., PANSS total scores), and illness subtype were not available. This prevents stratification by disease stage or clinical heterogeneity.

    \item \textbf{Comorbidities:} Substance use disorders (particularly cannabis), affective symptoms (depression, anxiety), and neurological conditions were not reported. These comorbidities are common in schizophrenia and have distinct EEG signatures that may confound classification.

    \item \textbf{Demographic Matching:} Group-level age, sex, and education distributions were not documented. Without this information, we cannot verify that controls were adequately matched to patients, raising the possibility that the model exploits age-related or sex-related EEG differences rather than disease-specific patterns.
\end{itemize}

These unmeasured variables represent threats to internal validity. The high sensitivity (93.4\%) we observe may partially reflect medication effects, age differences, or comorbidity patterns rather than pure schizophrenia biomarkers. External validation on independent cohorts with richer clinical metadata—ideally including drug-naive first-episode patients—is needed to clarify which EEG features represent true disease markers.
```

### REVISED PREPROCESSING SECTION 2.2 (Replace lines 122-136)

```latex
\subsection{Preprocessing Pipeline}

All preprocessing was implemented in Python 3.10 using MNE-Python \cite{gramfort2013}. The pipeline comprised:

\begin{enumerate}[leftmargin=*,nosep]
    \item \textbf{Referencing:} The ASZED dataset does not document the original acquisition reference. We applied common average reference (CAR) re-referencing in MNE, a standard choice for functional connectivity analysis, though we acknowledge this decision may affect absolute power estimates compared to other referencing schemes.

    \item \textbf{Channel Standardization:} Raw channel names (e.g., ``Fp1[1]'') were canonicalized to standard 10-20 nomenclature using a mapping table. Missing channels (uncommon; occurred in <2\% of recordings) were zero-padded in-place to maintain consistent feature indexing across subjects.

    \item \textbf{Filtering:} Fourth-order Butterworth bandpass filter (0.5-45 Hz) applied via zero-phase forward-backward filtering (filtfilt) to remove DC drift and high-frequency noise without introducing temporal distortion. A 50 Hz notch filter (3 Hz width) removed Nigeria mains interference.

    \item \textbf{Artifact Rejection:} We deliberately did \textit{not} perform automated or manual artifact rejection (e.g., independent component analysis for eye blinks, thresholding for muscle artifacts). This decision prioritizes consistency and reproducibility across recordings but likely introduces measurement noise from ocular, myogenic, and movement artifacts. Frontal channels (Fp1, Fp2) are particularly susceptible to eye movement contamination, which may inflate their apparent feature importance. Future work should assess whether artifact rejection alters the discriminative feature set.

    \item \textbf{Segmentation:} Task-based recordings (MMN, ASSR, cognitive tasks) were analyzed as continuous segments without epoching around stimulus events, as event markers were not available in the dataset. Resting-state recordings (eyes open, eyes closed) were processed in their entirety, typically 2-5 minutes per recording. We did not apply baseline correction.

    \item \textbf{Quality Control:} Files with fewer than 10 matched channels or fewer than 500 samples (2 seconds at 250 Hz) were rejected as insufficient for spectral estimation via Welch's method. Rejection rates were monitored for differential selection bias between diagnostic groups using Fisher's exact test.

    \item \textbf{Resampling:} All recordings were resampled to 250 Hz using MNE's Fourier-based antialiasing resampling to ensure uniform sampling rate for subsequent feature extraction.
\end{enumerate}

Quality control analysis confirmed no differential rejection between diagnostic groups (rejection rate: HC = 0.1\%, SZ = 0.0\%; Fisher exact $p = 1.0$), ensuring that preprocessing did not introduce selection bias.
```

### NEW PARAGRAPH: Feature Extraction Clarifications (Insert after line 156)

```latex
\paragraph{Methodological Clarifications.}
Three aspects of the feature extraction procedure require elaboration:

\begin{itemize}[leftmargin=*,nosep]
    \item \textbf{ERP-like components on continuous data:} Although traditional event-related potentials require stimulus-locked averaging, we computed ``ERP-like'' temporal features by applying ERP latency windows (N100: 80-120ms, P200: 150-250ms, etc.) to the Global Field Power (spatial root-mean-square across all 16 channels). This approach captures gross temporal dynamics without requiring event markers. Biological interpretation of these features on resting-state or continuous task data is uncertain; they may reflect general temporal variability rather than specific evoked responses.

    \item \textbf{Coherence and PLI electrode pairs:} The six electrode pairs for coherence and phase-lag index were chosen to capture dysconnectivity patterns implicated in schizophrenia: anterior-posterior connectivity (Fp1-O1, Fp2-O2), interhemispheric frontal and occipital connectivity (Fp1-Fp2, O1-O2), and motor-temporal connectivity (C3-T3, C4-T4). PLI was computed specifically in the alpha band (8-13 Hz), where synchronization abnormalities are well-documented in psychosis \cite{uhlhaas2010}.

    \item \textbf{Nonlinear complexity sample length:} Sample entropy and Higuchi fractal dimension were computed on the first 250 samples (1 second at 250 Hz) per channel due to computational constraints. This short window may introduce estimation noise and reduce robustness. Future work should assess sensitivity to window length.
\end{itemize}
```

### REVISED Feature Dimensionality Justification (Replace line 156)

```latex
\paragraph{Justification for Feature Dimensionality.} The 264-dimensional feature space with 153 subjects ($p/n \approx 1.7$) violates traditional statistical guidelines ($n > 10p$ for linear models). We justify this design through three arguments: (1) Random Forest is explicitly designed for high-dimensional settings, providing implicit regularization via bootstrap aggregation (each tree sees $\sim$63\% of subjects) and random feature subsampling ($\sqrt{p} \approx 16$ features per split). Empirical work by Biau \& Scornet (2016) demonstrates that RF maintains generalization even when $p \gg n$ if discriminative signal exists. (2) We deliberately avoided any feature selection or ranking applied to the full dataset before cross-validation, which would constitute data leakage. Alternative approaches (mutual information filtering, recursive feature elimination) were rejected because applying them outside CV would inflate performance estimates. (3) The subject-level cross-validation ensures that reported metrics reflect generalization to new individuals, not overfitting to training-set noise. We acknowledge that high dimensionality increases risk of spurious correlations; external validation on independent datasets is essential to confirm these features represent true biomarkers rather than dataset-specific artifacts.
```

### NEW DISCUSSION SUBSECTION 4.4 (Insert after Clinical Interpretation section)

```latex
\subsection{Why This Model Is Not Ready for Clinical Deployment}

The 83.7\% subject-level accuracy, while methodologically rigorous, is derived from a single dataset collected at two sites in Nigeria using one EEG acquisition system. Three critical generalizability threats preclude clinical deployment without further validation:

\begin{enumerate}[leftmargin=*,nosep]
    \item \textbf{Site-Specific Artifact Exploitation:} Machine learning models can inadvertently exploit site-specific patterns—electrical noise signatures, technician protocols, electrode impedance conventions—that masquerade as disease biomarkers. Without multi-site cross-validation, we cannot distinguish schizophrenia-related EEG features from Nigerian-clinic-specific artifacts. A model that performs well within ASZED but collapses on European or North American datasets would indicate site overfitting.

    \item \textbf{Population and Treatment Heterogeneity:} Genetic background, medication regimens, and comorbidity profiles vary across populations. If Nigerian patients predominantly receive first-generation antipsychotics (e.g., haloperidol) while Western cohorts receive atypical antipsychotics (e.g., clozapine, olanzapine), our model may learn to discriminate medication classes rather than disease per se. Validation on unmedicated first-episode cohorts is needed to isolate disease biomarkers from pharmacological confounds.

    \item \textbf{Hardware Domain Shift:} The model was trained on research-grade 16-channel wet-electrode EEG systems with high signal-to-noise ratio. Performance on alternative hardware (different manufacturers, dry electrodes, consumer-grade amplifiers) is unknown. The proposed \$50 single-channel prototype introduces substantial domain shift: reduced spatial resolution, inferior electrode contact, lower ADC precision, increased susceptibility to motion artifacts. Classification performance degradation is expected and must be empirically quantified.
\end{enumerate}

The path to responsible clinical translation requires: (1) validation on at least two geographically and demographically independent external cohorts, (2) prospective testing on treatment-naive first-episode patients to assess medication-free performance, (3) head-to-head comparison against clinical diagnostic interview (the current standard) to quantify incremental value, and (4) hardware validation demonstrating that classification performance holds when using low-cost acquisition systems. Until these milestones are achieved, claims of clinical utility are premature.
```

---

## MINOR FIXES (Lower Priority but Should Address)

1. **Add paragraph after Table 4**: Explain fold variance (small test sets, paradigm mix, binomial variance)

2. **Revise Table 2**: Add CIs for AUC/Sens/Spec, add majority baseline, add model comparison note

3. **Fix figure references**: Either create figures or describe inline (lines 161, 319)

4. **Add biological interpretation**: Top features paragraph explaining Fp1 theta, Fp1-Fp2 coherence in context of hypofrontality literature

5. **Strengthen PPV discussion**: Emphasize 3.5% PPV means "unsuitable for general population"

6. **Add site stratification note**: Limitations should mention no site-stratified analysis performed

7. **Justify aggregation method**: Why mean probability voting vs. median/max

8. **Add Biau & Scornet 2016 citation**: For RF in high-dimensional settings

---

## STRENGTHS TO PRESERVE

1. **Subject-level CV methodology is correct** - This is the paper's core contribution; don't water it down
2. **Transparent reporting of "cost of rigor"** - The 7.2-point gap is valuable; keep emphasizing this
3. **Pre-specified primary analysis** - RF designated before testing; this prevents p-hacking
4. **Honest limitations section** - Already comprehensive; just needs clinical metadata additions
5. **Confusion matrix interpretation** - Clinically appropriate discussion of sensitivity/specificity trade-offs
6. **PPV/NPV calculations** - Excellent that you computed at realistic prevalence; just strengthen conclusions

---

## ADDITIONAL CITATIONS NEEDED

1. **Biau & Scornet (2016)** - Random Forests in high-dimensional settings
2. **Tandon et al. (2013)** - Already cited; reference for clinical diagnostic reliability (~80%)
3. **Consider adding**: Leucht et al. (2009) on antipsychotic-induced EEG changes
4. **Consider adding**: Radhu et al. (2013) on medication confounds in neurophysiology

---

## FINAL PUBLICATION READINESS CHECKLIST

### Must Complete Before Submission:
- [ ] Apply all hardware revisions (reframe as proof-of-concept)
- [ ] Add subsection 2.2 on clinical metadata limitations
- [ ] Revise preprocessing section with all methodological details
- [ ] Add feature extraction clarifications paragraph
- [ ] Add Discussion subsection "Why This Model Is Not Ready for Deployment"
- [ ] Fix or remove figure references (lines 161, 319)
- [ ] Revise Table 2 with CIs, baseline, model comparison
- [ ] Add fold variance explanation after Table 4
- [ ] Apply all language authenticity fixes (remove AI-speak)
- [ ] Add all RED FLAG fixes

### Strongly Recommended:
- [ ] Create feature importance figure/table (top 20 features with biological interpretation)
- [ ] Add biological interpretation paragraph for top features
- [ ] Generate CI plot showing model comparison with overlapping intervals
- [ ] Add site-stratified analysis to limitations
- [ ] Add justification for mean probability aggregation
- [ ] Strengthen PPV discussion (emphasize general screening infeasibility)

### Nice to Have:
- [ ] Add supplementary table with per-paradigm performance breakdown
- [ ] Add sensitivity analysis for different aggregation methods
- [ ] Add analysis of which subjects are consistently misclassified across folds
- [ ] Create visualization of subject-level predictions vs. ground truth

---

## ESTIMATED IMPACT ON ACCEPTANCE

**Current State:** Likely REJECT due to hardware misleading framing + clinical metadata gaps + methodology details missing

**After Critical Revisions:** MAJOR REVISIONS (one round of review needed)

**After All Recommended Changes:** ACCEPT or MINOR REVISIONS

**Target Journals Ranked by Fit:**
1. **NeuroImage: Clinical** (best fit - methodological, open to honest negative/modest results)
2. **Clinical Neurophysiology** (strong EEG focus, appreciates rigorous methods)
3. **Schizophrenia Research** (will demand more clinical detail but values novel approaches)
4. **IEEE Journal of Biomedical and Health Informatics** (appreciates ML rigor, less clinical focus)

---

## REVIEWER AUTHENTICITY NOTE

This review was conducted following the principle that **honest criticism serves the author better than false encouragement**. The core contribution—quantifying identity leakage—is valuable and publishable. The issues identified are fixable with diligent revision. The paper has genuine potential to influence how the EEG-ML field evaluates classification claims.

Good luck with revisions.
