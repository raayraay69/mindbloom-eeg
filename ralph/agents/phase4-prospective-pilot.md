# Phase 4 Agent: Prospective Pilot Study

## MISSION
Prepare clinical partnership and IRB materials for blinded prospective validation.

## COMPLETION SIGNAL
When all tasks complete, output:
```
PHASE4_COMPLETE: Prospective pilot materials ready. IRB draft: [status], Clinical site: [identified/contacted]
```

## DEPENDENCY
**BLOCKED UNTIL**: Phase 2 AND Phase 3 complete

## CONSTRAINTS
- Graduate-level academic writing; no em dashes; no cliches
- IRB-compliant language and ethical considerations
- Realistic scope for pilot study (n=30-50)

## TASKS

### Task 4.1: Clinical Partnership Proposal
Create: `/Users/cash/Desktop/fifi/Research-Paper/CLINICAL_PROPOSAL.md`

Target sites:
- University psychiatric clinics
- Community mental health centers
- VA hospitals with psychiatry departments

Proposal contents:
1. Executive summary (1 page)
2. Scientific background and preliminary results
3. Proposed collaboration structure
4. Resource requirements (space, staff time, equipment)
5. Timeline and milestones
6. Data sharing and publication agreements
7. Contact information and next steps

Value proposition for clinical partners:
- No-cost screening tool evaluation
- Co-authorship on validation publication
- Early access to validated technology
- Contribution to mental health equity research

### Task 4.2: IRB Application Materials
Create: `/Users/cash/Desktop/fifi/Research-Paper/IRB_MATERIALS/`

Required documents:
- Protocol summary (study design, procedures, risks/benefits)
- Informed consent template
- HIPAA authorization form
- Data security plan
- Adverse event reporting procedures

Study design:
- **Type**: Prospective, blinded comparison
- **Population**: Adults (18-65) presenting for psychiatric evaluation
- **Sample size**: n=50 (25 schizophrenia spectrum, 25 controls)
- **Primary outcome**: Concordance between AI prediction and DSM-5 diagnosis
- **Secondary outcomes**: Usability, clinician acceptance, time-to-screen

Inclusion criteria:
- Age 18-65
- Able to provide informed consent
- Willing to undergo 5-minute EEG recording

Exclusion criteria:
- Active substance intoxication
- Neurological conditions affecting EEG (epilepsy, TBI)
- Inability to sit still for recording duration

### Task 4.3: Blinded Protocol Design
Create: `/Users/cash/Desktop/fifi/Research-Paper/PILOT_PROTOCOL.md`

Blinding procedure:
1. EEG recorded by trained technician (blinded to diagnosis)
2. AI prediction generated automatically (stored separately)
3. Clinical diagnosis made independently by psychiatrist
4. Unblinding occurs only after all subjects enrolled

Data collection:
- EEG recording (raw data, AI prediction, confidence)
- Demographics (age, sex, education)
- Clinical data (diagnosis, symptom severity if applicable)
- Recording quality metrics

Analysis plan:
- Primary: Sensitivity, specificity, PPV, NPV
- Secondary: ROC curve, optimal threshold identification
- Exploratory: Subgroup analysis by demographics

### Task 4.4: Usability Questionnaire
Create: `/Users/cash/Desktop/fifi/Research-Paper/USABILITY_QUESTIONNAIRE.md`

For clinicians (post-use):
1. System Usability Scale (SUS) - 10 items
2. Perceived usefulness (5-point Likert)
3. Integration feasibility (open-ended)
4. Concerns about AI-assisted screening (open-ended)

For participants (post-recording):
1. Comfort during EEG recording (5-point scale)
2. Willingness to use in future (yes/no/maybe)
3. Concerns about AI analysis (open-ended)

## SUCCESS CRITERION
- Complete IRB application package ready for submission
- At least one clinical site identified and contacted
- All protocol documents reviewed for ethical compliance

## TIMELINE CONSIDERATIONS
IRB review typically requires 4-8 weeks. Materials should be submission-ready, not dependent on immediate approval for Phase 4 completion.
