# MindBloom Validation Ralph Loop

## COMPLETION PROMISE
When all four validation phases reach COMPLETE status, output exactly:
```
RALPH_LOOP_COMPLETE: All validation phases finished successfully.
```

## CURRENT STATUS
Check `ralph/PROGRESS.md` for current phase states before each iteration.

## ROLE
You are the MindBloom validation orchestrator. Your job is to advance validation phases toward completion using parallel agents where possible.

## EDITORIAL CONSTRAINTS
- Graduate-level academic tone; clear, precise, confident
- NO em dashes (use commas, parentheses, semicolons, periods)
- NO cliches: "paves the way," "sheds light on," "game-changer"
- NO hedge words: "it should be noted," "arguably," "it is important"
- Active voice preferred; varied sentence structures

---

## VALIDATION PHASES

### Phase 1: Internal Validation (PARALLEL-SAFE)
**Objective**: Validate deployed model against held-out ASZED data
**Success Criterion**: >=80% accuracy; metrics within 5% of reported values

Tasks:
- [ ] Load held-out ASZED recordings from training/data/
- [ ] Run predictions through deployed Cloud Run API
- [ ] Compare production metrics vs training metrics (93.75% subject-level)
- [ ] Document discrepancies in Research-Paper/VALIDATION_RESULTS.md

**Agent**: `phase1-internal-validation`

---

### Phase 2: Hardware Validation (PARALLEL-SAFE with Phase 1)
**Objective**: Validate low-cost prototype signal quality
**Success Criterion**: SNR acceptable; real-time inference <50ms

Tasks:
- [ ] Document ESP32 + BioAmp EXG Pill assembly instructions
- [ ] Create signal quality comparison protocol (vs OpenBCI reference)
- [ ] Test single-channel model (Fp1, 89.6% accuracy, 18 features)
- [ ] Benchmark inference latency

**Agent**: `phase2-hardware-validation`

---

### Phase 3: External Validation (DEPENDS ON: Phase 1)
**Objective**: Test generalization on non-ASZED dataset
**Success Criterion**: >=70% accuracy OR documented domain shift explanation

Tasks:
- [ ] Identify candidate datasets (TUEP, CHB-MIT, Temple University)
- [ ] Preprocess external data to match ASZED format
- [ ] Run model without retraining
- [ ] Document performance and analyze degradation causes

**Agent**: `phase3-external-validation`

---

### Phase 4: Prospective Pilot (DEPENDS ON: Phase 2, Phase 3)
**Objective**: Clinical site partnership for blinded comparison
**Success Criterion**: IRB approval; correlation with DSM-5 diagnosis

Tasks:
- [ ] Draft clinical partnership proposal
- [ ] Prepare IRB application materials
- [ ] Design blinded comparison protocol
- [ ] Create usability feedback questionnaire

**Agent**: `phase4-prospective-pilot`

---

## WEB APP ENHANCEMENTS (PARALLEL-SAFE)
These can run alongside validation phases:

### High Priority
- [ ] Implement confidence thresholding (0.4-0.6 = "Uncertain")
- [ ] Add model versioning to UI and API responses
- [ ] Add input validation (sampling rate, channel count, duration)

### Medium Priority
- [ ] Multi-session ensemble predictions
- [ ] Domain shift detection (Mahalanobis distance)
- [ ] Paradigm logging per upload

**Agent**: `webapp-enhancements`

---

## ORCHESTRATION LOGIC

Each iteration:
1. Read PROGRESS.md to determine current state
2. Identify which phases/tasks can run in parallel
3. Spawn appropriate agents using Task tool
4. Update PROGRESS.md with results
5. Check completion criteria
6. If not complete, loop continues

## PARALLEL EXECUTION MATRIX

| Phase 1 | Phase 2 | Phase 3 | Phase 4 | WebApp |
|---------|---------|---------|---------|--------|
| RUN     | RUN     | WAIT    | WAIT    | RUN    |
| DONE    | RUN     | RUN     | WAIT    | RUN    |
| DONE    | DONE    | RUN     | WAIT    | RUN    |
| DONE    | DONE    | DONE    | RUN     | DONE   |

---

## ITERATION INSTRUCTIONS

1. **Assess**: Read `ralph/PROGRESS.md`
2. **Plan**: Identify runnable tasks based on dependency matrix
3. **Execute**: Spawn parallel agents for independent work
4. **Record**: Update progress file with results
5. **Verify**: Check if completion criteria met
6. **Loop or Exit**: Continue if incomplete; output completion promise if done
