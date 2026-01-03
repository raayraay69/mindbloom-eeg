# EEG Data Validation Improvements

## Overview
This document describes the comprehensive validation and quality assessment system added to the MindBloom EEG analysis platform.

## What Was Added

### 1. Backend Enhancements (`mind-bloom/backend/main.py`)

#### Signal Quality Assessment
- **SNR Calculation**: Computes Signal-to-Noise Ratio in dB using power spectral density
  - Signal band: 0.5-45 Hz (EEG range)
  - Noise band: >45 Hz
  - Returns SNR in decibels

- **Per-Channel Quality Assessment**:
  - Quality score (0-1) based on SNR and variance
  - Detection of zero/missing channels
  - Detection of noisy channels (high variance or low SNR)
  - Individual channel status tracking

- **Overall Signal Quality Metrics**:
  - Overall quality score (0-1)
  - Channels found vs. expected count
  - Zero channel count
  - Noisy channel count
  - Average SNR across all channels
  - Preprocessing filter success/failure tracking

#### Enhanced Preprocessing
- Updated `preprocess()` function to return filter status
- Tracks success/failure of:
  - DC removal (always succeeds)
  - Bandpass filter (0.5-45 Hz)
  - Notch filter (50 Hz line noise)
- Logs filter failures with specific error messages

#### New API Endpoints

**`/validate` Endpoint**:
- Validates EEG files without making predictions
- Returns detailed validation information:
  - File format and metadata
  - Channel status for all 16 expected channels
  - Signal quality metrics
  - Validation errors and warnings
  - Preview data (first 2 seconds)

**Enhanced `/predict` Endpoint**:
- Now returns comprehensive validation details alongside predictions
- Provides specific error messages with channel names
- Includes warnings for data quality issues
- Logs all validation events

#### Comprehensive Logging
- File-based logging (`eeg_validation.log`)
- Console logging for real-time monitoring
- Logs include:
  - File validation start/completion
  - Channel matching details
  - Filter success/failure
  - Quality scores
  - Prediction results
  - Errors with full stack traces

#### Detailed Error Messages
Before:
```
"Too few channels matched (8/16)"
```

After:
```
"Too few channels matched: 8/16 found (need at least 10).
Found: Fp1, Fp2, F3, F4, C3, C4, P3, P4.
Missing: O1, O2, F7, F8, T3, T4, T5, T6"
```

### 2. Frontend Enhancements

#### Type Definitions (`src/lib/types.ts`)
Added comprehensive types:
- `ChannelStatus`: Individual channel quality and detection status
- `SignalQuality`: Overall signal quality metrics
- `ValidationDetails`: Complete validation information
- Updated `AnalysisResult` to include validation data

#### Updated Actions (`src/app/(app)/dashboard/actions.ts`)
- `performAnalysis()` now accepts FormData with the actual file
- Calls real backend API endpoint
- Maps backend response to frontend types
- Proper error handling with specific messages

#### Upload Component Updates (`components/eeg-upload.tsx`)
- Accepts both .EDF and .BDF formats
- Sends actual file to backend via FormData
- Displays validation warnings via toast notifications
- Better error messages from backend

#### New Validation Details Component (`components/validation-details.tsx`)
Displays comprehensive validation information:

**Overall Signal Quality Card**:
- Quality score with progress bar
- Color-coded quality levels (green/yellow/red)
- Channels found/expected counts
- Zero and noisy channel counts
- Average SNR
- Preprocessing filter status badges

**Recording Information Card**:
- File format (EDF/BDF)
- Duration in seconds
- Original sampling rate
- Resampling indicator
- Total samples

**Channel Status Grid**:
- 16-channel grid with status for each channel
- Color-coded status:
  - Green: Good quality (Q >70%)
  - Yellow: Low quality or noisy
  - Gray: Missing/not found
- Individual quality scores
- Visual icons for status

**Validation Alerts**:
- Error alerts for critical issues
- Warning alerts for non-critical issues
- Detailed lists of all errors/warnings

#### Enhanced Analysis View (`components/analysis-view.tsx`)
- Collapsible "Data Quality & Validation Details" section
- Shows full validation information after analysis
- Integrates seamlessly with existing results display

## Quality Scoring System

### Channel Quality Score (0-1)
- Based on SNR: Good (>10dB) = 1.0, Poor (<-10dB) = 0.0
- Penalized for high variance (>100 µV)
- Zero channels = 0.0
- Missing channels = 0.0

### Overall Quality Score (0-1)
- Average of individual channel quality scores
- Multiplied by channel coverage penalty
- Example: 12/16 channels found = 0.75 multiplier

### Quality Thresholds
- **High Quality**: Score ≥ 0.7 (Green)
- **Medium Quality**: 0.4 ≤ Score < 0.7 (Yellow)
- **Low Quality**: Score < 0.4 (Red with warning)

## Validation Rules

### Critical Errors (Reject File)
1. Invalid file format (not .EDF or .BDF)
2. Recording too short (< 2 seconds)
3. Too few channels (< 10 of 16 expected)

### Warnings (Accept with Notice)
1. Missing channels (but ≥10 found)
2. Noisy channels detected
3. Low overall signal quality (< 0.3)
4. Filter failures (bandpass or notch)
5. Resampling performed

## User-Facing Improvements

### Before
- Generic error: "File upload error"
- No visibility into data quality
- No channel status information
- Silent filter failures

### After
- Specific errors: "Too few channels matched: 8/16 found (need at least 10). Found: Fp1, Fp2..."
- Visual quality score with progress bar
- Per-channel status grid showing exactly what was found
- Filter status badges showing which preprocessing steps succeeded
- Validation warnings for quality issues
- Expandable validation details panel

## Logging Examples

### Successful Validation
```
2026-01-02 10:15:32 - INFO - Validating file: sample.edf
2026-01-02 10:15:33 - INFO - File loaded: 19 channels, 120.5s duration, 256Hz
2026-01-02 10:15:33 - INFO - Validation complete: PASSED, Quality: 0.82, Channels: 15/16
```

### Failed Validation
```
2026-01-02 10:20:15 - ERROR - Validation failed for bad_file.edf: Recording too short: 1.2s (need at least 2 seconds)
```

### Filter Failure
```
2026-01-02 10:25:45 - WARNING - Notch filter failed: invalid sampling rate
```

## Benefits

1. **Transparency**: Users see exactly what's wrong with their data
2. **Debugging**: Detailed logs help identify systematic issues
3. **Quality Assurance**: Automatic quality scoring identifies problematic recordings
4. **User Confidence**: Clear quality metrics build trust in results
5. **Actionable Feedback**: Specific messages help users fix issues
6. **Research Integrity**: Track data quality for scientific validity

## Future Enhancements

Potential additions (not yet implemented):
- Signal preview visualization (plot first 2 seconds of data)
- Real-time validation during upload
- Downloadable validation reports
- Quality trends over multiple sessions
- Automatic artifact detection
- Advanced quality metrics (e.g., frequency band power ratios)

## API Documentation

### POST /validate
Validates EEG file without prediction.

**Request**: `multipart/form-data` with file

**Response**:
```json
{
  "validation": {
    "file_format": "EDF",
    "original_sampling_rate": 256.0,
    "resampled": true,
    "duration_seconds": 120.5,
    "total_samples": 30125,
    "channels_in_file": ["Fp1", "Fp2", ...],
    "channel_statuses": [...],
    "signal_quality": {...},
    "validation_passed": true,
    "validation_errors": [],
    "validation_warnings": ["Resampled from 256Hz to 250Hz"]
  },
  "preview": {
    "channel_names": [...],
    "sample_data": [...],
    "sampling_rate": 250.0,
    "time_points": [...]
  }
}
```

### POST /predict
Performs prediction with validation.

**Response** includes all validation details plus:
```json
{
  "success": true,
  "prediction": "No Schizophrenia Indicators",
  "probability": 0.3456,
  "risk_level": "Low",
  "confidence": 0.6544,
  "channels_matched": 15,
  "recording_length_seconds": 120.5,
  "validation": {...}
}
```

## Configuration

Environment variables:
- `NEXT_PUBLIC_API_URL`: Backend API URL (default: `http://localhost:8000/api`)

Log file location:
- `backend/eeg_validation.log`
