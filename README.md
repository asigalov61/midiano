# midiano
## Comprehensive Python package for MIDI music anomalies detection and features analysis

<img width="1024" height="1024" alt="midiano" src="https://github.com/user-attachments/assets/71dd0e2b-ec10-4b34-94bf-169fbc0c252c" />

***

## Installation

```sh
!pip install midiano
```

***

## Use as follows

### Quick Start

```python
from midiano import MIDIFeatureExtractor, MIDIProblemDetector

# Extract all features from a MIDI file
extractor = MIDIFeatureExtractor('song.mid')
features = extractor.extract_all()

# Detect concrete problems in the MIDI file
detector = MIDIProblemDetector(features)
problems = detector.detect()

# Print detected problems
for p in problems:
    print(f"[{p['severity'].upper()}] {p['category']}: {p['issue']} — {p['detail']}")
```

***

## Classes

### `MIDIFeatureExtractor`

Extracts a comprehensive set of **200+ features** from a MIDI file across 12 categories, designed for downstream anomaly detection and quality assessment.

```python
from midiano import MIDIFeatureExtractor

# Basic usage
extractor = MIDIFeatureExtractor('song.mid')

# Filter to a specific MIDI channel (e.g., channel 0 only)
extractor = MIDIFeatureExtractor('song.mid', filter_channel=0)

# Extract all features as a flat dictionary
features = extractor.extract_all()
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `midi_filepath` | `str` | required | Path to the MIDI file |
| `filter_channel` | `int` or `None` | `None` | If set, only extract notes from this MIDI channel |

**Key Properties:**

| Property | Type | Description |
|---|---|---|
| `filename` | `str` | Basename of the MIDI file |
| `parse_error` | `str` or `None` | Error message if parsing failed |
| `notes` | `list` | All note events sorted by (start, duration, pitch) |
| `tempos` | `list` | Tempo change events |
| `time_sigs` | `list` | Time signature events |
| `key_sigs` | `list` | Key signature events |
| `patches` | `dict` | Patch changes per track |
| `control_changes` | `list` | Control change events |
| `pitch_bends` | `list` | Pitch wheel events |
| `ticks_per_quarter` | `int` | Ticks per quarter note from the file header |

#### Feature Categories

`extract_all()` returns a flat dictionary with features prefixed by category:

**1. Structural / Metadata (`struct_`)**
File size, track count, note count, duration, tempo change count, time signature changes, empty track detection, note density, file efficiency.

```python
features = MIDIFeatureExtractor('song.mid').extract_all()

features['struct_total_notes']           # Total number of notes
features['struct_track_count']           # Number of tracks
features['struct_duration_secs']         # Duration in seconds
features['struct_notes_per_second']      # Note density
features['struct_empty_track_ratio']     # Ratio of empty tracks
features['struct_nonstandard_tpq']       # 1 if ticks/quarter is unusual
features['struct_bytes_per_note']        # File size efficiency
```

**2. Problem Detection (`prob_`)**
Zero-duration notes, velocity-zero notes, out-of-range pitches, negative durations, overlapping same-pitch notes, stuck notes, extreme bursts, tempo anomalies, excessive time-sig changes.

```python
features['prob_zero_duration_ratio']         # Ratio of zero-duration notes
features['prob_velocity_zero_ratio']          # Ratio of velocity-zero notes
features['prob_out_of_range_pitch_count']     # Pitches outside 0–127
features['prob_negative_duration_count']      # Notes with negative duration
features['prob_overlapping_same_pitch_ratio'] # Overlapping same-pitch/channel
features['prob_stuck_note_ratio']             # Notes held >16 beats
features['prob_extreme_burst_count']          # Onsets with >20 simultaneous notes
features['prob_single_velocity']              # 1 if all notes share one velocity
features['prob_all_max_velocity']             # 1 if all notes at velocity 127
```

**3. Instrument (`instruments_`)**
Unique instrument count, channel count, drum vs. melodic ratio, polytimbral flag, channel entropy.

```python
features['instruments_unique_count']      # Distinct (channel, patch) pairs
features['instruments_channel_count']     # Active MIDI channels
features['instruments_drum_note_ratio']   # Ratio of channel-10 notes
features['instruments_is_polytimbral']    # 1 if multiple instruments
features['instruments_channel_entropy']   # Shannon entropy of channel distribution
```

**4. Tempo & Pace (`tempo_`, `pace_`)**
BPM statistics (mean, std, min, max, range, median, IQR, skew, kurtosis, etc.), tempo change dynamics, notes per beat/bar.

```python
features['tempo_bpm_mean']           # Average BPM
features['tempo_bpm_range']          # BPM max - min
features['tempo_bpm_change_max']     # Largest single tempo change
features['pace_notes_per_beat']      # Note density per beat
features['pace_notes_per_bar']       # Note density per bar
```

**5. Grid & Rhythm (`grid_`)**
Quantization offset, start/duration alignment ratios, quarter/eighth/sixteenth note alignment, swing ratio, syncopation ratio, rhythmic entropy.

```python
features['grid_start_alignment_ratio']   # How well notes snap to grid
features['grid_on_quarter_ratio']        # Quarter-note alignment
features['grid_on_eighth_ratio']         # Eighth-note alignment
features['grid_swing_ratio']             # Swing feel detection
features['grid_syncopation_ratio']       # Off-beat note ratio
features['grid_rhythmic_entropy']        # Rhythmic complexity
```

**6. Harmonic & Key (`harmonic_`)**
Key estimation via Krumhansl-Schmuckler, key clarity/strength, pitch-class entropy, consonance/dissonance ratios, interval classification.

```python
features['harmonic_key_tonic']           # Estimated key (0=C, 1=C#, …)
features['harmonic_key_is_minor']        # 1 if minor, 0 if major
features['harmonic_key_clarity']         # Correlation with key profile
features['harmonic_key_strength']        # Gap between top-2 key candidates
features['harmonic_pitch_class_entropy'] # High = atonal
features['harmonic_consonance_ratio']    # Consonant interval ratio
features['harmonic_dissonance_ratio']    # Dissonant interval ratio
```

**7. Note (`delta_times_`, `durations_`, `pitches_`, `velocities_`, `pitch_intervals_`)**
Full statistical summaries (mean, std, min, max, range, median, Q25, Q75, IQR, skew, kurtosis, MAD, CV, entropy) for inter-onset intervals, durations, pitches, velocities, and melodic pitch intervals. Also includes pitch-class distribution, direction changes, step/leap ratios, and velocity dynamics.

```python
features['durations_mean']               # Average note duration
features['pitches_range']                # Pitch span
features['pitch_intervals_mean']         # Average melodic interval
features['pitches_step_ratio']           # Ratio of steps (≤2 semitones)
features['pitches_leap_ratio']           # Ratio of leaps (>4 semitones)
features['pitches_ascending_ratio']      # Upward melodic motion ratio
features['delta_times_zero_ratio']       # Simultaneous onset ratio
features['delta_times_nonzero_mean']     # Avg non-zero inter-onset interval
features['pitch_class_0_ratio']          # C note ratio (0–11 for C–B)
```

**8. Chord (`chords_`)**
Chord count, chord ratio, chord size statistics, uniqueness ratio, triad quality analysis (major, minor, diminished, augmented, sus2, sus4) — with correct inversion handling.

```python
features['chords_count']            # Number of simultaneous note groups (≥2)
features['chords_ratio']            # Chord onsets / total onsets
features['chords_size_mean']        # Average notes per chord
features['chords_major_ratio']      # Major triad ratio
features['chords_minor_ratio']      # Minor triad ratio
features['chords_uniqueness_ratio'] # Harmonic variety
```

**9. Polyphony (`polyphony_`)**
Max/mean polyphony (time-weighted), monophonic ratio, overlap ratio, multi-channel flag.

```python
features['polyphony_max']                # Maximum simultaneous voices
features['polyphony_mean']               # Time-weighted average polyphony
features['polyphony_monophonic_ratio']   # Ratio of time with ≤1 voice
features['polyphony_overlap_ratio']      # Notes overlapping with others
features['polyphony_multi_channel']      # 1 if multiple channels used
```

**10. Phrase (`phrases_`)**
Phrase segmentation via gap detection, phrase count, note count statistics, phrase length statistics.

```python
features['phrases_count']            # Number of detected phrases
features['phrases_avg_note_count']   # Average notes per phrase
features['phrases_length_mean']      # Average phrase length in ticks
```

**11. Dynamics (`dynamics_`)**
Velocity range, unique velocity count, dynamic range in dB, velocity entropy, consecutive same-velocity ratio, short-note ratio.

```python
features['dynamics_velocity_range']       # Max velocity - min velocity
features['dynamics_range_db']             # Dynamic range in decibels
features['dynamics_entropy']              # Velocity distribution entropy
features['dynamics_same_consecutive_ratio']  # Flat dynamics indicator
```

**12. Quality / Completeness (`quality_`)**
Presence of tempo/time-sig/key-sig metadata, default-value detection, sparsity/density flags, pitch range flags.

```python
features['quality_has_tempo']           # 1 if tempo events exist
features['quality_has_time_sig']        # 1 if time signature exists
features['quality_default_tempo']       # 1 if tempo ≈ 120 BPM default
features['quality_default_time_sig']    # 1 if time sig is 4/4 default
features['quality_very_sparse']         # 1 if <0.5 notes/second
features['quality_very_dense']          # 1 if >100 notes/second
features['quality_very_narrow_range']   # 1 if pitch range <1 octave
```

---

### `MIDIProblemDetector`

Rule-based detector that identifies concrete MIDI problems — corruption, malformation, encoding errors, and quality issues — from extracted features.

```python
from midiano import MIDIFeatureExtractor, MIDIProblemDetector

features = MIDIFeatureExtractor('song.mid').extract_all()
detector = MIDIProblemDetector(features)
problems = detector.detect()
```

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `features` | `dict` | Feature dictionary from `MIDIFeatureExtractor.extract_all()` |

**`detect()` Returns:**

A list of problem dictionaries, each containing:

| Key | Type | Description |
|---|---|---|
| `category` | `str` | `'corruption'`, `'structure'`, or `'quality'` |
| `issue` | `str` | Short problem identifier |
| `severity` | `str` | `'critical'`, `'warning'`, or `'info'` |
| `detail` | `str` | Human-readable description with values |

**Detected Problems:**

| Category | Issue | Severity Thresholds |
|---|---|---|
| **corruption** | `parse_error` | critical — file failed to parse |
| **corruption** | `no_notes` | critical — no note events found |
| **corruption** | `zero_duration_notes` | critical (>50%), warning (>10%), info (>1%) |
| **corruption** | `velocity_zero_notes` | critical (>30%), warning (>5%) |
| **corruption** | `out_of_range_pitches` | critical — any pitch outside 0–127 |
| **corruption** | `negative_durations` | critical — any negative duration |
| **corruption** | `overlapping_same_pitch` | warning (>30%), info (>10%) |
| **corruption** | `stuck_notes` | warning — notes held >16 beats |
| **corruption** | `extreme_burst` | warning (>50), info (>20 simultaneous onsets) |
| **structure** | `nonstandard_tpq` | info — unusual ticks per quarter |
| **structure** | `excessive_tempo_changes` | warning (>10/s), info (>2/s) |
| **structure** | `extreme_tempo` | warning — BPM <20 or >300 |
| **structure** | `large_tempo_range` | warning — >200 BPM variation |
| **structure** | `excessive_time_sig_changes` | warning — >10 time-sig changes |
| **structure** | `many_empty_tracks` | warning (>70%), info (>40%) |
| **quality** | `no_dynamics` | warning — single velocity value |
| **quality** | `all_max_velocity` | warning — all notes at velocity 127 |
| **quality** | `atonal` | warning — key clarity <0.3 |
| **quality** | `weak_tonality` | info — key clarity <0.5 |

```python
# Filter by severity
critical = [p for p in problems if p['severity'] == 'critical']
warnings = [p for p in problems if p['severity'] == 'warning']

# Filter by category
corruption = [p for p in problems if p['category'] == 'corruption']

# Pretty-print all problems
for p in problems:
    icon = {'critical': '🔴', 'warning': '🟡', 'info': '🔵'}[p['severity']]
    print(f"{icon} [{p['severity'].upper()}] {p['issue']}: {p['detail']}")
```

---

### High-Precision Timing Anomaly Detector (`analyze_midi_timings`)

While `MIDIProblemDetector` focuses on **structural corruption and rule-based quality issues**, the timing analyzer performs **voice-aware, grid-residual outlier detection** tailored to expressive human performances. It is robust to rubato, tempo changes, and graceful ornaments, and only flags IOIs whose residuals are strong statistical outliers relative to the performer's own baseline — without misclassifying musical expressivity as anomalies.

```python
from midiano import (
    analyze_midi_timings,
    print_summary_report,
    print_top_anomalies,
    print_per_voice_report,
    plot_timing_analysis,
)

# Run analysis with progress bars and verbose diagnostics
report = analyze_midi_timings(
    "song.mid",
    max_subdivision=16,
    verbose=True,
    show_progress_bar=True,
)

# Text reports
print_summary_report(report)
print_top_anomalies(report, top_n=20)
print_per_voice_report(report)

# Visual report (two stacked subplots)
plot_timing_analysis(report)
```

**Function Signatures:**

| Function | Description |
|---|---|
| `analyze_midi_timings(midi_path, max_subdivision=16, verbose=False, show_progress_bar=False)` | Parse MIDI and run voice-aware grid-residual anomaly detection. Returns a report dict. |
| `print_summary_report(report, top_n=20)` | Print high-level summary (notes, voices, IOI stats, anomaly count). |
| `print_top_anomalies(report, top_n=20)` | Print the most extreme flagged anomalies ranked by `|residual|`. |
| `print_per_voice_report(report)` | Print per-voice anomaly counts and detection thresholds. |
| `plot_timing_analysis(report, max_ioi_quantile=0.95, figsize=(15, 9))` | Plot the IOI timeline + residual signal with anomaly markers. Returns the matplotlib `Figure`. |

**`analyze_midi_timings` Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `midi_path` | `str` | required | Path to the `.mid` file. |
| `max_subdivision` | `int` | `16` | Upper bound on the integer grid-multiple considered plausible. |
| `verbose` | `bool` | `False` | If `True`, prints tempo-map, voice, and threshold diagnostics. |
| `show_progress_bar` | `bool` | `False` | If `True`, wraps internal loops with `tqdm` progress bars. |

**Report Dictionary Keys:**

| Key | Type | Description |
|---|---|---|
| `file` | `str` | MIDI file path. |
| `num_notes` | `int` | Total note count. |
| `num_iois` | `int` | Voice-filtered IOI count. |
| `num_voices` | `int` | Distinct `(track, channel)` voice count. |
| `estimated_subdivision` | `int` | Estimated grid steps per quarter. |
| `expected_grid_ms_nominal` | `float` | Nominal grid duration at the initial tempo (ms). |
| `mean_ioi_ms` | `float` | Mean IOI across all voices. |
| `median_ioi_ms` | `float` | Median IOI. |
| `std_ioi_ms` | `float` | Population std-dev of IOIs. |
| `mad_ioi_ms` | `float` | Robust MAD of IOIs. |
| `anomalies_all` | `list[dict]` | All flagged anomalies, sorted by `|residual|` descending. |
| `anomalies_by_voice` | `dict` | `{(track, channel): list[anomaly]}`. |
| `thresholds_by_voice` | `dict` | `{(track, channel): thresholds}`. |
| `voice_iois` | `dict` | `{(track, channel): list[ioi_record]}`. |
| `notes` | `list[dict]` | All notes with computed `start_ms`, `duration_ms`. |
| `tick2ms` | `callable` | Tick-to-ms closure honouring the tempo map. |
| `ticks_per_quarter` | `int` | PPQ resolution from the file header. |
| `tempo_changes` | `list[tuple]` | Sorted, de-duplicated `(tick, tempo_us)` pairs. |

**Detection Logic:**

Each IOI is expressed as `ratio = ioi_ms / local_grid_ms`, where the local grid is evaluated at the IOI's origin tick (so tempo changes are honoured). Ratios outside `[0.75, max_subdivision]` are skipped (grace notes, long rests). The remaining residuals:

```
residual_ms = ioi_ms - round(ratio) * local_grid_ms
```

form a per-voice distribution characterised by **median** and **MAD** (robust to expressive timing). An IOI is flagged if either test fires:

| Test | Condition | Reason Tag |
|---|---|---|
| **z-outlier** | `|z| > z_thresh` AND `|residual| > min_abs_for_z` | `'z_outlier'` |
| **local jitter** | rolling 8-IOI std exceeds both `jitter_abs_ms` and `jitter_rel × local_grid`, AND `|residual| > abs_dev_ms` | `'local_jitter'` |

**Default thresholds:**

| Parameter | Default | Description |
|---|---|---|
| `z_thresh` | `4.0` | Robust z-score threshold for outlier flagging. |
| `min_abs_for_z` | `35.0` ms | Minimum residual magnitude required to fire a z-outlier. |
| `jitter_abs_ms` | `40.0` ms | Absolute std threshold for the local-jitter test. |
| `jitter_rel` | `0.20` | Relative std threshold (`× local_grid`). |
| `abs_dev_ms` | `50.0` ms | Minimum residual magnitude required when the local-jitter path fires. |

**Plot Interpretation (`plot_timing_analysis`):**

The figure contains two stacked subplots sharing a time axis:

* **Top — IOI timeline.** Each IOI is a scatter dot **colored by its nearest grid multiple** (`viridis` colormap with colorbar). The continuous local grid duration is drawn as a **step line through every IOI** (so tempo/rubato contour is visible). Horizontal dashed lines mark `k × nominal_grid` for `k = 1..8` with inline labels. Flagged anomalies are overlaid as **red open circles**.
* **Bottom — Residual signal.** The signed residual `ioi_ms − nearest_int × local_grid` over time, with the **±z·MAD threshold band** shaded in red. Anomalies are marked with **red `x`** symbols. This subplot makes the detector's outlier logic visually self-evident.

```python
# Customize the plot
fig = plot_timing_analysis(report, max_ioi_quantile=0.99, figsize=(16, 10))

# Access flagged anomalies directly
for a in report['anomalies_all'][:10]:
    voice = a['voice']
    print(f"{voice} ioi={a['ioi_ms']:.1f}ms grid={a['local_expected_ms']:.1f}ms "
          f"mult={a['nearest_int']} resid={a['residual_ms']:+.1f}ms "
          f"z={a.get('z', 0):.2f} reasons={a['reasons']}")

# Iterate per-voice thresholds
for voice, th in report['thresholds_by_voice'].items():
    print(f"{voice}: med_res={th['med_residual']:.2f}ms "
          f"mad_res={th['mad_residual']:.2f}ms n={th['n_iois_in_voice']}")
```

**Why this complements `MIDIProblemDetector`:**

| Aspect | `MIDIProblemDetector` | `analyze_midi_timings` |
|---|---|---|
| **Goal** | Find encoding corruption and quality issues. | Find timing outliers in expressive performances. |
| **Method** | Rule-based thresholds on aggregate features. | Per-voice robust statistics on grid residuals. |
| **Output** | Categorical problem list (critical/warning/info). | Ranked anomaly records with residual/z/jitter values. |
| **Sensitive to** | Malformed MIDI, missing metadata, encoding bugs. | Erratic timing, sustained local jitter, mistimed onsets. |
| **Robust to** | Rubato, tempo changes, ornaments. | Same — performance-adaptive by design. |

---

### Batch Processing

```python
from midiano import MIDIFeatureExtractor, MIDIProblemDetector
import os, json

midi_dir = 'path/to/midi/files'
results = []

for fname in tqdm.tqdm(os.listdir(midi_dir)):
    if not fname.endswith(('.mid', '.midi')):
        continue
    path = os.path.join(midi_dir, fname)
    
    extractor = MIDIFeatureExtractor(path)
    features = extractor.extract_all()
    
    if 'error' not in features:
        detector = MIDIProblemDetector(features)
        problems = detector.detect()
        features['problem_count'] = len(problems)
        features['critical_count'] = sum(1 for p in problems if p['severity'] == 'critical')
    
    results.append(features)

# Save features as JSON
with open('features.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
```

### Anomaly Detection with Isolation Forest

```python
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from midiano import MIDIFeatureExtractor

# Collect features from multiple MIDI files
feature_dicts = []
for path in midi_file_paths:
    f = MIDIFeatureExtractor(path).extract_all()
    if 'error' not in f:
        feature_dicts.append(f)

# Convert to DataFrame and clean
df = pd.DataFrame(feature_dicts).set_index('filename')
df = df.dropna(axis=1, how='any')          # drop columns with NaN
df = df.select_dtypes(include='number')     # keep only numeric columns

# Scale and detect anomalies
scaler = StandardScaler()
X = scaler.fit_transform(df)

iso = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = iso.fit_predict(X)  # -1 = anomaly, 1 = normal

# Show flagged files
anomalies = df[df['anomaly'] == -1]
print(f"Flagged {len(anomalies)} anomalous MIDI files:")
print(anomalies.index.tolist())
```

***

## Attribution

* midiano was passionately vibe-coded with [Z AI](https://chat.z.ai/)
* Artwork is a courtesy of [Microsoft Copilot](https://copilot.microsoft.com/)

***

### Project Los Angeles
### Tegridy Code 2026