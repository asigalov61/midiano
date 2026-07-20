#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#
#	MIDI Ano Python Module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2026
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
#
###################################################################################
###################################################################################
#
#   Copyright 2026 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
'''

###################################################################################
###################################################################################

print('=' * 70)
print('Loading midiano Python module...')
print('Please wait...')
print('=' * 70)

__version__ = '1.0.0'

print('midiano module version', __version__)
print('=' * 70)

###################################################################################
###################################################################################

from . import MIDI
import numpy as np
from scipy import stats
from collections import Counter, defaultdict, deque
import math
import os
import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import statistics
import matplotlib.pyplot as plt

# Optional progress-bar dependency (graceful fallback)
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

###################################################################################

class MIDIFeatureExtractor:
    """
    Extracts a comprehensive set of features from a MIDI file.
    Designed for downstream problematic MIDI detection and anomaly detection.

    MIDI.py score format:
        score[0] = ticks_per_quarter
        score[1:] = list of tracks
        Each track = list of events
        Note event: ['note', start, duration, channel, pitch, velocity]
        Tempo event: ['set_tempo', time, microseconds_per_quarter]
        Time sig: ['time_signature', time, nn, dd, ccc, bb]
        Key sig: ['key_signature', time, key_name]
        Patch: ['patch_change', time, channel, patch]
        Control: ['control_change', time, channel, controller, value]
        Pitch wheel: ['pitch_wheel', time, channel, value]
    """

    MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    # Interval class by semitone difference % 12
    # 0=Unison, 1=m2/M7, 2=M2/m7, 3=m3/M6, 4=M3/m6, 5=P4/P5, 6=Tritone
    INTERVAL_CLASSES = [0, 1, 2, 1, 3, 4, 2, 5, 2, 4, 3, 1]

    STANDARD_TPQ = {24, 48, 60, 72, 96, 120, 192, 240, 480, 960, 1920}

    def __init__(self, midi_filepath, filter_channel=None):
        self.midi_filepath = midi_filepath
        self.filename = os.path.basename(midi_filepath)
        self.filter_channel = filter_channel
        self.score = None
        self.ticks_per_quarter = None
        self.notes = []
        self.tempos = []
        self.time_sigs = []
        self.key_sigs = []
        self.patches = {}
        self.control_changes = []
        self.pitch_bends = []
        self.track_note_counts = {}
        self.parse_error = None
        self.file_size_bytes = 0

        self._load_and_parse()

    # =================================================================
    # LOADING & PARSING
    # =================================================================
    def _load_and_parse(self):
        try:
            with open(self.midi_filepath, 'rb') as f:
                midi_data = f.read()
            self.file_size_bytes = len(midi_data)
            self.score = MIDI.midi2score(midi_data)
            self.ticks_per_quarter = self.score[0]

            for itrack, track in enumerate(self.score[1:]):
                track_notes = 0
                for event in track:
                    etype = event[0]
                    if etype == 'note':
                        if self.filter_channel is None or event[3] == self.filter_channel:
                            self.notes.append(event)
                            track_notes += 1
                    elif etype == 'set_tempo':
                        self.tempos.append(event)
                    elif etype == 'time_signature':
                        self.time_sigs.append(event)
                    elif etype == 'key_signature':
                        self.key_sigs.append(event)
                    elif etype == 'patch_change':
                        self.patches.setdefault(itrack, []).append(event)
                    elif etype == 'control_change':
                        self.control_changes.append(event)
                    elif etype == 'pitch_wheel':
                        self.pitch_bends.append(event)
                self.track_note_counts[itrack] = track_notes

            self.notes.sort(key=lambda x: (x[1], x[2], x[4]))
        except Exception as e:
            self.parse_error = str(e)

    # =================================================================
    # STATISTICAL HELPERS
    # =================================================================
    def _get_stat_dict(self, arr, prefix):
        """Compute comprehensive statistics for a numeric array."""
        features = {}
        stat_names = ['mean', 'std', 'min', 'max', 'range', 'median', 'q25', 'q75',
                      'iqr', 'skew', 'kurtosis', 'mad', 'cv', 'entropy']

        if len(arr) == 0:
            for s in stat_names:
                features[f'{prefix}_{s}'] = np.nan
            return features

        arr = np.asarray(arr, dtype=float)
        features[f'{prefix}_mean'] = float(np.mean(arr))
        features[f'{prefix}_std'] = float(np.std(arr))
        features[f'{prefix}_min'] = float(np.min(arr))
        features[f'{prefix}_max'] = float(np.max(arr))
        features[f'{prefix}_range'] = float(np.max(arr) - np.min(arr))
        features[f'{prefix}_median'] = float(np.median(arr))
        features[f'{prefix}_q25'] = float(np.percentile(arr, 25))
        features[f'{prefix}_q75'] = float(np.percentile(arr, 75))
        features[f'{prefix}_iqr'] = float(features[f'{prefix}_q75'] - features[f'{prefix}_q25'])
        features[f'{prefix}_mad'] = float(np.median(np.abs(arr - np.median(arr))))

        if features[f'{prefix}_std'] > 0:
            features[f'{prefix}_skew'] = float(stats.skew(arr))
            features[f'{prefix}_kurtosis'] = float(stats.kurtosis(arr))
            mean_val = features[f'{prefix}_mean']
            # FIX: Use threshold instead of exact zero check to avoid
            # meaningless extreme CV values when mean is near zero
            features[f'{prefix}_cv'] = float(features[f'{prefix}_std'] / mean_val) if abs(mean_val) > 1e-10 else np.nan
        else:
            features[f'{prefix}_skew'] = 0.0
            features[f'{prefix}_kurtosis'] = 0.0
            features[f'{prefix}_cv'] = 0.0

        int_arr = np.round(arr).astype(int)
        features[f'{prefix}_entropy'] = self._entropy(int_arr.tolist())
        return features

    def _entropy(self, sequence):
        """Shannon entropy of a discrete sequence."""
        if not sequence:
            return np.nan
        counts = Counter(sequence)
        total = len(sequence)
        probs = [c / total for c in counts.values()]
        # FIX: max(0.0, ...) prevents -0.0 from floating-point arithmetic
        return max(0.0, -sum(p * math.log2(p) for p in probs if p > 0))

    def _ticks_to_seconds(self, max_tick):
        """Convert a tick position to wall-clock seconds using the tempo map."""
        if not self.ticks_per_quarter:
            return 0.0
        tempo_map = {0: 500000}  # default 120 BPM
        for t in self.tempos:
            tempo_map[t[1]] = t[2]

        sorted_times = sorted(tempo_map.keys())
        duration_secs = 0.0
        prev_time = 0
        current_tempo = 500000

        for t in sorted_times:
            if t > max_tick:
                break
            delta_ticks = t - prev_time
            duration_secs += (delta_ticks * current_tempo) / (self.ticks_per_quarter * 1e6)
            current_tempo = tempo_map[t]
            prev_time = t

        delta_ticks = max_tick - prev_time
        if delta_ticks > 0:
            duration_secs += (delta_ticks * current_tempo) / (self.ticks_per_quarter * 1e6)
        return duration_secs

    # =================================================================
    # 1. STRUCTURAL / METADATA FEATURES
    # =================================================================
    def _get_structural_features(self):
        features = {}
        features['struct_parse_error'] = 1 if self.parse_error else 0
        features['struct_file_size_bytes'] = self.file_size_bytes
        features['struct_ticks_per_quarter'] = self.ticks_per_quarter or 0
        features['struct_nonstandard_tpq'] = 0 if self.ticks_per_quarter in self.STANDARD_TPQ else 1
        features['struct_track_count'] = len(self.score) - 1 if self.score else 0
        features['struct_total_notes'] = len(self.notes)
        features['struct_tempo_change_count'] = len(self.tempos)
        features['struct_time_sig_change_count'] = len(self.time_sigs)
        features['struct_key_sig_change_count'] = len(self.key_sigs)
        features['struct_control_change_count'] = len(self.control_changes)
        features['struct_pitch_bend_count'] = len(self.pitch_bends)

        # Empty tracks
        if self.track_note_counts:
            empty = sum(1 for c in self.track_note_counts.values() if c == 0)
            features['struct_empty_track_count'] = empty
            features['struct_empty_track_ratio'] = empty / len(self.track_note_counts)
        else:
            features['struct_empty_track_count'] = 0
            features['struct_empty_track_ratio'] = np.nan

        # Notes per track distribution
        if self.track_note_counts:
            features.update(self._get_stat_dict(
                list(self.track_note_counts.values()), 'struct_notes_per_track'))

        # Duration & density
        if self.notes:
            max_tick = max(n[1] + n[2] for n in self.notes)
            features['struct_duration_ticks'] = max_tick
            dur_s = self._ticks_to_seconds(max_tick)
            features['struct_duration_secs'] = dur_s
            features['struct_notes_per_second'] = len(self.notes) / dur_s if dur_s > 0 else np.nan
        else:
            features['struct_duration_ticks'] = 0
            features['struct_duration_secs'] = 0.0
            features['struct_notes_per_second'] = np.nan

        # Tempo changes per second
        dur = features.get('struct_duration_secs', 0)
        features['struct_tempo_changes_per_second'] = (
            len(self.tempos) / dur if dur and dur > 0 else np.nan)

        # File efficiency
        if self.notes:
            features['struct_bytes_per_note'] = self.file_size_bytes / len(self.notes)
        else:
            features['struct_bytes_per_note'] = np.nan

        return features

    # =================================================================
    # 2. PROBLEM DETECTION FEATURES (corruption / malformation)
    # =================================================================
    def _get_problem_features(self):
        features = {}
        nan_fill = {
            'prob_zero_duration_count': 0, 'prob_zero_duration_ratio': np.nan,
            'prob_velocity_zero_count': 0, 'prob_velocity_zero_ratio': np.nan,
            'prob_out_of_range_pitch_count': 0, 'prob_out_of_range_pitch_ratio': np.nan,
            'prob_negative_duration_count': 0, 'prob_negative_duration_ratio': np.nan,
            'prob_overlapping_same_pitch_count': 0, 'prob_overlapping_same_pitch_ratio': np.nan,
            'prob_stuck_note_count': 0, 'prob_stuck_note_ratio': np.nan,
            'prob_very_short_duration_count': 0, 'prob_very_short_duration_ratio': np.nan,
            'prob_very_long_duration_count': 0, 'prob_very_long_duration_ratio': np.nan,
            'prob_extreme_pitch_count': 0, 'prob_extreme_pitch_ratio': np.nan,
            'prob_unique_velocity_count': 0, 'prob_single_velocity': 0,
            'prob_all_max_velocity': 0, 'prob_all_min_velocity': 0,
            'prob_max_simultaneous_onset': 0, 'prob_extreme_burst_count': 0,
            'prob_tempo_extreme_count': 0, 'prob_tempo_range': 0,
            'prob_excessive_time_sig_changes': 0,
        }

        if not self.notes:
            features.update(nan_fill)
            return features

        durations = np.array([n[2] for n in self.notes])
        pitches = np.array([n[4] for n in self.notes])
        velocities = np.array([n[5] for n in self.notes])
        start_times = np.array([n[1] for n in self.notes])
        channels = np.array([n[3] for n in self.notes])
        total = len(self.notes)

        # --- Zero duration notes ---
        zero_dur = int(np.sum(durations == 0))
        features['prob_zero_duration_count'] = zero_dur
        features['prob_zero_duration_ratio'] = zero_dur / total

        # --- Velocity zero notes (note-off-by-velocity-zero encoding) ---
        vel_zero = int(np.sum(velocities == 0))
        features['prob_velocity_zero_count'] = vel_zero
        features['prob_velocity_zero_ratio'] = vel_zero / total

        # --- Out-of-range pitches ---
        oor = int(np.sum((pitches < 0) | (pitches > 127)))
        features['prob_out_of_range_pitch_count'] = oor
        features['prob_out_of_range_pitch_ratio'] = oor / total

        # --- Negative durations ---
        neg_dur = int(np.sum(durations < 0))
        features['prob_negative_duration_count'] = neg_dur
        features['prob_negative_duration_ratio'] = neg_dur / total

        # --- Overlapping same-pitch & same-channel notes ---
        # FIX: Use sweep-line to detect ALL overlapping pairs, not just adjacent
        note_by_key = defaultdict(list)
        for idx, n in enumerate(self.notes):
            note_by_key[(n[3], n[4])].append((n[1], n[1] + n[2], idx))

        overlap_note_indices = set()
        for intervals in note_by_key.values():
            if len(intervals) < 2:
                continue
            intervals.sort()
            # Sweep line: track active notes and mark overlapping ones
            events = []
            for s, e, idx in intervals:
                events.append((s, 1, idx))   # start
                events.append((e, -1, idx))  # end
            # Sort by tick, then ends before starts at same tick
            events.sort(key=lambda x: (x[0], x[1]))

            active = set()
            for _, delta, idx in events:
                if delta == 1:  # start event
                    if active:
                        overlap_note_indices.add(idx)
                        overlap_note_indices.update(active)
                    active.add(idx)
                else:  # end event
                    active.discard(idx)

        overlap_count = len(overlap_note_indices)
        features['prob_overlapping_same_pitch_count'] = overlap_count
        features['prob_overlapping_same_pitch_ratio'] = overlap_count / total

        # --- Stuck notes (held > 16 beats) ---
        long_threshold = 16 * (self.ticks_per_quarter or 480)
        stuck_count = int(np.sum(durations > long_threshold))
        features['prob_stuck_note_count'] = stuck_count
        features['prob_stuck_note_ratio'] = stuck_count / total

        # --- Very short durations (< 5 ticks) ---
        vshort = int(np.sum((durations > 0) & (durations < 5)))
        features['prob_very_short_duration_count'] = vshort
        features['prob_very_short_duration_ratio'] = vshort / total

        # --- Very long durations (> 4 beats) ---
        beat4 = 4 * (self.ticks_per_quarter or 480)
        vlong = int(np.sum(durations > beat4))
        features['prob_very_long_duration_count'] = vlong
        features['prob_very_long_duration_ratio'] = vlong / total

        # --- Extreme pitch (outside piano range 21-108) ---
        extreme = int(np.sum((pitches < 21) | (pitches > 108)))
        features['prob_extreme_pitch_count'] = extreme
        features['prob_extreme_pitch_ratio'] = extreme / total

        # --- Velocity uniformity ---
        unique_vels = len(np.unique(velocities))
        features['prob_unique_velocity_count'] = unique_vels
        features['prob_single_velocity'] = 1 if unique_vels == 1 else 0
        features['prob_all_max_velocity'] = 1 if (unique_vels == 1 and velocities[0] == 127) else 0
        features['prob_all_min_velocity'] = 1 if (unique_vels == 1 and velocities[0] == 1) else 0

        # --- Simultaneous note bursts ---
        onset_counts = Counter(start_times)
        if onset_counts:
            max_burst = max(onset_counts.values())
            features['prob_max_simultaneous_onset'] = max_burst
            features['prob_extreme_burst_count'] = sum(
                1 for c in onset_counts.values() if c > 20)
        else:
            features['prob_max_simultaneous_onset'] = 0
            features['prob_extreme_burst_count'] = 0

        # --- Tempo anomalies ---
        bpms = [60000000.0 / t[2] for t in self.tempos if t[2] > 0]
        if bpms:
            features['prob_tempo_extreme_count'] = sum(1 for b in bpms if b < 20 or b > 300)
            features['prob_tempo_range'] = max(bpms) - min(bpms)
        else:
            features['prob_tempo_extreme_count'] = 0
            features['prob_tempo_range'] = 0

        # --- Excessive time-signature changes ---
        features['prob_excessive_time_sig_changes'] = 1 if len(self.time_sigs) > 10 else 0

        return features

    # =================================================================
    # 3. INSTRUMENT FEATURES
    # =================================================================
    def _get_instrument_features(self):
        features = {}
        patch_set = set()
        channel_set = set()
        drum_notes = 0
        melodic_notes = 0

        for n in self.notes:
            ch = n[3]
            channel_set.add(ch)
            if ch == 9:
                drum_notes += 1
            else:
                melodic_notes += 1

        for track_patches in self.patches.values():
            for p in track_patches:
                # Track patches uniquely by (channel, patch)
                patch_set.add((p[2], p[3]))

        total = len(self.notes) if self.notes else 1
        features['instruments_unique_count'] = len(patch_set)
        features['instruments_channel_count'] = len(channel_set)
        features['instruments_drum_note_ratio'] = drum_notes / total
        features['instruments_melodic_note_ratio'] = melodic_notes / total
        features['instruments_is_polytimbral'] = 1 if len(patch_set) > 1 else 0

        channel_counts = Counter(n[3] for n in self.notes)
        features['instruments_channel_entropy'] = self._entropy(list(channel_counts.values()))

        return features

    # =================================================================
    # 4. TEMPO & PACE FEATURES
    # =================================================================
    def _get_tempo_features(self):
        features = {}

        bpms = [60000000.0 / t[2] for t in self.tempos if t[2] > 0]
        if not bpms:
            bpms = [120.0]

        features.update(self._get_stat_dict(bpms, 'tempo_bpm'))

        # Tempo change dynamics
        if len(bpms) > 1:
            bpm_diffs = np.abs(np.diff(bpms))
            features['tempo_bpm_change_mean'] = float(np.mean(bpm_diffs))
            features['tempo_bpm_change_max'] = float(np.max(bpm_diffs))
            features['tempo_bpm_change_std'] = float(np.std(bpm_diffs))
        else:
            features['tempo_bpm_change_mean'] = 0.0
            features['tempo_bpm_change_max'] = 0.0
            features['tempo_bpm_change_std'] = 0.0

        # Pace
        if self.notes and self.ticks_per_quarter:
            start_times = np.array([n[1] for n in self.notes])
            span = start_times[-1] - start_times[0] if len(start_times) > 1 else self.ticks_per_quarter
            num_beats = span / self.ticks_per_quarter
            features['pace_notes_per_beat'] = len(self.notes) / num_beats if num_beats > 0 else np.nan

            beats_per_bar = 4.0
            if self.time_sigs:
                # In MIDI.py, time_sig = ['time_signature', time, nn, dd, ccc, bb]
                nn = self.time_sigs[0][2]
                dd = self.time_sigs[0][3]
                if dd >= 0:
                    beats_per_bar = nn * (4.0 / (2 ** dd))
            num_bars = num_beats / beats_per_bar
            features['pace_notes_per_bar'] = len(self.notes) / num_bars if num_bars > 0 else np.nan
        else:
            features['pace_notes_per_beat'] = np.nan
            features['pace_notes_per_bar'] = np.nan

        return features

    # =================================================================
    # 5. GRID & RHYTHM FEATURES
    # =================================================================
    def _get_grid_features(self):
        features = {}
        if not self.notes or not self.ticks_per_quarter:
            return features

        start_times = np.array([n[1] for n in self.notes], dtype=float)
        durations = np.array([n[2] for n in self.notes], dtype=float)

        # Normalize to 24 PPQN
        factor = 24.0 / self.ticks_per_quarter
        grid_starts = start_times * factor
        grid_durations = durations * factor

        # Quantization error
        start_offsets = np.abs(grid_starts - np.round(grid_starts))
        dur_offsets = np.abs(grid_durations - np.round(grid_durations))

        features['grid_start_offset_mean'] = float(np.mean(start_offsets))
        features['grid_start_offset_std'] = float(np.std(start_offsets))
        features['grid_dur_offset_mean'] = float(np.mean(dur_offsets))
        features['grid_dur_offset_std'] = float(np.std(dur_offsets))

        # Alignment ratios (with tolerance for floating-point)
        tol = 0.01
        features['grid_start_alignment_ratio'] = float(np.sum(start_offsets < tol) / len(start_offsets))
        features['grid_dur_alignment_ratio'] = float(np.sum(dur_offsets < tol) / len(dur_offsets))

        # Granular alignment: quarter, eighth, sixteenth
        for name, divisor in [('quarter', 6.0), ('eighth', 3.0), ('sixteenth', 1.5)]:
            mod_vals = np.mod(grid_starts, divisor)
            aligned = np.minimum(mod_vals, divisor - mod_vals)
            features[f'grid_on_{name}_ratio'] = float(np.sum(aligned < 0.1) / len(grid_starts))

        # Swing detection
        eighth_notes = grid_durations[(grid_durations >= 2) & (grid_durations <= 4)]
        if len(eighth_notes) > 10:
            int_eighths = np.round(eighth_notes).astype(int)
            if int_eighths.min() >= 0:
                counts = np.bincount(int_eighths, minlength=5)
                if len(counts) > 4:
                    counts = counts[2:5]
                    if counts[1] > 0:
                        features['grid_swing_ratio'] = float((counts[0] + counts[2]) / counts[1])
                    else:
                        features['grid_swing_ratio'] = np.nan
                else:
                    features['grid_swing_ratio'] = np.nan
            else:
                features['grid_swing_ratio'] = np.nan
        else:
            features['grid_swing_ratio'] = np.nan

        # Syncopation
        off_beat = grid_starts[np.mod(grid_starts, 6.0) != 0]
        features['grid_syncopation_ratio'] = float(len(off_beat) / len(grid_starts))

        # Rhythmic entropy
        rounded_durs = np.round(grid_durations).astype(int)
        features['grid_rhythmic_entropy'] = self._entropy(rounded_durs.tolist())

        features.update(self._get_stat_dict(grid_durations, 'grid_durations'))
        return features

    # =================================================================
    # 6. HARMONIC & KEY FEATURES
    # =================================================================
    def _get_harmonic_features(self):
        features = {}
        if not self.notes:
            return features

        pitches = np.array([n[4] for n in self.notes])
        pitch_classes = pitches % 12

        # Key estimation via Krumhansl-Schmuckler
        pc_hist = np.bincount(pitch_classes, minlength=12)
        pc_dist = pc_hist / len(pitches)

        correlations = []
        for i in range(12):
            rotated = np.roll(pc_dist, i)
            major_c = np.corrcoef(rotated, self.MAJOR_PROFILE)[0, 1]
            minor_c = np.corrcoef(rotated, self.MINOR_PROFILE)[0, 1]
            correlations.append((major_c, i, 'major'))
            correlations.append((minor_c, i, 'minor'))

        correlations.sort(key=lambda x: x[0], reverse=True)
        best_corr, best_key, best_mode = correlations[0]
        second_corr = correlations[1][0]

        features['harmonic_key_tonic'] = best_key
        features['harmonic_key_is_minor'] = 1 if best_mode == 'minor' else 0
        features['harmonic_key_clarity'] = float(best_corr) if not np.isnan(best_corr) else 0.0
        features['harmonic_key_strength'] = float(best_corr - second_corr) if not (
            np.isnan(best_corr) or np.isnan(second_corr)) else 0.0

        # Pitch-class entropy (high = atonal)
        features['harmonic_pitch_class_entropy'] = self._entropy(pc_hist.tolist())

        # Vertical consonance / dissonance
        onset_dict = defaultdict(list)
        for n in self.notes:
            onset_dict[n[1]].append(n[4])

        consonance_count = 0
        dissonance_count = 0
        for chord_pitches in onset_dict.values():
            if len(chord_pitches) < 2:
                continue
            for i in range(len(chord_pitches)):
                for j in range(i + 1, len(chord_pitches)):
                    ic = self.INTERVAL_CLASSES[abs(chord_pitches[i] - chord_pitches[j]) % 12]
                    if ic in [3, 4, 5]:
                        consonance_count += 1
                    elif ic in [1, 2, 6]:
                        dissonance_count += 1

        total_iv = consonance_count + dissonance_count
        features['harmonic_consonance_ratio'] = consonance_count / total_iv if total_iv > 0 else np.nan
        features['harmonic_dissonance_ratio'] = dissonance_count / total_iv if total_iv > 0 else np.nan
        features['harmonic_total_intervals'] = total_iv

        # Key signature metadata
        features['harmonic_has_key_signature'] = 1 if self.key_sigs else 0
        features['harmonic_key_sig_change_count'] = len(self.key_sigs)

        return features

    # =================================================================
    # 7. NOTE FEATURES
    # =================================================================
    def _get_note_features(self):
        features = {}
        if not self.notes:
            return features

        start_times = np.array([n[1] for n in self.notes])
        durations = np.array([n[2] for n in self.notes])
        pitches = np.array([n[4] for n in self.notes])
        velocities = np.array([n[5] for n in self.notes])
        channels = np.array([n[3] for n in self.notes])

        # Inter-onset intervals (global, across all channels)
        if len(start_times) > 1:
            ioi = np.diff(start_times)
            features.update(self._get_stat_dict(ioi, 'delta_times'))
            features['delta_times_zero_ratio'] = float(np.sum(ioi == 0) / len(ioi))
            # FIX: Add non-zero IOI statistics — much more meaningful for
            # polyphonic music where many notes share the same onset
            nonzero_ioi = ioi[ioi > 0]
            if len(nonzero_ioi) > 0:
                features.update(self._get_stat_dict(nonzero_ioi, 'delta_times_nonzero'))
            else:
                features.update(self._get_stat_dict([], 'delta_times_nonzero'))
        else:
            features.update(self._get_stat_dict([], 'delta_times'))
            features['delta_times_zero_ratio'] = np.nan
            features.update(self._get_stat_dict([], 'delta_times_nonzero'))

        features.update(self._get_stat_dict(durations, 'durations'))
        features.update(self._get_stat_dict(pitches, 'pitches'))
        features.update(self._get_stat_dict(velocities, 'velocities'))

        # FIX: Pitch intervals & contour computed PER-CHANNEL, excluding
        # simultaneous notes. Computing across all channels is musically
        # meaningless (e.g., a bass note followed by a melody note is
        # not a melodic interval).
        channel_notes = defaultdict(list)
        for n in self.notes:
            channel_notes[n[3]].append(n)

        all_pitch_diffs = []
        for ch, ch_notes in channel_notes.items():
            ch_notes_sorted = sorted(ch_notes, key=lambda x: (x[1], x[4]))
            ch_starts = np.array([n[1] for n in ch_notes_sorted])
            ch_pitches = np.array([n[4] for n in ch_notes_sorted])
            if len(ch_pitches) > 1:
                ch_ioi = np.diff(ch_starts)
                ch_pitch_diffs = np.diff(ch_pitches)
                # Only include intervals between non-simultaneous notes
                melodic_diffs = ch_pitch_diffs[ch_ioi > 0]
                if len(melodic_diffs) > 0:
                    all_pitch_diffs.extend(melodic_diffs.tolist())

        if all_pitch_diffs:
            pitch_diffs = np.array(all_pitch_diffs)
            features.update(self._get_stat_dict(pitch_diffs, 'pitch_intervals'))
            signs = np.sign(pitch_diffs)
            # Filter out zero signs for direction change detection
            nonzero_signs = signs[signs != 0]
            features['pitches_direction_changes'] = int(np.sum(np.diff(nonzero_signs) != 0)) if len(nonzero_signs) > 1 else 0
            features['pitches_ascending_ratio'] = float(np.sum(pitch_diffs > 0) / len(pitch_diffs))
            features['pitches_descending_ratio'] = float(np.sum(pitch_diffs < 0) / len(pitch_diffs))
            features['pitches_step_ratio'] = float(np.sum(np.abs(pitch_diffs) <= 2) / len(pitch_diffs))
            features['pitches_leap_ratio'] = float(np.sum(np.abs(pitch_diffs) > 4) / len(pitch_diffs))
        else:
            features.update(self._get_stat_dict([], 'pitch_intervals'))
            features['pitches_direction_changes'] = 0
            features['pitches_ascending_ratio'] = np.nan
            features['pitches_descending_ratio'] = np.nan
            features['pitches_step_ratio'] = np.nan
            features['pitches_leap_ratio'] = np.nan

        # Pitch-class distribution
        pc_hist = np.bincount(pitches % 12, minlength=12) / len(pitches)
        for i in range(12):
            features[f'pitch_class_{i}_ratio'] = float(pc_hist[i])

        # Velocity dynamics
        if len(velocities) > 1:
            vel_diffs = np.diff(velocities)
            features['velocities_change_mean'] = float(np.mean(vel_diffs))
            features['velocities_change_std'] = float(np.std(vel_diffs))
        else:
            features['velocities_change_mean'] = np.nan
            features['velocities_change_std'] = np.nan

        # Channel distribution
        ch_counts = Counter(channels)
        features['notes_active_channels'] = len(ch_counts)
        features['notes_channel_entropy'] = self._entropy(list(ch_counts.values()))

        return features

    # =================================================================
    # 8. CHORD FEATURES
    # =================================================================
    def _get_chord_features(self):
        features = {}
        if not self.notes:
            return features

        onset_dict = defaultdict(list)
        for n in self.notes:
            onset_dict[n[1]].append(n[4])

        chords_raw = [tuple(sorted(p)) for p in onset_dict.values() if len(p) >= 2]
        single_onsets = sum(1 for p in onset_dict.values() if len(p) == 1)
        total_onsets = len(onset_dict)

        features['chords_count'] = len(chords_raw)
        features['chords_single_onset_count'] = single_onsets
        features['chords_ratio'] = len(chords_raw) / total_onsets if total_onsets > 0 else np.nan

        if chords_raw:
            chord_sizes = [len(c) for c in chords_raw]
            features.update(self._get_stat_dict(chord_sizes, 'chords_size'))
            features['chords_unique_raw_count'] = len(set(chords_raw))
            features['chords_uniqueness_ratio'] = len(set(chords_raw)) / len(chords_raw)

            # FIX: Triad quality analysis — try each pitch class as root
            # instead of always using the lowest note, which misses inversions
            triad_types = Counter()
            for c in chords_raw:
                if len(c) >= 3:
                    pc_set = sorted(set(p % 12 for p in c))
                    quality = self._classify_triad(pc_set)
                    if quality:
                        triad_types[quality] += 1

            total_triads = sum(triad_types.values()) or 1
            for t in ['major', 'minor', 'diminished', 'augmented', 'sus4', 'sus2']:
                features[f'chords_{t}_ratio'] = triad_types.get(t, 0) / total_triads
        else:
            features['chords_unique_raw_count'] = 0
            features['chords_uniqueness_ratio'] = np.nan

        return features

    @staticmethod
    def _classify_triad(pc_set):
        """
        Classify a chord's triad quality by trying each pitch class as root.
        This correctly handles inversions that the previous approach missed.
        
        Args:
            pc_set: sorted list of unique pitch classes (0-11)
        
        Returns:
            String quality name or None
        """
        for root in pc_set:
            intervals_from_root = set((pc - root) % 12 for pc in pc_set)
            # Check in order of specificity
            if 4 in intervals_from_root and 7 in intervals_from_root:
                return 'major'
            elif 3 in intervals_from_root and 7 in intervals_from_root:
                return 'minor'
            elif 3 in intervals_from_root and 6 in intervals_from_root:
                return 'diminished'
            elif 4 in intervals_from_root and 8 in intervals_from_root:
                return 'augmented'
            elif 5 in intervals_from_root and 7 in intervals_from_root:
                return 'sus4'
            elif 2 in intervals_from_root and 7 in intervals_from_root:
                return 'sus2'
        return None

    # =================================================================
    # 9. POLYPHONY FEATURES
    # =================================================================
    def _get_polyphony_features(self):
        features = {}
        if not self.notes:
            return features

        start_times = [n[1] for n in self.notes]
        end_times = [n[1] + n[2] for n in self.notes]

        # Build event list: end events (-1) before start events (+1) at same tick
        events = []
        for t in start_times:
            events.append((t, 1))
        for t in end_times:
            events.append((t, -1))
        # Sort by tick, then by delta: end (-1) comes before start (+1) at the same tick
        events.sort(key=lambda x: (x[0], x[1]))

        max_poly = 0
        current_poly = 0
        poly_history = []

        # FIX: Time-weighted polyphony computation
        # Also fix: clamp current_poly to prevent negative values caused by
        # zero-duration notes (where end event at same tick precedes start)
        prev_tick = events[0][0] if events else 0
        weighted_sum = 0.0
        total_ticks = 0

        for tick, delta in events:
            # Accumulate time at current polyphony level
            duration = tick - prev_tick
            if duration > 0:
                weighted_sum += current_poly * duration
                total_ticks += duration

            current_poly = max(0, current_poly + delta)  # FIX: clamp to prevent negative
            if current_poly > max_poly:
                max_poly = current_poly
            poly_history.append(current_poly)
            prev_tick = tick

        features['polyphony_max'] = max_poly

        # Time-weighted mean (more musically meaningful than event-based)
        if total_ticks > 0:
            features['polyphony_mean'] = float(weighted_sum / total_ticks)
        elif poly_history:
            features['polyphony_mean'] = float(np.mean(poly_history))
        else:
            features['polyphony_mean'] = 0.0

        # Event-based std (kept for compatibility)
        features['polyphony_std'] = float(np.std(poly_history)) if poly_history else 0.0

        # Monophonic ratio
        mono = sum(1 for p in poly_history if p <= 1)
        features['polyphony_monophonic_ratio'] = mono / len(poly_history) if poly_history else np.nan

        # Overlapping notes ratio
        overlapping = 0
        held = []
        for n in self.notes:
            held = [e for e in held if e > n[1]]
            if held:
                overlapping += 1
            held.append(n[1] + n[2])
        features['polyphony_overlap_ratio'] = overlapping / len(self.notes)

        # Multi-channel flag
        channels = set(n[3] for n in self.notes)
        features['polyphony_multi_channel'] = 1 if len(channels) > 1 else 0

        return features

    # =================================================================
    # 10. PHRASE FEATURES
    # =================================================================
    def _get_phrase_features(self):
        features = {}
        if len(self.notes) < 2:
            features['phrases_count'] = 0 if not self.notes else 1
            features['phrases_avg_note_count'] = len(self.notes) or np.nan
            return features

        start_times = np.array([n[1] for n in self.notes])
        ioi = np.diff(start_times)

        if len(ioi) == 0:
            features['phrases_count'] = 1
            features['phrases_avg_note_count'] = len(self.notes)
            return features

        med_dt = np.median(ioi)
        iqr_dt = np.percentile(ioi, 75) - np.percentile(ioi, 25)
        threshold_a = med_dt + 1.5 * iqr_dt
        threshold_b = np.percentile(ioi, 90)
        phrase_threshold = max(threshold_a, threshold_b)
        if phrase_threshold <= 0:
            phrase_threshold = med_dt if med_dt > 0 else 1.0

        # Build phrases using IOI (gap between note i and note i+1)
        phrases = [[0]]
        for i in range(len(ioi)):
            if ioi[i] > phrase_threshold:
                phrases.append([])
            phrases[-1].append(i + 1)

        phrase_note_counts = [len(p) for p in phrases]
        features['phrases_count'] = len(phrases)
        features['phrases_avg_note_count'] = float(np.mean(phrase_note_counts))
        features['phrases_note_count_std'] = float(np.std(phrase_note_counts))
        features['phrases_min_note_count'] = int(np.min(phrase_note_counts))
        features['phrases_max_note_count'] = int(np.max(phrase_note_counts))

        # Phrase lengths in ticks
        phrase_lengths = []
        for p in phrases:
            if len(p) >= 2:
                phrase_lengths.append(start_times[p[-1]] - start_times[p[0]])
        if phrase_lengths:
            features.update(self._get_stat_dict(phrase_lengths, 'phrases_length'))
        else:
            features.update(self._get_stat_dict([], 'phrases_length'))

        return features

    # =================================================================
    # 11. DYNAMICS FEATURES
    # =================================================================
    def _get_dynamics_features(self):
        features = {}
        if not self.notes:
            return features

        velocities = np.array([n[5] for n in self.notes])

        features['dynamics_velocity_range'] = int(np.max(velocities) - np.min(velocities))
        features['dynamics_velocity_unique'] = int(len(np.unique(velocities)))

        # Dynamic range in dB
        vmin, vmax = int(np.min(velocities)), int(np.max(velocities))
        if vmin > 0 and vmax > 0:
            features['dynamics_range_db'] = float(20 * np.log10(vmax / vmin))
        else:
            features['dynamics_range_db'] = np.nan

        # Velocity histogram entropy
        vel_counts = Counter(velocities)
        features['dynamics_entropy'] = self._entropy(list(vel_counts.values()))

        # Same-velocity consecutive ratio
        if len(velocities) > 1:
            features['dynamics_same_consecutive_ratio'] = float(
                np.sum(np.diff(velocities) == 0) / len(velocities))
        else:
            features['dynamics_same_consecutive_ratio'] = np.nan

        # Short-note ratio (likely articulation artifacts)
        if self.ticks_per_quarter:
            short_thresh = self.ticks_per_quarter / 4
            short_count = sum(1 for n in self.notes if 0 < n[2] < short_thresh)
            features['dynamics_short_note_ratio'] = short_count / len(self.notes)
        else:
            features['dynamics_short_note_ratio'] = np.nan

        return features

    # =================================================================
    # 12. QUALITY / COMPLETENESS FEATURES
    # =================================================================
    def _get_quality_features(self):
        features = {}

        features['quality_has_tempo'] = 1 if self.tempos else 0
        features['quality_has_time_sig'] = 1 if self.time_sigs else 0
        features['quality_has_key_sig'] = 1 if self.key_sigs else 0

        # Default-value detection
        if self.tempos:
            first_bpm = 60000000.0 / self.tempos[0][2] if self.tempos[0][2] > 0 else 120.0
            features['quality_default_tempo'] = 1 if abs(first_bpm - 120.0) < 0.1 else 0
        else:
            features['quality_default_tempo'] = 1

        if self.time_sigs:
            ts = self.time_sigs[0]
            # Check for 4/4 default: nn=4, dd=2 (2^2=4 denominator)
            features['quality_default_time_sig'] = 1 if ts[2] == 4 and ts[3] == 2 else 0
        else:
            features['quality_default_time_sig'] = 1

        # Density flags
        nps = 0.0
        if self.notes:
            max_tick = max(n[1] + n[2] for n in self.notes)
            dur_s = self._ticks_to_seconds(max_tick)
            nps = len(self.notes) / dur_s if dur_s > 0 else 0.0

        features['quality_very_sparse'] = 1 if (nps > 0 and nps < 0.5) else 0
        features['quality_very_dense'] = 1 if nps > 100 else 0

        # Pitch range flags
        if self.notes:
            pitches = [n[4] for n in self.notes]
            pr = max(pitches) - min(pitches)
            features['quality_pitch_range_full'] = 1 if (min(pitches) <= 36 and max(pitches) >= 96) else 0
            features['quality_very_narrow_range'] = 1 if pr < 12 else 0
        else:
            features['quality_pitch_range_full'] = 0
            features['quality_very_narrow_range'] = 1

        return features

    # =================================================================
    # MASTER EXTRACTOR
    # =================================================================
    def extract_all(self):
        if self.parse_error:
            return {"error": self.parse_error, "filename": self.filename,
                    "struct_parse_error": 1, "struct_total_notes": 0}

        if not self.notes:
            return {"error": "No notes found", "filename": self.filename,
                    "struct_parse_error": 0, "struct_total_notes": 0}

        all_features = {"filename": self.filename}
        all_features.update(self._get_structural_features())
        all_features.update(self._get_problem_features())
        all_features.update(self._get_instrument_features())
        all_features.update(self._get_tempo_features())
        all_features.update(self._get_grid_features())
        all_features.update(self._get_harmonic_features())
        all_features.update(self._get_note_features())
        all_features.update(self._get_chord_features())
        all_features.update(self._get_polyphony_features())
        all_features.update(self._get_phrase_features())
        all_features.update(self._get_dynamics_features())
        all_features.update(self._get_quality_features())
        return all_features


# =====================================================================
# PROBLEM DETECTOR — Rule-based detection of concrete MIDI problems
# =====================================================================

class MIDIProblemDetector:
    """
    Rule-based detector for concrete problems in MIDI files:
    corruption, malformation, encoding errors, and quality issues.
    """

    def __init__(self, features):
        self.f = features
        self.problems = []

    def _add(self, category, issue, severity, detail=""):
        self.problems.append({
            'category': category,
            'issue': issue,
            'severity': severity,   # 'critical', 'warning', 'info'
            'detail': detail
        })

    def detect(self):
        f = self.f

        # --- Parse failure ---
        if f.get('struct_parse_error') == 1:
            self._add('structure', 'parse_error', 'critical',
                      f.get('error', 'Unknown parse error'))
            return self.problems

        if f.get('struct_total_notes', 1) == 0:
            self._add('structure', 'no_notes', 'critical',
                      'MIDI file contains no note events')
            return self.problems

        # --- Zero-duration notes ---
        r = f.get('prob_zero_duration_ratio', 0) or 0
        if r > 0.5:
            self._add('corruption', 'zero_duration_notes', 'critical',
                      f'{r:.1%} of notes have zero duration')
        elif r > 0.1:
            self._add('corruption', 'zero_duration_notes', 'warning',
                      f'{r:.1%} of notes have zero duration')
        elif r > 0.01:
            self._add('corruption', 'zero_duration_notes', 'info',
                      f'{r:.1%} of notes have zero duration')

        # --- Velocity-zero notes (note-off encoding) ---
        r = f.get('prob_velocity_zero_ratio', 0) or 0
        if r > 0.3:
            self._add('corruption', 'velocity_zero_notes', 'critical',
                      f'{r:.1%} of notes have velocity zero (note-off-by-velocity encoding)')
        elif r > 0.05:
            self._add('corruption', 'velocity_zero_notes', 'warning',
                      f'{r:.1%} of notes have velocity zero')

        # --- Out-of-range pitches ---
        r = f.get('prob_out_of_range_pitch_ratio', 0) or 0
        if r > 0:
            self._add('corruption', 'out_of_range_pitches', 'critical',
                      f'{r:.1%} of notes have pitches outside MIDI range 0-127')

        # --- Negative durations ---
        r = f.get('prob_negative_duration_ratio', 0) or 0
        if r > 0:
            self._add('corruption', 'negative_durations', 'critical',
                      f'{r:.1%} of notes have negative durations')

        # --- Overlapping same-pitch same-channel ---
        r = f.get('prob_overlapping_same_pitch_ratio', 0) or 0
        if r > 0.3:
            self._add('corruption', 'overlapping_same_pitch', 'warning',
                      f'{r:.1%} of notes overlap with same pitch on same channel')
        elif r > 0.1:
            self._add('corruption', 'overlapping_same_pitch', 'info',
                      f'{r:.1%} of notes overlap with same pitch on same channel')

        # --- Stuck notes ---
        r = f.get('prob_stuck_note_ratio', 0) or 0
        if r > 0.01:
            self._add('corruption', 'stuck_notes', 'warning',
                      f'{r:.1%} of notes held >16 beats (stuck notes)')

        # --- Extreme bursts ---
        burst = f.get('prob_max_simultaneous_onset', 0) or 0
        if burst > 50:
            self._add('corruption', 'extreme_burst', 'warning',
                      f'Max simultaneous onset: {burst} notes at same tick')
        elif burst > 20:
            self._add('corruption', 'extreme_burst', 'info',
                      f'Max simultaneous onset: {burst} notes at same tick')

        # --- Non-standard TPQ ---
        if f.get('struct_nonstandard_tpq') == 1:
            self._add('structure', 'nonstandard_tpq', 'info',
                      f'Non-standard ticks per quarter: {f.get("struct_ticks_per_quarter")}')

        # --- Excessive tempo changes ---
        tps = f.get('struct_tempo_changes_per_second')
        if tps is not None and not (isinstance(tps, float) and np.isnan(tps)):
            if tps > 10:
                self._add('structure', 'excessive_tempo_changes', 'warning',
                          f'{tps:.1f} tempo changes/second')
            elif tps > 2:
                self._add('structure', 'excessive_tempo_changes', 'info',
                          f'{tps:.1f} tempo changes/second')

        # --- Extreme tempo values ---
        tmin = f.get('tempo_bpm_min')
        tmax = f.get('tempo_bpm_max')
        if tmin is not None and tmax is not None and not (isinstance(tmin, float) and np.isnan(tmin)) and not (isinstance(tmax, float) and np.isnan(tmax)):
            if tmin < 20 or tmax > 300:
                self._add('structure', 'extreme_tempo', 'warning',
                          f'Tempo range: {tmin:.0f}-{tmax:.0f} BPM')

        # --- Large tempo range ---
        trange = f.get('tempo_bpm_range', 0) or 0
        if not (isinstance(trange, float) and np.isnan(trange)) and trange > 200:
            self._add('structure', 'large_tempo_range', 'warning',
                      f'Tempo varies by {trange:.0f} BPM')

        # --- Excessive time-sig changes ---
        if f.get('prob_excessive_time_sig_changes') == 1:
            self._add('structure', 'excessive_time_sig_changes', 'warning',
                      f'{f.get("struct_time_sig_change_count", 0)} time signature changes')

        # --- Empty tracks ---
        er = f.get('struct_empty_track_ratio', 0) or 0
        if not (isinstance(er, float) and np.isnan(er)):
            if er > 0.7:
                self._add('structure', 'many_empty_tracks', 'warning',
                          f'{er:.1%} of tracks are empty')
            elif er > 0.4:
                self._add('structure', 'many_empty_tracks', 'info',
                          f'{er:.1%} of tracks are empty')

        # --- No dynamics ---
        if f.get('prob_single_velocity') == 1:
            self._add('quality', 'no_dynamics', 'warning',
                      'All notes have the same velocity')

        # --- All max velocity ---
        if f.get('prob_all_max_velocity') == 1:
            self._add('quality', 'all_max_velocity', 'warning',
                      'All notes at maximum velocity 127')

        # --- Atonal / weak key ---
        kc = f.get('harmonic_key_clarity', 1)
        if kc is not None and not (isinstance(kc, float) and np.isnan(kc)):
            if kc < 0.3:
                self._add('quality', 'atonal', 'warning',
                          f'Very low key clarity ({kc:.3f}) — likely atonal or random')
            elif kc < 0.5:
                self._add('quality', 'weak_tonality', 'info',
                          f'Low key clarity ({kc:.3f}) — weak tonal center')

        # --- Very short notes ---
        r = f.get('prob_very_short_duration_ratio', 0) or 0
        if not (isinstance(r, float) and np.isnan(r)) and r > 0.3:
            self._add('quality', 'very_short_notes', 'warning',
                      f'{r:.1%} of notes <5 ticks duration')

        # --- Missing metadata ---
        if f.get('quality_has_tempo') == 0:
            self._add('quality', 'missing_tempo', 'info',
                      'No tempo event (defaults to 120 BPM)')
        if f.get('quality_has_time_sig') == 0:
            self._add('quality', 'missing_time_sig', 'info',
                      'No time signature (defaults to 4/4)')

        # --- Extreme density ---
        nps = f.get('struct_notes_per_second', 0) or 0
        if not (isinstance(nps, float) and np.isnan(nps)):
            if nps > 100:
                self._add('quality', 'extreme_density', 'warning',
                          f'{nps:.1f} notes/second (extremely dense)')
            elif nps < 0.5 and (f.get('struct_duration_secs', 0) or 0) > 10:
                self._add('quality', 'very_sparse', 'info',
                          f'{nps:.2f} notes/second (very sparse)')

        # --- Narrow pitch range ---
        if f.get('quality_very_narrow_range') == 1:
            self._add('quality', 'narrow_pitch_range', 'info',
                      'Pitch range < 1 octave')

        # --- Poor grid alignment ---
        ga = f.get('grid_start_alignment_ratio', 1) or 1
        if not (isinstance(ga, float) and np.isnan(ga)) and ga < 0.1:
            self._add('quality', 'poor_grid_alignment', 'info',
                      f'Only {ga:.1%} of notes align to grid (unquantized or corrupted)')

        # --- High dissonance ---
        dr = f.get('harmonic_dissonance_ratio', 0) or 0
        if not (isinstance(dr, float) and np.isnan(dr)) and dr > 0.7:
            self._add('quality', 'high_dissonance', 'info',
                      f'{dr:.1%} of intervals are dissonant')

        # --- Inefficient file ---
        bpn = f.get('struct_bytes_per_note', 0) or 0
        if not (isinstance(bpn, float) and np.isnan(bpn)) and bpn > 1000:
            self._add('quality', 'inefficient_file', 'info',
                      f'{bpn:.0f} bytes per note (high overhead)')

        return self.problems

    def get_severity_summary(self):
        if not self.problems:
            return {'critical': 0, 'warning': 0, 'info': 0}
        return dict(Counter(p['severity'] for p in self.problems))

    def is_problematic(self):
        s = self.get_severity_summary()
        return s.get('critical', 0) > 0 or s.get('warning', 0) > 0

    def get_report(self):
        lines = [f"MIDI Problem Report: {self.f.get('filename', 'unknown')}",
                 "=" * 60]
        if not self.problems:
            lines.append("No problems detected.")
            return '\n'.join(lines)

        for sev in ['critical', 'warning', 'info']:
            issues = [p for p in self.problems if p['severity'] == sev]
            if issues:
                lines.append(f"\n[{sev.upper()}]")
                for p in issues:
                    lines.append(f"  [{p['category']}] {p['issue']}: {p['detail']}")

        s = self.get_severity_summary()
        lines.append(f"\nSummary: {s.get('critical', 0)} critical, "
                     f"{s.get('warning', 0)} warnings, {s.get('info', 0)} info")
        return '\n'.join(lines)


# =====================================================================
# ANOMALY DETECTOR — Statistical anomaly detection
# =====================================================================

class MIDIAnomalyDetector:
    """
    Statistical anomaly detection for MIDI files.
    Uses Z-score analysis against reference ranges, plus cross-feature heuristics.
    Can also output feature vectors for ML-based anomaly detection.
    """

    # Reference ranges: (mean, std, min_expected, max_expected)
    REFERENCE_RANGES = {
        'tempo_bpm_mean':               (120, 40,  40,  240),
        'tempo_bpm_std':                (15,  20,  0,   100),
        'pace_notes_per_beat':          (4,   3,   0.5, 20),
        'grid_start_alignment_ratio':   (0.7, 0.25,0.1, 1.0),
        'grid_start_offset_mean':       (0.1, 0.1, 0.0, 0.45),
        'harmonic_key_clarity':         (0.75,0.15,0.3, 1.0),
        'harmonic_consonance_ratio':    (0.6, 0.15,0.2, 0.95),
        'durations_cv':                 (1.0, 0.5, 0.1, 3.0),
        'delta_times_cv':               (1.5, 0.8, 0.2, 4.0),
        'delta_times_nonzero_cv':       (1.2, 0.6, 0.1, 3.0),
        'velocities_cv':                (0.2, 0.1, 0.0, 0.5),
        'velocities_mean':              (80,  15,  30,  120),
        'pitches_mean':                 (60,  10,  30,  90),
        'pitches_std':                  (12,  5,   3,   25),
        'pitches_range':                (48,  15,  12,  80),
        'polyphony_max':                (8,   6,   1,   30),
        'polyphony_mean':               (3,   2,   0.5, 10),
        'polyphony_overlap_ratio':      (0.4, 0.25,0.0, 0.95),
        'prob_zero_duration_ratio':     (0.0, 0.02,0.0, 0.1),
        'prob_velocity_zero_ratio':     (0.0, 0.01,0.0, 0.05),
        'struct_notes_per_second':      (8,   6,   0.5, 40),
        'chords_ratio':                 (0.4, 0.2, 0.05,0.9),
        'dynamics_entropy':             (3.0, 1.0, 0.5, 5.5),
        'instruments_unique_count':     (3,   2,   1,   12),
        'phrases_avg_note_count':       (15,  10,  3,   60),
        'grid_rhythmic_entropy':        (3.0, 1.0, 1.0, 6.0),
        'pitches_step_ratio':           (0.35,0.15,0.1, 0.7),
        'pitches_leap_ratio':           (0.15,0.1, 0.0, 0.4),
        'dynamics_velocity_unique':     (15,  10,  1,   50),
    }

    def __init__(self, features, reference_ranges=None):
        self.f = features
        self.ref = reference_ranges or self.REFERENCE_RANGES
        self.anomalies = []

    def _add(self, feature, value, expected_range, z_score, direction):
        self.anomalies.append({
            'feature': feature,
            'value': value,
            'expected_range': expected_range,
            'z_score': z_score,
            'direction': direction  # 'high' or 'low'
        })

    def detect(self):
        """Detect anomalies using Z-score analysis against reference ranges."""
        self.anomalies = []
        
        for feature, (mean, std, min_exp, max_exp) in self.ref.items():
            value = self.f.get(feature)
            
            # Skip if feature is missing or NaN
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
                
            # Calculate Z-score
            if std > 0:
                z_score = (value - mean) / std
            else:
                z_score = 0.0 if value == mean else float('inf')
            
            # Check if outside expected range
            if value < min_exp:
                self._add(feature, value, (min_exp, max_exp), z_score, 'low')
            elif value > max_exp:
                self._add(feature, value, (min_exp, max_exp), z_score, 'high')
            
            # Flag significant Z-score anomalies (|z| > 2)
            elif abs(z_score) > 2.0:
                direction = 'high' if z_score > 0 else 'low'
                self._add(feature, value, (min_exp, max_exp), z_score, direction)
        
        # Cross-feature heuristics
        self._detect_cross_feature_anomalies()
        
        return self.anomalies

    def _detect_cross_feature_anomalies(self):
        """Detect anomalies based on relationships between features."""
        f = self.f
        
        # High density but poor alignment (likely corrupted import)
        nps = f.get('struct_notes_per_second', 0) or 0
        alignment = f.get('grid_start_alignment_ratio', 1) or 1
        if not (isinstance(nps, float) and np.isnan(nps)) and not (isinstance(alignment, float) and np.isnan(alignment)):
            if nps > 20 and alignment < 0.2:
                self._add('density_vs_alignment', f'{nps:.1f} nps / {alignment:.1%} alignment',
                          (0, 1), float('inf'), 'high')
        
        # Many notes but no phrases (stream of notes)
        total_notes = f.get('struct_total_notes', 0)
        phrases = f.get('phrases_count', 0)
        if total_notes > 500 and phrases <= 2:
            self._add('phrases_vs_notes', f'{total_notes} notes / {phrases} phrases',
                      (3, 60), float('inf'), 'low')
        
        # Extreme polyphony for melodic instrument
        is_drum = (f.get('instruments_drum_note_ratio', 0) or 0) > 0.8
        max_poly = f.get('polyphony_max', 0) or 0
        if not is_drum and max_poly > 20:
            self._add('melodic_polyphony', max_poly,
                      (1, 30), float('inf'), 'high')

    def get_anomaly_score(self):
        """Compute a composite anomaly score (higher = more anomalous)."""
        if not self.anomalies:
            return 0.0
        
        score = 0.0
        for a in self.anomalies:
            z = abs(a['z_score'])
            if z == float('inf'):
                score += 5.0
            else:
                score += min(z, 5.0)  # Cap individual contribution
        
        return score

    def get_report(self):
        lines = [f"MIDI Anomaly Report: {self.f.get('filename', 'unknown')}",
                 "=" * 60]
        
        if not self.anomalies:
            lines.append("No anomalies detected.")
            return '\n'.join(lines)
        
        # Sort by absolute Z-score (most anomalous first)
        sorted_anomalies = sorted(self.anomalies, 
                                  key=lambda x: abs(x['z_score']), reverse=True)
        
        for a in sorted_anomalies:
            z = a['z_score']
            if z == float('inf'):
                z_str = 'INF'
            else:
                z_str = f'{z:.2f}'
            
            lines.append(f"  [{a['direction'].upper()}] {a['feature']}: "
                         f"value={a['value']}, expected={a['expected_range']}, "
                         f"z-score={z_str}")
        
        lines.append(f"\nComposite Anomaly Score: {self.get_anomaly_score():.2f}")
        return '\n'.join(lines)


# =====================================================================
# ML ANOMALY DETECTOR — Isolation Forest based
# =====================================================================

class MLAnomalyDetector:
    """
    Machine Learning-based anomaly detection using Isolation Forest.
    Trains on a corpus of MIDI feature dicts, then flags outliers.
    """

    # Features most indicative of anomalies (avoiding highly correlated/redundant ones)
    ANOMALY_FEATURES = [
        'tempo_bpm_mean', 'tempo_bpm_std', 'tempo_bpm_range',
        'pace_notes_per_beat',
        'grid_start_alignment_ratio', 'grid_start_offset_mean', 'grid_syncopation_ratio',
        'harmonic_key_clarity', 'harmonic_consonance_ratio', 'harmonic_pitch_class_entropy',
        'durations_mean', 'durations_cv', 'durations_entropy',
        'delta_times_mean', 'delta_times_nonzero_cv', 'delta_times_zero_ratio',
        'pitches_mean', 'pitches_std', 'pitches_range',
        'pitches_step_ratio', 'pitches_leap_ratio',
        'velocities_mean', 'velocities_cv', 'velocities_entropy',
        'polyphony_max', 'polyphony_mean', 'polyphony_overlap_ratio',
        'prob_zero_duration_ratio', 'prob_velocity_zero_ratio',
        'prob_overlapping_same_pitch_ratio', 'prob_stuck_note_ratio',
        'struct_notes_per_second', 'struct_tempo_changes_per_second',
        'struct_total_notes', 'struct_track_count',
        'chords_ratio', 'chords_size_mean',
        'dynamics_entropy', 'dynamics_velocity_unique',
        'instruments_unique_count', 'instruments_channel_entropy',
        'phrases_count', 'phrases_avg_note_count',
    ]

    def __init__(self, feature_list=None):
        self.feature_list = feature_list or self.ANOMALY_FEATURES
        self.model = None
        self.scaler = None
        self.feature_names = []

    def _features_to_vector(self, features_dict):
        """Convert a feature dict to a numeric vector, filling NaN with 0."""
        vector = []
        for fname in self.feature_list:
            val = features_dict.get(fname, 0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                vector.append(0.0)
            else:
                vector.append(float(val))
        return vector

    def _prepare_matrix(self, features_list):
        """Prepare feature matrix from list of feature dicts."""
        matrix = []
        valid_indices = []
        
        for i, fdict in enumerate(features_list):
            # Skip files with parse errors
            if fdict.get('struct_parse_error') == 1 or fdict.get('error'):
                continue
            vector = self._features_to_vector(fdict)
            matrix.append(vector)
            valid_indices.append(i)
        
        return np.array(matrix), valid_indices

    def train(self, features_list, contamination=0.1, random_state=42):
        """
        Train Isolation Forest on a list of feature dicts.
        
        Args:
            features_list: List of feature dicts from MIDIFeatureExtractor
            contamination: Expected proportion of anomalies (0.0-0.5)
            random_state: Random seed for reproducibility
        """
        X, valid_indices = self._prepare_matrix(features_list)
        
        if len(X) < 10:
            raise ValueError(f"Need at least 10 valid MIDI files for training, got {len(X)}")
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
            max_samples='auto',
            bootstrap=False,
            verbose=2
        )
        self.model.fit(X_scaled)
        
        self.feature_names = self.feature_list
        return valid_indices

    def predict(self, features_dict):
        """
        Predict if a single MIDI file is anomalous.
        
        Returns:
            dict with keys:
                'is_anomaly': bool
                'anomaly_score': float (negative = more anomalous)
                'feature_contributions': dict of feature -> contribution score
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        vector = self._features_to_vector(features_dict)
        X = np.array([vector])
        X_scaled = self.scaler.transform(X)
        
        # Predict: -1 = anomaly, 1 = normal
        prediction = self.model.predict(X_scaled)[0]
        
        # Score: negative = more anomalous
        score = self.model.decision_function(X_scaled)[0]
        
        # Feature contributions (how much each feature deviates from mean)
        deviations = np.abs(X_scaled[0])
        
        contributions = {}
        for i, fname in enumerate(self.feature_names):
            if i < len(deviations):
                contributions[fname] = float(deviations[i])
        
        # Sort by contribution (highest deviation first)
        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            'is_anomaly': prediction == -1,
            'anomaly_score': float(score),
            'feature_contributions': sorted_contributions
        }

    def predict_batch(self, features_list):
        """
        Predict anomalies for a batch of MIDI files.
        
        Returns:
            List of prediction dicts (same format as predict())
            Only includes files that were parseable
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X, valid_indices = self._prepare_matrix(features_list)
        
        if len(X) == 0:
            return [], []
        
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        results = []
        for i in range(len(X)):
            deviations = np.abs(X_scaled[i])
            contributions = {}
            for j, fname in enumerate(self.feature_names):
                if j < len(deviations):
                    contributions[fname] = float(deviations[j])
            
            sorted_contributions = dict(
                sorted(contributions.items(), key=lambda x: x[1], reverse=True)
            )
            
            results.append({
                'is_anomaly': predictions[i] == -1,
                'anomaly_score': float(scores[i]),
                'feature_contributions': sorted_contributions,
                'filename': features_list[valid_indices[i]].get('filename', f'file_{valid_indices[i]}')
            })
        
        return results, valid_indices

    def get_anomaly_summary(self, results):
        """Summarize batch prediction results."""
        total = len(results)
        anomalies = sum(1 for r in results if r['is_anomaly'])
        
        if total == 0:
            return {"total": 0, "anomalies": 0, "anomaly_rate": 0}
        
        scores = [r['anomaly_score'] for r in results]
        
        return {
            'total': total,
            'anomalies': anomalies,
            'anomaly_rate': anomalies / total,
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'score_min': float(np.min(scores)),
            'score_max': float(np.max(scores)),
            'most_anomalous': min(results, key=lambda x: x['anomaly_score'])['filename'],
            'least_anomalous': max(results, key=lambda x: x['anomaly_score'])['filename'],
        }


# =====================================================================
# BATCH PROCESSOR
# =====================================================================

def process_midi_directory(directory_path, filter_channel=None, max_files=None):
    """
    Process all MIDI files in a directory and extract features.
    
    Returns:
        list of feature dicts
        list of filenames that failed to parse
    """
    import glob
    
    midi_files = glob.glob(os.path.join(directory_path, '**', '*.mid'), recursive=True)
    midi_files += glob.glob(os.path.join(directory_path, '**', '*.midi'), recursive=True)
    
    if max_files:
        midi_files = midi_files[:max_files]
    
    features_list = []
    errors = []
    
    for filepath in tqdm.tqdm(midi_files):
        try:
            extractor = MIDIFeatureExtractor(filepath, filter_channel=filter_channel)
            features = extractor.extract_all()
            features_list.append(features)
            
            if features.get('error') or features.get('struct_parse_error') == 1:
                errors.append({
                    'filename': os.path.basename(filepath),
                    'error': features.get('error', 'Unknown error')
                })
        except Exception as e:
            errors.append({
                'filename': os.path.basename(filepath),
                'error': str(e)
            })
    
    return features_list, errors


def full_analysis(midi_filepath, ml_detector=None):
    """
    Perform complete analysis of a single MIDI file:
    feature extraction, problem detection, and anomaly detection.
    
    Args:
        midi_filepath: Path to MIDI file
        ml_detector: Optional trained MLAnomalyDetector instance
    
    Returns:
        Comprehensive analysis dict
    """
    # Extract features
    extractor = MIDIFeatureExtractor(midi_filepath)
    features = extractor.extract_all()
    
    if features.get('error'):
        return {
            'filename': os.path.basename(midi_filepath),
            'error': features['error'],
            'features': features,
            'problems': [],
            'statistical_anomalies': [],
            'ml_anomaly': None
        }
    
    # Rule-based problem detection
    problem_detector = MIDIProblemDetector(features)
    problems = problem_detector.detect()
    
    # Statistical anomaly detection
    anomaly_detector = MIDIAnomalyDetector(features)
    anomalies = anomaly_detector.detect()
    
    # ML-based anomaly detection (if model provided)
    ml_result = None
    if ml_detector is not None:
        try:
            ml_result = ml_detector.predict(features)
        except Exception as e:
            ml_result = {'error': str(e)}
    
    return {
        'filename': os.path.basename(midi_filepath),
        'features': features,
        'problems': problems,
        'problem_summary': problem_detector.get_severity_summary(),
        'is_problematic': problem_detector.is_problematic(),
        'statistical_anomalies': anomalies,
        'anomaly_score': anomaly_detector.get_anomaly_score(),
        'ml_anomaly': ml_result,
        'is_anomalous': anomaly_detector.get_anomaly_score() > 10.0 or 
                        (ml_result and ml_result.get('is_anomaly', False))
    }


def generate_analysis_report(analysis):
    """Generate a human-readable analysis report."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"MIDI ANALYSIS REPORT: {analysis['filename']}")
    lines.append("=" * 70)
    
    if analysis.get('error'):
        lines.append(f"\nFATAL ERROR: {analysis['error']}")
        return '\n'.join(lines)
    
    # Problem Summary
    lines.append("\n--- PROBLEM DETECTION ---")
    problems = analysis.get('problems', [])
    if problems:
        for sev in ['critical', 'warning', 'info']:
            issues = [p for p in problems if p['severity'] == sev]
            if issues:
                lines.append(f"\n  [{sev.upper()}]")
                for p in issues:
                    lines.append(f"    [{p['category']}] {p['issue']}: {p['detail']}")
    else:
        lines.append("  No problems detected.")
    
    ps = analysis.get('problem_summary', {})
    lines.append(f"\n  Summary: {ps.get('critical', 0)} critical, "
                 f"{ps.get('warning', 0)} warnings, {ps.get('info', 0)} info")
    lines.append(f"  Is Problematic: {'YES' if analysis.get('is_problematic') else 'NO'}")
    
    # Statistical Anomalies
    lines.append("\n--- STATISTICAL ANOMALY DETECTION ---")
    anomalies = analysis.get('statistical_anomalies', [])
    if anomalies:
        sorted_anomalies = sorted(anomalies, key=lambda x: abs(x['z_score']), reverse=True)
        for a in sorted_anomalies[:10]:  # Top 10
            z = a['z_score']
            z_str = 'INF' if z == float('inf') else f'{z:.2f}'
            lines.append(f"  [{a['direction'].upper()}] {a['feature']}: "
                        f"value={a['value']}, expected={a['expected_range']}, z={z_str}")
    else:
        lines.append("  No statistical anomalies detected.")
    
    lines.append(f"\n  Composite Anomaly Score: {analysis.get('anomaly_score', 0):.2f}")
    
    # ML Anomaly Detection
    ml = analysis.get('ml_anomaly')
    if ml:
        lines.append("\n--- ML-BASED ANOMALY DETECTION ---")
        if ml.get('error'):
            lines.append(f"  Error: {ml['error']}")
        else:
            lines.append(f"  Is Anomaly: {'YES' if ml.get('is_anomaly') else 'NO'}")
            lines.append(f"  Anomaly Score: {ml.get('anomaly_score', 0):.4f}")
            
            # Top contributing features
            contributions = ml.get('feature_contributions', {})
            top_features = list(contributions.items())[:5]
            if top_features:
                lines.append("  Top Contributing Features:")
                for fname, contrib in top_features:
                    lines.append(f"    {fname}: {contrib:.3f}")
    
    # Final Verdict
    lines.append("\n--- FINAL VERDICT ---")
    if analysis.get('is_anomalous'):
        lines.append("  ⚠️  THIS MIDI FILE IS FLAGGED AS ANOMALOUS/PROBLEMATIC")
    else:
        lines.append("  ✓  This MIDI file appears normal")
    
    return '\n'.join(lines)

###################################################################################

# =============================================================================
# High-precision MIDI timing anomaly detector
# Voice-aware + grid-residual based. Adapted for expressive human performances.
# =============================================================================

# =============================================================================
# Tempo map construction
# =============================================================================
def build_tempo_map(score, verbose=False, show_progress_bar=False):
    """Extract a de-duplicated, sorted list of (tick, tempo_us) tempo changes.

    Walks every track of a parsed MIDI score and collects `set_tempo` events.
    If none are present, defaults to 500000 µs/quarter (120 BPM). The first
    event is forced to occur at tick 0 so subsequent tick→ms conversion has a
    valid anchor for any note in the file.

    Parameters
    ----------
    score : list
        Score list as returned by `MIDI.midi2score`. Index 0 holds
        `ticks_per_quarter`; subsequent indices are per-track event lists.
    verbose : bool, default False
        If True, prints the number of tempo changes discovered and the
        equivalent BPM range.
    show_progress_bar : bool, default False
        If True, wraps the per-track iteration with a `tqdm` progress bar.

    Returns
    -------
    list[tuple[int, int]]
        Sorted, de-duplicated list of `(tick, tempo_microseconds)` pairs,
        guaranteed to start at tick 0.
    """
    tempo_changes = []
    tracks = range(1, len(score))
    if show_progress_bar:
        tracks = tqdm(tracks, desc="Scanning tempo map", leave=False)
    for itrack in tracks:
        for ev in score[itrack]:
            if ev[0] == 'set_tempo':
                tempo_changes.append((ev[1], ev[2]))

    if not tempo_changes:
        tempo_changes = [(0, 500000)]
    tempo_changes.sort(key=lambda x: x[0])

    # Anchor at tick 0
    if tempo_changes[0][0] > 0:
        tempo_changes.insert(0, (0, tempo_changes[0][1]))

    # De-duplicate by tick (keep first occurrence)
    seen, deduped = set(), []
    for t, tp in tempo_changes:
        if t not in seen:
            seen.add(t)
            deduped.append((t, tp))

    if verbose:
        bpms = [60_000_000 / tp for _, tp in deduped]
        print(f"[tempo] {len(deduped)} tempo change(s); "
              f"BPM range {min(bpms):.2f}–{max(bpms):.2f}")
    return deduped


def ticks_to_ms_function(tempo_changes, ticks_per_quarter):
    """Build a closure mapping MIDI ticks to absolute time in milliseconds.

    Precomputes per-segment cumulative millisecond offsets, then performs a
    binary search at call time to find the active tempo segment and add the
    proportional intra-segment contribution.

    Parameters
    ----------
    tempo_changes : list[tuple[int, int]]
        Sorted `(tick, tempo_us)` pairs from `build_tempo_map`.
    ticks_per_quarter : int
        PPQ resolution from the MIDI header.

    Returns
    -------
    callable
        `f(tick: int) -> float` returning milliseconds since the start.
    """
    boundaries, cum_ms = [], 0.0
    for i, (tick, tempo_us) in enumerate(tempo_changes):
        boundaries.append({'tick': tick, 'tempo_us': tempo_us, 'cum_ms': cum_ms})
        if i + 1 < len(tempo_changes):
            delta_ticks = tempo_changes[i + 1][0] - tick
            cum_ms += delta_ticks * (tempo_us / 1000.0) / ticks_per_quarter

    def f(tick_query):
        lo, hi = 0, len(boundaries) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if boundaries[mid]['tick'] <= tick_query:
                lo = mid + 1
            else:
                hi = mid - 1
        seg = boundaries[max(0, hi)]
        delta = tick_query - seg['tick']
        return seg['cum_ms'] + delta * (seg['tempo_us'] / 1000.0) / ticks_per_quarter

    return f


def robust_mad(data):
    """Median Absolute Deviation scaled to be ~consistent with std-deviation.

    Uses the standard 1.4826 consistency constant for Gaussian data. Returns
    0.0 for empty input.

    Parameters
    ----------
    data : iterable[float]
        Numeric samples.

    Returns
    -------
    float
        Scaled MAD (≥ 0).
    """
    if not data:
        return 0.0
    med = statistics.median(data)
    return statistics.median([abs(x - med) for x in data]) * 1.4826


# =============================================================================
# Subdivision estimation
# =============================================================================
def estimate_subdivision(ioi_values_ms, quarter_ms,
                         candidates=(1, 2, 3, 4, 6, 8, 12, 16)):
    """Pick the subdivision whose grid best explains the IOI distribution.

    For each candidate `s` we form a grid `quarter_ms / s`, then check what
    fraction of IOIs are close to an integer multiple of that grid. A small
    prior bonus is given to conventional subdivisions (4, 8, 16) to avoid
    spurious choices on very rubato material. If no candidate reaches a
    satisfactory fit, the function falls back to 4 subdivisions per quarter.

    Parameters
    ----------
    ioi_values_ms : iterable[float]
        All IOIs (in ms) across voices; values ≤ 5 ms are discarded.
    quarter_ms : float
        Duration of one quarter note in ms (evaluated at the initial tempo).
    candidates : tuple[int], optional
        Subdivisions per quarter to evaluate.

    Returns
    -------
    tuple[int, float]
        `(best_subdivision, grid_ms)` where `grid_ms = quarter_ms / best_sub`.
    """
    vals = np.array([v for v in ioi_values_ms if v > 5.0])
    if len(vals) == 0:
        return 4, quarter_ms / 4.0

    best_score, best_sub = -1.0, 4
    for s in candidates:
        grid = quarter_ms / s
        ratios = vals / grid
        # Keep fractional ratios in a sane range; do *not* filter <0.5 because
        # coarse grids that produce tiny fractional ratios should be penalised.
        valid_ratios = ratios[(ratios >= 0.1) & (ratios <= 32)]
        if len(valid_ratios) == 0:
            continue
        residuals = np.abs(valid_ratios - np.round(valid_ratios))
        frac_close = float(np.mean(residuals < 0.15))
        prior = 1.0 if s in (4, 8, 16) else 0.9
        score = frac_close * prior
        if score > best_score:
            best_score, best_sub = score, s

    if best_score < 0.4:
        best_sub = 4
    return best_sub, quarter_ms / best_sub


def local_expected_ms_for_ioi(tick2ms, ticks_per_quarter, start_tick, subdivision):
    """Return the expected duration (ms) of one grid step starting at `start_tick`.

    The duration is computed in tick-space then converted via the tempo map so
    that tempo changes are honoured. This is the local "unit grid" against
    which IOIs are expressed as integer multiples.

    Parameters
    ----------
    tick2ms : callable
        Tick→ms closure from `ticks_to_ms_function`.
    ticks_per_quarter : int
        PPQ resolution.
    start_tick : int
        Tick at which the previous note began (the IOI's origin).
    subdivision : int
        Number of grid steps per quarter note.

    Returns
    -------
    float
        Local grid duration in ms (clamped to ≥ 0.0001 to avoid div-by-zero).
    """
    ticks_per_sub = ticks_per_quarter / subdivision
    return max(0.0001, tick2ms(start_tick + ticks_per_sub) - tick2ms(start_tick))


# =============================================================================
# Voice-aware IOI extraction
# =============================================================================
def compute_voice_iois(notes, tick2ms, min_ioi_ms=8.0,
                      verbose=False, show_progress_bar=False):
    """Group notes into monophonic voices and compute inter-onset intervals.

    A *voice* is defined as the set of notes sharing the same
    `(track, channel)` pair. Within each voice, notes are sorted by
    `(start_tick, note)` and consecutive onset differences are recorded as
    IOIs. IOIs shorter than `min_ioi_ms` (retriggers / within-chord) or
    longer than 5000 ms (phrase boundaries) are discarded.

    Parameters
    ----------
    notes : list[dict]
        Note dictionaries with at least `start_tick`, `start_ms`, `track`,
        and `channel` keys.
    tick2ms : callable
        Tick→ms closure (unused here but kept for API symmetry).
    min_ioi_ms : float, default 8.0
        Minimum IOI duration to retain.
    verbose : bool, default False
        If True, prints per-voice IOI counts.
    show_progress_bar : bool, default False
        Wrap the per-voice loop with a progress bar.

    Returns
    -------
    dict[tuple[int,int], list[dict]]
        Mapping `(track, channel) -> list[ioi_dict]`. Each `ioi_dict` has
        `index_in_voice`, `ioi_ms`, `prev_note`, `cur_note`, `voice`.
    """
    voices = defaultdict(list)
    for n in notes:
        voices[(n['track'], n['channel'])].append(n)

    voice_iois = {}
    voice_iter = voices.items()
    if show_progress_bar:
        voice_iter = tqdm(voice_iter, desc="Extracting voice IOIs", leave=False)
    for key, vnotes in voice_iter:
        vnotes.sort(key=lambda x: (x['start_tick'], x['note']))
        iois = []
        for i in range(1, len(vnotes)):
            prev, cur = vnotes[i - 1], vnotes[i]
            ioi = cur['start_ms'] - prev['start_ms']
            if ioi < min_ioi_ms or ioi > 5000.0:
                continue
            iois.append({
                'index_in_voice': len(iois),
                'ioi_ms': ioi,
                'prev_note': prev,
                'cur_note': cur,
                'voice': key,
            })
        voice_iois[key] = iois
        if verbose:
            print(f"[voice {key}] {len(iois)} IOIs")
    return voice_iois


# =============================================================================
# Residual-based anomaly detection (performance-adaptive)
# =============================================================================
def detect_anomalies_voice(voice_iois, tick2ms, ticks_per_quarter, subdivision,
                          max_multiple=16,
                          z_thresh=4.0,
                          min_abs_for_z=35.0,
                          jitter_abs_ms=40.0, jitter_rel=0.20,
                          abs_dev_ms=50.0,
                          verbose=False):
    """Flag IOIs whose grid-residual is a statistical outlier for this voice.

    For each IOI we compute the ratio between its measured duration and the
    local grid step (`quarter_ms / subdivision`, evaluated at the IOI's
    origin tick). Ratios outside `[0.75, max_multiple]` are skipped (grace
    notes, long rests). The remaining residuals `ioi_ms - round(ratio) * grid`
    form a per-voice distribution whose median and MAD are used as a
    performance-adaptive baseline. An IOI is flagged if either:

    * its residual is a strong z-outlier (`|z| > z_thresh` AND
      `|residual| > min_abs_for_z` ms), **or**
    * the local 8-IOI window shows sustained jitter (`std > jitter_abs_ms`
      AND `std > jitter_rel * local_grid`) AND the residual magnitude
      exceeds `abs_dev_ms`.

    Parameters
    ----------
    voice_iois : list[dict]
        IOI records for a single voice (output of `compute_voice_iois`).
    tick2ms : callable
        Tick→ms closure.
    ticks_per_quarter : int
        PPQ resolution.
    subdivision : int
        Estimated subdivisions per quarter (see `estimate_subdivision`).
    max_multiple : int, default 16
        Largest integer grid-multiple considered plausible.
    z_thresh : float, default 4.0
        Robust z-score threshold for outlier flagging.
    min_abs_for_z : float, default 35.0
        Minimum residual magnitude (ms) required to fire a z-outlier.
    jitter_abs_ms : float, default 40.0
        Absolute std threshold for the local-jitter test.
    jitter_rel : float, default 0.20
        Relative std threshold (`* local_grid`) for the local-jitter test.
    abs_dev_ms : float, default 50.0
        Minimum residual magnitude required when the local-jitter path fires.
    verbose : bool, default False
        If True, prints the per-voice robust statistics.

    Returns
    -------
    tuple[list[dict], dict]
        `(anomalies, thresholds)`. `anomalies` is the list of flagged IOI
        records (augmented with `residual_ms`, `z`, `std_window`, `reasons`).
        `thresholds` summarises the per-voice robust statistics and the
        detection parameters used.
    """
    if not voice_iois:
        return [], {}

    # First pass: per-IOI ratio + residual
    residuals = []
    for ioi in voice_iois:
        local_exp = local_expected_ms_for_ioi(
            tick2ms, ticks_per_quarter, ioi['prev_note']['start_tick'], subdivision)
        ratio = ioi['ioi_ms'] / local_exp if local_exp > 1e-6 else float('inf')
        ioi['local_expected_ms'] = local_exp
        ioi['ratio'] = ratio

        # Skip extreme rests and very short ornaments (e.g. grace notes)
        if ratio > max_multiple or ratio < 0.75:
            ioi['flag'] = False
            continue

        nearest = max(1, int(round(ratio)))
        ioi['nearest_int'] = nearest
        ioi['residual_ms'] = ioi['ioi_ms'] - nearest * local_exp
        residuals.append(ioi['residual_ms'])
        ioi['flag'] = True

    if not residuals:
        return [], {}

    med_res = statistics.median(residuals)
    mad_res = robust_mad(residuals)
    if mad_res < 1.0:
        mad_res = 1.0

    # Second pass: apply outlier tests with an 8-IOI rolling window
    anomalies, window = [], deque(maxlen=8)
    for ioi in voice_iois:
        if not ioi.get('flag', False):
            continue
        res = ioi['residual_ms']
        local_exp = ioi['local_expected_ms']
        window.append(res)
        std_w = statistics.pstdev(list(window)) if len(window) > 1 else 0.0
        z = (res - med_res) / mad_res

        reasons = []
        if abs(z) > z_thresh and abs(res) > min_abs_for_z:
            reasons.append('z_outlier')
        if std_w > jitter_abs_ms and std_w > jitter_rel * local_exp:
            reasons.append('local_jitter')

        if 'z_outlier' in reasons or ('local_jitter' in reasons and abs(res) > abs_dev_ms):
            ioi['reasons'] = reasons
            ioi['z'] = z
            ioi['std_window'] = std_w
            anomalies.append(ioi)

    thresholds = {
        'z_thresh': z_thresh,
        'min_abs_for_z': min_abs_for_z,
        'jitter_abs_ms': jitter_abs_ms,
        'jitter_rel': jitter_rel,
        'abs_dev_ms': abs_dev_ms,
        'med_residual': med_res,
        'mad_residual': mad_res,
        'n_iois_in_voice': len(voice_iois),
    }
    if verbose:
        print(f"[voice] n={len(voice_iois)} med_res={med_res:.2f}ms "
              f"mad_res={mad_res:.2f}ms flagged={len(anomalies)}")
    return anomalies, thresholds


# =============================================================================
# Main analysis
# =============================================================================
def analyze_midi_timings(midi_path, max_subdivision=16,
                         verbose=False, show_progress_bar=False):
    """Parse a MIDI file and run voice-aware, grid-residual anomaly detection.

    Pipeline:
      1. Parse the MIDI file and build a tempo-aware `tick → ms` map.
      2. Extract every `note` event into a normalised dict.
      3. Group notes into `(track, channel)` voices and compute IOIs.
      4. Estimate the dominant subdivision per quarter from the IOI
         distribution.
      5. Per voice, compute grid residuals and apply robust outlier tests.

    Parameters
    ----------
    midi_path : str
        Path to a `.mid` file.
    max_subdivision : int, default 16
        Upper bound on the integer grid-multiple considered plausible.
    verbose : bool, default False
        Propagated to tempo/voice/anomaly subroutines for diagnostic prints.
    show_progress_bar : bool, default False
        Propagated to subroutines; wraps loops with `tqdm` bars.

    Returns
    -------
    dict or None
        Analysis report. `None` if the file contains no notes. Keys include
        `num_notes`, `num_iois`, `num_voices`, `estimated_subdivision`,
        `expected_grid_ms_nominal`, IOI statistics, `anomalies_all`,
        `anomalies_by_voice`, `thresholds_by_voice`, `voice_iois`, `notes`,
        `tick2ms`, `ticks_per_quarter`, `tempo_changes`.
    """
    with open(midi_path, 'rb') as f:
        score = MIDI.midi2score(f.read())

    ticks_per_quarter = score[0]
    tempo_changes = build_tempo_map(
        score, verbose=verbose, show_progress_bar=show_progress_bar)
    tick2ms = ticks_to_ms_function(tempo_changes, ticks_per_quarter)

    # ----- Extract note events -----
    notes = []
    track_iter = range(1, len(score))
    if show_progress_bar:
        track_iter = tqdm(track_iter, desc="Extracting notes", leave=False)
    for itrack in track_iter:
        for ev in score[itrack]:
            if ev[0] == 'note':
                notes.append({
                    'start_tick': ev[1],
                    'start_ms':   tick2ms(ev[1]),
                    'dur_tick':   ev[2],
                    'channel':    ev[3],
                    'note':        ev[4],
                    'vel':         ev[5],
                    'track':       itrack,
                })
    if not notes:
        if verbose:
            print("No notes found.")
        return None

    notes.sort(key=lambda x: (x['start_tick'], x['note']))
    for n in notes:
        n['duration_ms'] = (tick2ms(n['start_tick'] + n['dur_tick'])
                            - tick2ms(n['start_tick']))

    # ----- Voice IOIs -----
    voice_iois = compute_voice_iois(
        notes, tick2ms, min_ioi_ms=8.0,
        verbose=verbose, show_progress_bar=show_progress_bar)

    all_iois = [ioi['ioi_ms'] for v in voice_iois.values() for ioi in v]
    quarter_ms = tempo_changes[0][1] / 1000.0
    est_sub, expected_grid_ms = estimate_subdivision(all_iois, quarter_ms)
    if verbose:
        print(f"[grid] estimated subdivision/quarter = {est_sub} "
              f"(grid = {expected_grid_ms:.3f} ms)")

    # ----- Per-voice anomaly detection -----
    anomalies_by_voice, thresholds_by_voice, all_anomalies = {}, {}, []
    voice_iter = voice_iois.items()
    if show_progress_bar:
        voice_iter = tqdm(voice_iter, desc="Detecting anomalies", leave=False)
    for voice, iois in voice_iter:
        an, th = detect_anomalies_voice(
            iois, tick2ms, ticks_per_quarter, est_sub,
            max_multiple=max_subdivision, verbose=verbose)
        anomalies_by_voice[voice] = an
        thresholds_by_voice[voice] = th
        all_anomalies.extend(an)

    all_anomalies.sort(key=lambda a: abs(a['residual_ms']), reverse=True)

    return {
        'file': midi_path,
        'num_notes': len(notes),
        'num_iois': len(all_iois),
        'num_voices': len(voice_iois),
        'estimated_subdivision': est_sub,
        'expected_grid_ms_nominal': expected_grid_ms,
        'mean_ioi_ms':   statistics.mean(all_iois)   if all_iois else 0.0,
        'median_ioi_ms': statistics.median(all_iois) if all_iois else 0.0,
        'std_ioi_ms':    statistics.pstdev(all_iois) if all_iois else 0.0,
        'mad_ioi_ms':    robust_mad(all_iois),
        'anomalies_all': all_anomalies,
        'anomalies_by_voice': anomalies_by_voice,
        'thresholds_by_voice': thresholds_by_voice,
        'voice_iois': voice_iois,
        'notes': notes,
        'tick2ms': tick2ms,
        'ticks_per_quarter': ticks_per_quarter,
        'tempo_changes': tempo_changes,
    }


# =============================================================================
# Reporting — text
# =============================================================================
def print_summary_report(report, top_n=20):
    """Print the high-level summary block of a timing-analysis report.

    Includes file path, note/voice/IOI counts, estimated subdivision, nominal
    grid duration, IOI distribution statistics, and the total number of
    flagged anomalies.

    Parameters
    ----------
    report : dict
        Report returned by `analyze_midi_timings`.
    top_n : int, default 20
        Maximum number of top anomalies to list (by |residual|).

    Returns
    -------
    None
    """
    print("High-precision MIDI Timing Analysis (voice-aware, grid-residual)")
    print("---------------------------------------------------------------")
    print(f"File                         : {report['file']}")
    print(f"Notes                        : {report['num_notes']}")
    print(f"Voices (track,channel)       : {report['num_voices']}")
    print(f"IOIs (voice-filtered)        : {report['num_iois']}")
    print(f"Estimated subdivision/quarter: {report['estimated_subdivision']}")
    print(f"Nominal grid (ms @ tick 0)   : {report['expected_grid_ms_nominal']:.3f}")
    print(f"Mean IOI (ms)                : {report['mean_ioi_ms']:.3f}")
    print(f"Median IOI (ms)              : {report['median_ioi_ms']:.3f}")
    print(f"Std IOI (ms)                 : {report['std_ioi_ms']:.3f}")
    print(f"MAD IOI (ms)                 : {report['mad_ioi_ms']:.3f}")
    print(f"Flagged anomalies (total)    : {len(report['anomalies_all'])}")
    print()


def print_top_anomalies(report, top_n=20):
    """Pretty-print the `top_n` most extreme flagged anomalies.

    Each row shows the voice, measured IOI, local grid duration, integer
    grid-multiple, signed residual, robust z-score, local jitter (std),
    and the active reason tags.

    Parameters
    ----------
    report : dict
        Report returned by `analyze_midi_timings`.
    top_n : int, default 20
        Number of anomalies to print (sorted by |residual| descending).

    Returns
    -------
    None
    """
    if not report['anomalies_all']:
        print("No anomalies flagged. ✅")
        return
    print("Top anomalies (ranked by |residual|):")
    print(f"{'voice':>10} {'ioi_ms':>9} {'grid_ms':>9} {'mult':>5} "
          f"{'resid_ms':>9} {'z':>6} {'jitter':>7}  reasons")
    for a in report['anomalies_all'][:top_n]:
        print(f"{str(a['voice']):>10} {a['ioi_ms']:9.2f} "
              f"{a['local_expected_ms']:9.2f} {a['nearest_int']:>5d} "
              f"{a['residual_ms']:+9.2f} {a.get('z', 0):6.2f} "
              f"{a.get('std_window', 0):7.2f}  {','.join(a['reasons'])}")


def print_per_voice_report(report):
    """Print a compact per-voice summary: IOI count, anomaly count, thresholds.

    Parameters
    ----------
    report : dict
        Report returned by `analyze_midi_timings`.

    Returns
    -------
    None
    """
    print("\nPer-voice anomaly counts:")
    print(f"{'voice':>10} {'#iois':>7} {'#anom':>6}  thresholds")
    for voice, an in sorted(report['anomalies_by_voice'].items(),
                            key=lambda kv: -len(kv[1])):
        th = report['thresholds_by_voice'][voice]
        n_iois = th.get('n_iois_in_voice', 0)
        print(f"{str(voice):>10} {n_iois:>7d} {len(an):>6d}  "
              f"z={th['z_thresh']:.1f} jit={th['jitter_abs_ms']:.0f}ms "
              f"med_res={th['med_residual']:.1f}ms")


# =============================================================================
# Reporting — plot
# =============================================================================
def plot_timing_analysis(report, max_ioi_quantile=0.95, figsize=(15, 9)):
    """Plot IOI timeline, local grid contour, and the residual anomaly signal.

    Two stacked subplots:

    * **Top — IOI timeline.** Each IOI is drawn as a scatter dot, coloured
      by its `nearest_int` grid-multiple. The local grid duration is drawn
      as a step line through every IOI (so tempo/rubato contour is visible).
      Horizontal dashed reference lines mark `k * nominal_grid` for
      k = 1..8 with inline labels. Flagged anomalies are overlaid as red
      `x` markers.

    * **Bottom — Residual signal.** The signed residual
      `ioi_ms - nearest_int * local_grid` over time, with shaded ±z-threshold
      bands derived from each voice's robust MAD. Anomalies are again marked
      in red. This subplot makes the outlier logic visually self-evident.

    Parameters
    ----------
    report : dict
        Report returned by `analyze_midi_timings`.
    max_ioi_quantile : float, default 0.95
        Quantile of the IOI distribution used to set the top subplot's y-axis
        upper bound (with 1.25× headroom).
    figsize : tuple[float, float], default (15, 9)
        Matplotlib figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure (also displayed inline in Jupyter).
    """
    g = report['expected_grid_ms_nominal']

    # Flatten & time-sort IOIs across all voices
    all_iois_flat = [ioi for v in report['voice_iois'].values() for ioi in v]
    all_iois_flat.sort(key=lambda x: x['cur_note']['start_ms'])
    if not all_iois_flat:
        print("No IOIs to plot.")
        return None

    xs = np.array([ioi['cur_note']['start_ms'] / 1000.0 for ioi in all_iois_flat])
    ys = np.array([ioi['ioi_ms'] for ioi in all_iois_flat])
    mults = np.array([ioi.get('nearest_int', 0) for ioi in all_iois_flat])
    grid_vals = np.array([ioi.get('local_expected_ms', g) for ioi in all_iois_flat])

    # Anomalies
    anom = report['anomalies_all']
    anom_x = np.array([a['cur_note']['start_ms'] / 1000.0 for a in anom])
    anom_y = np.array([a['ioi_ms'] for a in anom])
    anom_r = np.array([a['residual_ms'] for a in anom])

    # Residuals + threshold bands (use a single voice-wide MAD summary)
    resids = np.array([ioi.get('residual_ms', 0.0) for ioi in all_iois_flat])
    mads = [th['mad_residual'] for th in report['thresholds_by_voice'].values()
            if th]
    z_thr = next(iter(report['thresholds_by_voice'].values()))['z_thresh'] \
            if report['thresholds_by_voice'] else 4.0
    med_res = (statistics.median([th['med_residual'] for th in
                            report['thresholds_by_voice'].values() if th])
               if mads else 0.0)
    mad_res = max(statistics.median(mads), 1.0) if mads else 1.0
    band = z_thr * mad_res

    # ----- Figure -----
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize,
                                         sharex=True, height_ratios=[2.5, 1])

    # --- Top: IOI timeline ---
    # Use scatter (no connecting lines) to avoid the previous "spaghetti"
    scatter = ax_top.scatter(xs, ys, c=mults, cmap='viridis',
                             s=18, alpha=0.7, label='IOI (colored by grid multiple)')
    # Continuous local-grid step line through every IOI
    ax_top.step(xs, grid_vals, where='post', color='C2', alpha=0.85,
                linewidth=1.4, label='Local expected grid (per IOI)')
    # Anomaly overlay
    if len(anom_x):
        ax_top.scatter(anom_x, anom_y, facecolors='none', edgecolors='red',
                       s=90, linewidths=1.8, marker='o',
                       label='Flagged anomalies', zorder=5)
    # k * grid reference lines with labels
    y_max_data = np.quantile(ys, max_ioi_quantile) if len(ys) else g * 8
    y_max = max(g * 8.5, float(y_max_data) * 1.25)
    for k in range(1, 9):
        yk = k * g
        if yk > y_max:
            break
        ax_top.axhline(yk, color='gray', alpha=0.25, linestyle='--', linewidth=0.8)
        ax_top.text(xs[-1] * 0.995, yk, f' {k}×grid',
                    va='center', ha='right', fontsize=8, color='dimgray')

    ax_top.set_ylim(0, y_max)
    ax_top.set_ylabel('IOI (ms)')
    ax_top.set_title('MIDI IOIs with local-grid anomalies (voice-aware)')
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc='upper right', fontsize=9)
    cbar = fig.colorbar(scatter, ax=ax_top, pad=0.01)
    cbar.set_label('Nearest grid multiple')

    # --- Bottom: residual signal ---
    ax_bot.axhline(0, color='black', linewidth=0.8, alpha=0.6)
    ax_bot.fill_between(xs, med_res - band, med_res + band,
                        color='red', alpha=0.10,
                        label=f'±{z_thr:.0f}·MAD band (z-threshold)')
    ax_bot.plot(xs, resids, color='C0', alpha=0.55, linewidth=0.9)
    ax_bot.scatter(xs, resids, s=12, color='C0', alpha=0.7)
    if len(anom_x):
        ax_bot.scatter(anom_x, anom_r, color='red', s=55, marker='x',
                       linewidths=1.8, label='Flagged anomalies', zorder=5)
    ax_bot.set_xlabel('Time (seconds)')
    ax_bot.set_ylabel('Residual (ms)')
    ax_bot.set_title('Grid-residual signal (signed)')
    ax_bot.grid(alpha=0.25)
    ax_bot.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    plt.show()
    return fig

###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the MIDI Ano Python module
###################################################################################