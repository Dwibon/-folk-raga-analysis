"""
segment_dataset.py

Splits each recording in the dataset into fixed-length, overlapping
segments, optionally trimming the first/last few seconds of the
original recording first, and writes a new segment-level metadata CSV
alongside the existing recording-level one.

Segment size / overlap (defaults):
    20s segments, 50% overlap (10s hop). This is standard MIR practice -
    long enough per segment to capture meaningful melodic/phrase content,
    with enough overlap that content near a segment boundary isn't lost
    entirely from every window.

Edge trimming ("if only required"):
    Removes the first and last EDGE_TRIM_SEC seconds of each recording
    ONLY IF doing so still leaves enough audio for at least one full
    segment afterward. Short clips are left untrimmed automatically -
    you don't need to decide this per file.

Output:
    - Segmented WAV files under dataset_segments/<genre>/<segment_id>.wav
    - data/segment_metadata.csv - one row per segment, carrying over every
      column from the original recording-level metadata (genre, singer,
      gender, subgenre, etc.) plus segment-specific fields.

Requires: pip install librosa soundfile pandas numpy tqdm
          (all already in your existing environment)

Usage:
    python segment_dataset.py
"""

import os
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────
METADATA_PATH    = "dataset/metadata.csv"    # matches your actual layout
AUDIO_DIRS_ROOT   = "dataset"                # dataset/<genre>/<recording_id>.wav
OUTPUT_AUDIO_DIR  = "dataset_segments"       # segments written here, mirrored by genre
OUTPUT_METADATA   = "dataset/segment_metadata.csv"

SEGMENT_SEC = 20.0                         # segment length
OVERLAP_SEC = 10.0                         # overlap between consecutive segments (50%)
HOP_SEC     = SEGMENT_SEC - OVERLAP_SEC    # 10s hop

EDGE_TRIM_SEC          = 5.0    # trim this much off start/end, only if safe to do so
SILENCE_RMS_THRESHOLD  = 0.01   # flag (not remove) segments quieter than this

TARGET_SR = 22050
# ─────────────────────────────────────────────────────────


def build_genre_folder_map(audio_root):
    """Maps lowercased genre name -> actual folder name on disk, so a
    metadata value like 'Kamrupi' still finds a folder named 'kamrupi'
    (or any other casing) without needing an exact case match."""
    folder_map = {}
    if os.path.isdir(audio_root):
        for entry in os.listdir(audio_root):
            if os.path.isdir(os.path.join(audio_root, entry)):
                folder_map[entry.lower()] = entry
    return folder_map


GENRE_FOLDER_MAP = build_genre_folder_map(AUDIO_DIRS_ROOT)


def resolve_genre_folder(genre):
    """Returns the actual on-disk folder name for a given genre value,
    case-insensitively. Falls back to the raw genre string if no match
    is found (so the original error message still makes sense)."""
    return GENRE_FOLDER_MAP.get(str(genre).lower(), genre)


def find_audio_path(recording_id, genre):
    filename = recording_id if str(recording_id).lower().endswith(".wav") else f"{recording_id}.wav"
    actual_folder = resolve_genre_folder(genre)
    path = os.path.join(AUDIO_DIRS_ROOT, actual_folder, filename)
    return path if os.path.exists(path) else None


def maybe_trim_edges(y, sr, edge_trim_sec=EDGE_TRIM_SEC, segment_sec=SEGMENT_SEC):
    """Trims edge_trim_sec off both ends, but only if the remaining audio
    is still long enough for at least one full segment. Otherwise returns
    the audio unchanged."""
    total_sec = len(y) / sr
    remaining_sec = total_sec - 2 * edge_trim_sec

    if remaining_sec >= segment_sec:
        start = int(edge_trim_sec * sr)
        end = len(y) - int(edge_trim_sec * sr)
        return y[start:end], True
    return y, False


def segment_audio(y, sr, segment_sec=SEGMENT_SEC, hop_sec=HOP_SEC):
    """Yields (segment_audio, start_sec, end_sec) for each window."""
    seg_len = int(segment_sec * sr)
    hop_len = int(hop_sec * sr)

    if len(y) < seg_len:
        # Too short for even one full segment - keep the whole thing as
        # a single short segment rather than discarding it silently.
        yield y, 0.0, len(y) / sr
        return

    start = 0
    while start + seg_len <= len(y):
        yield y[start:start + seg_len], start / sr, (start + seg_len) / sr
        start += hop_len

    # Capture a meaningful trailing remainder (more than half a hop) so
    # content near the end of the song isn't silently dropped.
    if start < len(y) and (len(y) - start) > (hop_len / 2):
        yield y[start:], start / sr, len(y) / sr


def get_recording_id(row):
    for col in ["Recoding ID", "recording_id", "Recording ID"]:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    raise KeyError("No recording ID column found in metadata (checked common variants).")


def main():
    meta = pd.read_csv(METADATA_PATH)
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    segment_rows = []
    skipped = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Segmenting"):
        recording_id = get_recording_id(row)
        genre = row["genre"]

        audio_path = find_audio_path(recording_id, genre)
        if audio_path is None:
            skipped.append((recording_id, "file not found"))
            continue

        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        y, trimmed = maybe_trim_edges(y, sr)

        genre_out_dir = os.path.join(OUTPUT_AUDIO_DIR, resolve_genre_folder(genre))
        os.makedirs(genre_out_dir, exist_ok=True)

        base_id = os.path.splitext(recording_id)[0]

        for i, (seg_audio, start_sec, end_sec) in enumerate(segment_audio(y, sr)):
            duration = end_sec - start_sec
            rms = float(np.sqrt(np.mean(seg_audio ** 2))) if len(seg_audio) > 0 else 0.0

            segment_id = f"{base_id}_seg{i:02d}"
            out_path = os.path.join(genre_out_dir, f"{segment_id}.wav")
            sf.write(out_path, seg_audio, sr)

            new_row = row.to_dict()
            new_row.update({
                "segment_id": segment_id,
                "recording_id": recording_id,
                "segment_index": i,
                "segment_start_sec": round(start_sec, 2),
                "segment_end_sec": round(end_sec, 2),
                "segment_duration_sec": round(duration, 2),
                "edge_trimmed": trimmed,
                "rms_energy": round(rms, 5),
                "likely_silent": rms < SILENCE_RMS_THRESHOLD,
                "segment_filepath": out_path,
            })
            segment_rows.append(new_row)

    segment_df = pd.DataFrame(segment_rows)
    segment_df.to_csv(OUTPUT_METADATA, index=False)

    print(f"\n{'='*50}")
    print("Done.")
    print(f"  Recordings processed : {len(meta) - len(skipped)}/{len(meta)}")
    print(f"  Segments generated   : {len(segment_df)}")
    print(f"  Likely-silent segments flagged (not removed): {int(segment_df['likely_silent'].sum())}")
    print(f"  Segment metadata saved to: {OUTPUT_METADATA}")
    if skipped:
        print(f"\n  Skipped {len(skipped)} recording(s):")
        for rid, reason in skipped:
            print(f"    {rid}: {reason}")


if __name__ == "__main__":
    main()
