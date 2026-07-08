import librosa
import numpy as np
import os

TARGET_SR = 22050      # standard for librosa
SKIP_SEC  = 20         # skip first 20s (instrumental intro)
MIN_DURATION = 30      # minimum usable duration after skipping (seconds)


def load_audio(filepath, skip_intro=True, trim_silence=True):
    """
    Load a WAV file, resample to 22050 Hz mono,
    skip intro, and trim trailing silence.

    Returns:
        y  : np.ndarray — audio time series
        sr : int        — sample rate (always TARGET_SR)
    
    Raises:
        ValueError if file is too short after preprocessing.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    # Load as mono, resample to TARGET_SR
    y, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # Skip first SKIP_SEC (instrumental intro)
    if skip_intro:
        skip_samples = int(SKIP_SEC * sr)
        if len(y) > skip_samples + int(MIN_DURATION * sr):
            y = y[skip_samples:]
        else:
            # song too short to skip — skip less aggressively
            skip_samples = int(5 * sr)
            y = y[skip_samples:]

    # Trim leading/trailing silence
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=30)

    # Sanity check
    remaining = librosa.get_duration(y=y, sr=sr)
    if remaining < MIN_DURATION:
        raise ValueError(
            f"{filepath}: only {remaining:.1f}s usable after preprocessing "
            f"(original: {duration:.1f}s). Check the file."
        )

    return y, sr


def get_duration(y, sr=TARGET_SR):
    return librosa.get_duration(y=y, sr=sr)


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Usage: python preprocess.py path/to/AFM0001.wav
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <path_to_audio_file>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading: {path}")

    y, sr = load_audio(path)

    print(f"  Sample rate : {sr} Hz")
    print(f"  Duration    : {get_duration(y, sr):.2f} seconds")
    print(f"  Samples     : {len(y)}")
    print(f"  Min amplitude: {y.min():.4f}")
    print(f"  Max amplitude: {y.max():.4f}")
    print("Preprocessing OK.")