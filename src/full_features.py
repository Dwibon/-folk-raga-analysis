# src/features.py

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from preprocess import load_audio

# ── CONFIG ────────────────────────────────────────────────
AUDIO_DIRS = {
    "Kamrupi"   : "dataset/Kamrupi",
    "Goalpariya": "dataset/Goalpariya",
}
METADATA_PATH = "dataset/metadata.csv"
OUTPUT_PATH   = "outputs/features/full_features.csv"
N_MFCC        = 20
# ─────────────────────────────────────────────────────────


def extract_mfccs(y, sr, n_mfcc=N_MFCC):
    """
    MFCC mean + std for each coefficient.
    Captures timbral/vocal texture.
    Returns vector of length n_mfcc * 2 = 40.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mean = np.mean(mfcc, axis=1)
    std  = np.std(mfcc,  axis=1)
    return np.concatenate([mean, std])


def extract_chroma(y, sr):
    """
    12-dimensional pitch class distribution, mean + std.
    Captures melodic character — less singer-dependent than MFCCs.
    Returns vector of length 24.
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    mean   = np.mean(chroma, axis=1)
    std    = np.std(chroma,  axis=1)
    return np.concatenate([mean, std])


def extract_tempo_and_spectral(y, sr):
    """
    Tempo + 3 spectral features (centroid, rolloff, ZCR), mean + std.
    Captures rhythmic and brightness character.
    Returns vector of length 7.
    """
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo     = float(np.atleast_1d(tempo)[0])

    # Spectral centroid (brightness)
    centroid      = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = float(np.mean(centroid))
    centroid_std  = float(np.std(centroid))

    # Spectral rolloff
    rolloff      = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std  = float(np.std(rolloff))

    # Zero crossing rate
    zcr      = librosa.feature.zero_crossing_rate(y)
    zcr_mean = float(np.mean(zcr))
    zcr_std  = float(np.std(zcr))

    return np.array([
        tempo,
        centroid_mean, centroid_std,
        rolloff_mean,  rolloff_std,
        zcr_mean,      zcr_std,
    ])


def build_feature_names(n_mfcc=N_MFCC):
    mfcc_cols = (
        [f"mfcc{i+1}_mean" for i in range(n_mfcc)] +
        [f"mfcc{i+1}_std"  for i in range(n_mfcc)]
    )
    chroma_cols = (
        [f"chroma{i+1}_mean" for i in range(12)] +
        [f"chroma{i+1}_std"  for i in range(12)]
    )
    spectral_cols = [
        "tempo",
        "centroid_mean", "centroid_std",
        "rolloff_mean",  "rolloff_std",
        "zcr_mean",      "zcr_std",
    ]
    return mfcc_cols + chroma_cols + spectral_cols


def extract_all(metadata_path, audio_dirs, output_path, n_mfcc=N_MFCC):
    df = pd.read_csv(metadata_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    feature_names = build_feature_names(n_mfcc)
    rows   = []
    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        recording_id = row["Recoding ID"]
        genre        = row["genre"]
        singer       = row["singer"]

        filepath = os.path.join(audio_dirs[genre], recording_id)

        try:
            y, sr = load_audio(filepath)

            mfcc_feat     = extract_mfccs(y, sr, n_mfcc)          # 40
            chroma_feat   = extract_chroma(y, sr)                  # 24
            spectral_feat = extract_tempo_and_spectral(y, sr)      #  7
            features      = np.concatenate([
                mfcc_feat, chroma_feat, spectral_feat
            ])                                                      # 71 total

            row_dict = {
                "recording_id" : recording_id,
                "genre"        : genre,
                "singer"       : singer,
            }
            for name, val in zip(feature_names, features):
                row_dict[name] = val

            rows.append(row_dict)

        except Exception as e:
            errors.append((recording_id, str(e)))
            print(f"\n  ERROR — {recording_id}: {e}")

    features_df = pd.DataFrame(rows)
    features_df.to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"Done.")
    print(f"  Extracted : {len(rows)}/{len(df)} songs")
    print(f"  Errors    : {len(errors)}")
    print(f"  Saved to  : {output_path}")
    print(f"  Shape     : {features_df.shape}")
    return features_df


# ── Quick test on one file ────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        test_path = "dataset/Kamrupi/AFM0001.wav"
        print(f"Testing on: {test_path}")
        y, sr = load_audio(test_path)

        mfcc_feat     = extract_mfccs(y, sr)
        chroma_feat   = extract_chroma(y, sr)
        spectral_feat = extract_tempo_and_spectral(y, sr)
        features      = np.concatenate([mfcc_feat, chroma_feat, spectral_feat])

        print(f"  MFCC vector length     : {len(mfcc_feat)}")
        print(f"  Chroma vector length   : {len(chroma_feat)}")
        print(f"  Spectral vector length : {len(spectral_feat)}")
        print(f"  Total feature length   : {len(features)}")
        print(f"  Tempo                  : {spectral_feat[0]:.2f} BPM")
        print(f"  Chroma means (first 4) : {chroma_feat[:4].round(3)}")
        print("All features OK.")
    else:
        extract_all(METADATA_PATH, AUDIO_DIRS, OUTPUT_PATH)