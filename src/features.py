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
OUTPUT_PATH   = "outputs/features/mfcc_features.csv"
N_MFCC        = 20
# ─────────────────────────────────────────────────────────


def extract_mfccs(y, sr, n_mfcc=N_MFCC):
    """
    Extract MFCC mean and std for each coefficient.
    Returns a flat vector of length n_mfcc * 2.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # mfcc shape: (n_mfcc, time_frames)
    mean = np.mean(mfcc, axis=1)   # shape: (n_mfcc,)
    std  = np.std(mfcc,  axis=1)   # shape: (n_mfcc,)
    return np.concatenate([mean, std])


def build_feature_names(n_mfcc=N_MFCC):
    mean_cols = [f"mfcc{i+1}_mean" for i in range(n_mfcc)]
    std_cols  = [f"mfcc{i+1}_std"  for i in range(n_mfcc)]
    return mean_cols + std_cols


def extract_all(metadata_path, audio_dirs, output_path, n_mfcc=N_MFCC):
    df = pd.read_csv(metadata_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    feature_names = build_feature_names(n_mfcc)
    rows = []
    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCCs"):
        recording_id = row["Recoding ID"]
        genre        = row["genre"]
        singer       = row["singer"]

        folder   = audio_dirs[genre]
        filepath = os.path.join(folder, recording_id)

        try:
            y, sr    = load_audio(filepath)
            features = extract_mfccs(y, sr, n_mfcc)

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
    print(f"  Extracted : {len(rows)}/{ len(df)} songs")
    print(f"  Errors    : {len(errors)}")
    print(f"  Saved to  : {output_path}")
    print(f"  Shape     : {features_df.shape}")
    return features_df


# ── Quick test on one file first ─────────────────────────
if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # Test on a single file before running full extraction
        test_path = "dataset/Kamrupi/AFM0001.wav"
        print(f"Testing on: {test_path}")
        y, sr = load_audio(test_path)
        features = extract_mfccs(y, sr)
        print(f"  Feature vector length : {len(features)}")
        print(f"  MFCC means (first 5)  : {features[:5].round(3)}")
        print(f"  MFCC stds  (first 5)  : {features[20:25].round(3)}")
        print("MFCC extraction OK.")
    else:
        extract_all(METADATA_PATH, AUDIO_DIRS, OUTPUT_PATH)