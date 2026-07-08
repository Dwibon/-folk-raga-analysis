# src/verify_dataset.py

import os
import pandas as pd

# ── CONFIG — adjust these two lines to match your exact folder names ──
AUDIO_DIRS = {
    "Kamrupi"   : "dataset/Kamrupi",
    "Goalpariya": "dataset/Goalpariya",   # change if your folder is Goapariya
}
METADATA_PATH = "dataset/metadata.csv"
# ─────────────────────────────────────────────────────────────────────

df = pd.read_csv(METADATA_PATH)

found    = []
missing  = []

for _, row in df.iterrows():
    recording_id = row["Recoding ID"]          # e.g. AFM0001.wav
    genre        = row["genre"]                # Kamrupi or Goalpariya

    folder = AUDIO_DIRS.get(genre)
    if folder is None:
        missing.append((recording_id, f"Unknown genre: {genre}"))
        continue

    filepath = os.path.join(folder, recording_id)

    if os.path.exists(filepath):
        found.append(filepath)
    else:
        # Try lowercase filename in case
        alt = os.path.join(folder, recording_id.lower())
        if os.path.exists(alt):
            found.append(alt)
        else:
            missing.append((recording_id, filepath))

print(f"✓ Found  : {len(found)}/{len(df)}")
print(f"✗ Missing: {len(missing)}/{len(df)}")

if missing:
    print("\nMissing files:")
    for rid, path in missing:
        print(f"  {rid}  →  {path}")
else:
    print("\nAll 110 files accounted for. Ready for feature extraction.")