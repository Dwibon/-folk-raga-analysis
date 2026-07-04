"""
download_and_segment.py

Reads audio_source_catalog.csv and, for each row:
  1. Downloads the video's audio via yt-dlp (WAV)
  2. Two modes, chosen automatically per row:
       - If the row has trim_start/trim_end filled in: cuts exactly that
         range and saves it as ONE clip. Use this for single/duet songs
         where you already know the dead-zone to remove (e.g. a spoken
         intro) - no guessing via silence detection needed.
       - If trim_start/trim_end are blank: runs auto silence-splitting.
         Use this for true multi-song jukebox/compilation uploads where
         finding each song boundary by ear isn't worth the time.
  3. Saves segments into dataset/<genre>/ with consistent naming
  4. Writes dataset_manifest.csv logging every final clip

Catalog CSV should have two extra columns for this to work:
  trim_start, trim_end   (format: seconds, or mm:ss, e.g. "15" or "1:05")
Leave both blank for compilation rows.

RUN THIS LOCALLY (not in a sandboxed environment without YouTube access).
Requires: pip install yt-dlp pydub    +    ffmpeg installed system-wide

Usage:
    python download_and_segment.py
"""

import csv
import subprocess
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence

CATALOG = "audio_source_catalog_trimming.csv"
RAW_DIR = Path("raw_audio")
OUT_DIR = Path("dataset")
MANIFEST = "dataset_manifest.csv"

MIN_SEGMENT_MS = 8000        # discard segments shorter than this (too short for melodic analysis)
MIN_SILENCE_LEN_MS = 1500    # how long a silence must be to count as a song/phrase boundary
SILENCE_OFFSET_DB = 16       # segment is silent if quieter than (avg_dBFS - this)


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[:40]


def parse_timestamp(ts: str):
    """Accepts '15', '1:05', or '' -> returns seconds (float) or None."""
    ts = (ts or "").strip()
    if not ts:
        return None
    if ":" in ts:
        mins, secs = ts.split(":")
        return int(mins) * 60 + float(secs)
    return float(ts)


def download_audio(url: str, out_stub: Path) -> Path:
    """Downloads audio as WAV; returns the resulting file path."""
    cmd = [
        "yt-dlp", "-x", "--audio-format", "wav",
        "-o", f"{out_stub}.%(ext)s", url,
    ]
    subprocess.run(cmd, check=True)
    return out_stub.with_suffix(".wav")


def manual_trim_and_save(raw_path: Path, genre: str, subgenre: str,
                          singer: str, base_name: str,
                          trim_start, trim_end, manifest_writer) -> None:
    """For rows where the exact good range is already known - cuts once,
    no silence-detection guessing."""
    audio = AudioSegment.from_wav(raw_path)
    start_ms = int((trim_start or 0) * 1000)
    end_ms = int(trim_end * 1000) if trim_end is not None else len(audio)
    clip = audio[start_ms:end_ms]

    if len(clip) < MIN_SEGMENT_MS:
        print(f"  WARNING: trimmed clip for {base_name} is very short "
              f"({len(clip)/1000:.1f}s) - check your trim_start/trim_end values")

    genre_dir = OUT_DIR / genre
    genre_dir.mkdir(parents=True, exist_ok=True)
    out_path = genre_dir / f"{base_name}.wav"
    clip.export(out_path, format="wav")
    manifest_writer.writerow([
        str(out_path), genre, subgenre, singer,
        round(len(clip) / 1000, 1), base_name,
    ])


def segment_and_save(raw_path: Path, genre: str, subgenre: str,
                      singer: str, base_name: str, manifest_writer) -> None:
    audio = AudioSegment.from_wav(raw_path)
    chunks = split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=audio.dBFS - SILENCE_OFFSET_DB,
        keep_silence=300,
    )
    genre_dir = OUT_DIR / genre
    genre_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks):
        if len(chunk) < MIN_SEGMENT_MS:
            continue
        fname = f"{base_name}_{i:02d}.wav"
        out_path = genre_dir / fname
        chunk.export(out_path, format="wav")
        manifest_writer.writerow([
            str(out_path), genre, subgenre, singer,
            round(len(chunk) / 1000, 1), base_name,
        ])


def main():
    RAW_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    with open(CATALOG, newline="", encoding="utf-8") as f, \
         open(MANIFEST, "w", newline="", encoding="utf-8") as mf:

        reader = csv.DictReader(f)
        writer = csv.writer(mf)
        writer.writerow(["filepath", "genre", "subgenre", "singer",
                          "duration_sec", "source_video_title"])

        for row in reader:
            url = (row.get("video_url") or "").strip()
            if not url or url.startswith("#"):
                continue

            genre = row["genre"].strip()
            subgenre = (row.get("subgenre") or "").strip()
            singer = (row.get("singer") or "").strip()
            base_name = safe_name(row["title"])
            raw_stub = RAW_DIR / base_name
            raw_path = raw_stub.with_suffix(".wav")

            if not raw_path.exists():
                print(f"Downloading: {row['title']}")
                try:
                    download_audio(url, raw_stub)
                except subprocess.CalledProcessError:
                    print(f"  FAILED to download: {url}")
                    continue

            trim_start = parse_timestamp(row.get("trim_start", ""))
            trim_end = parse_timestamp(row.get("trim_end", ""))

            if trim_start is not None or trim_end is not None:
                print(f"Manual trim: {base_name} [{trim_start or 0}s - {trim_end or 'end'}]")
                manual_trim_and_save(raw_path, genre, subgenre, singer,
                                      base_name, trim_start, trim_end, writer)
            else:
                print(f"Auto-splitting (compilation): {base_name}")
                segment_and_save(raw_path, genre, subgenre, singer, base_name, writer)

    print(f"\nDone. Segments in ./{OUT_DIR}/, manifest at ./{MANIFEST}")


if __name__ == "__main__":
    main()
