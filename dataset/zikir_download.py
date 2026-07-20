"""
download_and_segment_zikir.py

Downloads and segments the Zikir catalog (zikir_metadata.csv) into the same
dataset/ folder already used for Goalpariya and Kamrupi (-> dataset/Zikir/...),
reusing raw_audio/ as a shared download cache.

Differs from download_and_segment.py in three ways, to match how this
catalog was compiled:

  1. JUKEBOX ROWS SHARE A URL. Several rows point at the same "jukebox"
     video (one video, many songs) - one row per song, same link repeated.
     Each unique video is downloaded ONCE and cached in raw_audio/<id>.wav;
     every row referencing it reuses that cached file instead of
     re-downloading.

  2. SEGMENTS ARE NAMED FROM THE CATALOG, NOT GENERIC INDICES. For a
     jukebox video, the audio is silence-split, then the resulting chunks
     are matched 1:1, IN ORDER, to the rows sharing that URL - this assumes
     the rows were entered in the same order the songs appear in the video.
     If the chunk count doesn't match the row count, nothing is guessed:
     the chunks are saved with generic names and the video is logged to
     needs_manual_review.csv so you can fix it by hand (see "manual trim
     override" below) rather than risking a wrong title on a clip.

  3. NON-VIDEO LINKS ARE SKIPPED WITH A NOTE. Some rows link to JioSaavn,
     Grokipedia, a YouTube "hashtag" page, a channel page, a YT Music
     search query, or a YT Music playlist - none of these are a single
     downloadable video. Any URL that doesn't reduce to a normal
     youtube.com/watch, youtu.be/, or /shorts/ link is skipped and logged
     to skipped_links.csv with a reason.

Bonus: zikir_metadata.csv turned out to actually be an RTF file saved with
a .csv extension (common when exporting from TextEdit/Pages in "Rich Text"
mode). This script detects and converts that automatically, but it's worth
re-exporting as plain-text CSV/UTF-8 next time to avoid relying on that.

MANUAL TRIM OVERRIDE (optional):
  Add trim_start / trim_end columns (seconds or mm:ss) to any row and that
  row is cut directly from the cached raw download instead of going through
  silence-splitting - useful for fixing anything flagged in
  needs_manual_review.csv, or for cutting out a spoken intro.

RUN THIS LOCALLY (not in a sandboxed environment without YouTube access).
Requires: pip install yt-dlp pydub    +    ffmpeg installed system-wide

Usage (run from your IKS-Internship/ project root, so dataset/ and
raw_audio/ line up with what download_and_segment.py already created):
    python download_and_segment_zikir.py [path/to/zikir_metadata.csv]

Defaults to dataset/zikir_metadata.csv if no path is given.
"""

import csv
import io
import re
import subprocess
import sys
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence

CATALOG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("zikir_metadata.csv")
RAW_DIR = Path("raw_audio")          # shared cache across all genres
OUT_DIR = Path("dataset")            # segments land in dataset/Zikir/...
MANIFEST = "zikir_manifest.csv"      # kept separate from dataset_manifest.csv on
                                      # purpose - that file is opened in "w" mode by
                                      # the original script, so reusing its name here
                                      # would wipe out the Goalpariya/Kamrupi entries
SKIPPED_LOG = "skipped_links.csv"
REVIEW_LOG = "needs_manual_review.csv"
FAILED_LOG = "failed_downloads.csv"

MIN_SEGMENT_MS = 8000        # discard segments shorter than this (too short for melodic analysis)
MIN_SILENCE_LEN_MS = 1500    # how long a silence must be to count as a song/phrase boundary
SILENCE_OFFSET_DB = 16       # segment is silent if quieter than (avg_dBFS - this)


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[:40]


def extract_video_id(url: str):
    """Returns the 11-char video ID for a normal watch/short/youtu.be link,
    or None for anything that isn't a single downloadable video - playlists,
    channel pages, hashtag pages, search queries, and non-YouTube domains
    all correctly fall through to None."""
    m = re.search(
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        url,
    )
    return m.group(1) if m else None


def parse_timestamp(ts: str):
    """Accepts '15', '1:05', or '' -> returns seconds (float) or None."""
    ts = (ts or "").strip()
    if not ts:
        return None
    if ":" in ts:
        mins, secs = ts.split(":")
        return int(mins) * 60 + float(secs)
    return float(ts)


def load_catalog_rows(path: Path):
    """Reads the catalog as CSV, transparently handling the case where the
    file is actually an RTF export rather than plain-text CSV."""
    raw = path.read_bytes().decode("utf-8", errors="replace")

    if raw.lstrip().startswith("{\\rtf"):
        print("NOTE: catalog looks like an RTF file saved with a .csv extension, "
              "not plain-text CSV - converting on the fly. (Next time, export as "
              "'Plain Text' to avoid needing this.)")
        raw = _rtf_to_csv_text(raw)

    reader = csv.DictReader(io.StringIO(raw))
    reader.fieldnames = [(f or "").strip() for f in reader.fieldnames]
    return list(reader)


def _rtf_to_csv_text(raw: str) -> str:
    m = re.search(r"\\partightenfactor0\s*\n", raw)
    body = raw[m.end():] if m else raw
    body = re.sub(r"\\f0\\fs\d+\s*\\cf0\s*", "", body, count=1)
    body = re.sub(
        r"\\'([0-9a-fA-F]{2})",
        lambda mo: bytes([int(mo.group(1), 16)]).decode("cp1252", "replace"),
        body,
    )
    body = re.sub(r"\\\s*\n", "\n", body)   # per-row line-continuation marker -> real newline
    body = re.sub(r"\}+\s*$", "", body)     # trailing RTF close-brace(s)
    return body


def download_audio(url: str, out_stub: Path) -> Path:
    """Downloads audio as WAV; returns the resulting file path."""
    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "-o", f"{out_stub}.%(ext)s", url]
    subprocess.run(cmd, check=True)
    return out_stub.with_suffix(".wav")


def save_clip(clip, genre, stem, subgenre, singer, gender, song_title,
              manifest_writer, used_names) -> Path:
    """Writes one clip to dataset/<genre>/, de-duplicating filenames, and
    logs it to the manifest."""
    genre_dir = OUT_DIR / genre
    genre_dir.mkdir(parents=True, exist_ok=True)

    base = safe_name(stem)
    name, n = base, 2
    while name in used_names:
        name = f"{base}_{n}"
        n += 1
    used_names.add(name)

    if len(clip) < MIN_SEGMENT_MS:
        print(f"  WARNING: '{song_title}' clip is very short ({len(clip)/1000:.1f}s)")

    out_path = genre_dir / f"{name}.wav"
    clip.export(out_path, format="wav")
    manifest_writer.writerow([
        str(out_path), genre, subgenre, singer, gender, song_title,
        round(len(clip) / 1000, 1),
    ])
    return out_path


def main():
    RAW_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    rows = load_catalog_rows(CATALOG)

    skipped, failed, review_needed = [], [], []
    used_names = set()

    # Group rows by extractable video ID, preserving catalog order within
    # each group and across groups.
    groups = {}
    order = []
    for row in rows:
        url = (row.get("YouTube URL") or "").strip()
        title = (row.get("Song Title") or "").strip()
        if not url:
            continue
        vid = extract_video_id(url)
        if vid is None:
            skipped.append([
                title, row.get("singer", ""), url,
                "Not a single downloadable YouTube video "
                "(playlist / channel / hashtag / search page / non-YouTube link)",
            ])
            continue
        if vid not in groups:
            groups[vid] = []
            order.append(vid)
        groups[vid].append(row)

    with open(MANIFEST, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["filepath", "genre", "subgenre", "singer", "gender",
                          "song_title", "duration_sec"])

        for vid in order:
            group_rows = groups[vid]
            url = (group_rows[0].get("YouTube URL") or "").strip()
            genre = (group_rows[0].get("genre") or "").strip()
            raw_path = RAW_DIR / f"{vid}.wav"

            if not raw_path.exists():
                print(f"Downloading ({len(group_rows)} song(s) from this video): {url}")
                try:
                    download_audio(url, RAW_DIR / vid)
                except subprocess.CalledProcessError as e:
                    print(f"  FAILED to download: {url}")
                    for row in group_rows:
                        failed.append([row.get("Song Title", ""), url, str(e)])
                    continue
            else:
                print(f"Already downloaded, reusing cache ({len(group_rows)} song(s)): {url}")

            audio = AudioSegment.from_wav(raw_path)

            # Rows with an explicit manual trim are cut directly and pulled
            # out of the silence-split pool for this video.
            auto_rows = []
            for row in group_rows:
                t_start = parse_timestamp(row.get("trim_start", ""))
                t_end = parse_timestamp(row.get("trim_end", ""))
                if t_start is not None or t_end is not None:
                    start_ms = int((t_start or 0) * 1000)
                    end_ms = int(t_end * 1000) if t_end is not None else len(audio)
                    clip = audio[start_ms:end_ms]
                    stem = f"{row.get('singer','')}_{row.get('Song Title','')}_{vid}"
                    save_clip(clip, genre, stem, row.get("subgenre", ""),
                              row.get("singer", ""), row.get("Gender", ""),
                              row.get("Song Title", ""), writer, used_names)
                else:
                    auto_rows.append(row)

            if not auto_rows:
                continue

            if len(auto_rows) == 1:
                # Single song for this video - whole clip, no splitting needed.
                row = auto_rows[0]
                stem = f"{row.get('singer','')}_{row.get('Song Title','')}_{vid}"
                save_clip(audio, genre, stem, row.get("subgenre", ""),
                          row.get("singer", ""), row.get("Gender", ""),
                          row.get("Song Title", ""), writer, used_names)
                continue

            # Jukebox: silence-split, then try to line chunks up with rows in order.
            chunks = split_on_silence(
                audio,
                min_silence_len=MIN_SILENCE_LEN_MS,
                silence_thresh=audio.dBFS - SILENCE_OFFSET_DB,
                keep_silence=300,
            )
            chunks = [c for c in chunks if len(c) >= MIN_SEGMENT_MS]

            if len(chunks) == len(auto_rows):
                for row, clip in zip(auto_rows, chunks):
                    stem = f"{row.get('singer','')}_{row.get('Song Title','')}_{vid}"
                    save_clip(clip, genre, stem, row.get("subgenre", ""),
                              row.get("singer", ""), row.get("Gender", ""),
                              row.get("Song Title", ""), writer, used_names)
            else:
                print(f"  MISMATCH: {url} split into {len(chunks)} segment(s) but "
                      f"catalog lists {len(auto_rows)} song(s) for it - saving with "
                      f"generic names, flagged for manual review.")
                for i, clip in enumerate(chunks):
                    stem = f"{genre}_{vid}_{i:02d}"
                    save_clip(clip, genre, stem, "", "", "",
                              "UNKNOWN - needs manual review", writer, used_names)
                review_needed.append([
                    url, len(auto_rows), len(chunks),
                    "; ".join(r.get("Song Title", "") for r in auto_rows),
                ])

    if skipped:
        with open(SKIPPED_LOG, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["song_title", "singer", "url", "reason"])
            w.writerows(skipped)
        print(f"\n{len(skipped)} row(s) skipped (non single-video links) - see {SKIPPED_LOG}")

    if failed:
        with open(FAILED_LOG, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["song_title", "url", "error"])
            w.writerows(failed)
        print(f"{len(failed)} download(s) failed - see {FAILED_LOG}")
        print("Fix (e.g. check the URL still works, or try again later), then just re-run this script.")

    if review_needed:
        with open(REVIEW_LOG, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["url", "expected_song_count", "actual_segment_count", "expected_song_titles"])
            w.writerows(review_needed)
        print(f"{len(review_needed)} video(s) need manual review (segment count mismatch) - see {REVIEW_LOG}")
        print("Fix by adding trim_start/trim_end for those rows in the catalog, then re-run.")

    print(f"\nDone. Segments in ./{OUT_DIR}/<genre>/, manifest at ./{MANIFEST}")


if __name__ == "__main__":
    main()