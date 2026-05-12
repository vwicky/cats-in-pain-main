import os
import json
import time
import datetime

# --- ENVIRONMENT FIXES ---
os.environ["DLCLIGHT"] = "True"
import matplotlib
matplotlib.use('Agg')

import subprocess
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FOLDERS = [
    'data/dataset/snippets_v2',
    'data/dataset/tiktok_snippets',
]
OUTPUT_FOLDER = 'data/dataset/labeled_deeplabcut_videos'
LOG_FILE      = 'data/dataset/downsample_log.jsonl'

TARGET_FPS       = 5
SUPERANIMAL_MODEL = 'superanimal_quadruped'   # NOTE: unused in this script
BATCH_SIZE        = 32                         # NOTE: unused in this script

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def log_entry(path: str, **kwargs) -> None:
    """Append one JSON record to the JSONL log file."""
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "video": path,
        **kwargs,
    }
    with open(path=LOG_FILE, mode='a', encoding='utf-8') as fh:
        fh.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
# Collect videos
# --------------------------------------------------------------------------- #

all_video_tasks: list[str] = []

for folder in INPUT_FOLDERS:
    if os.path.exists(folder):
        files = [
            os.path.join(os.getcwd(), folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        ]
        all_video_tasks.extend(files)
    else:
        print(f"[WARNING] Input folder not found, skipping: {folder}")
        log_entry(
            path=folder,
            status="skipped",
            reason="input_folder_not_found",
        )

print(f"Total videos found: {len(all_video_tasks)}")

# --------------------------------------------------------------------------- #
# Process each video
# --------------------------------------------------------------------------- #

output_abs_path = os.path.abspath(OUTPUT_FOLDER)

pbar = tqdm(all_video_tasks, desc="Overall Progress", unit="video")

for video_path in pbar:
    filename = os.path.basename(video_path)
    pbar.set_description(f"Processing {filename[:30]}...")

    downsampled_video_path = os.path.join(output_abs_path, filename)

    # ---- Step A: Downsample with ffmpeg ----------------------------------- #

    if os.path.exists(downsampled_video_path):
        tqdm.write(f"[SKIP] Already exists: {filename}")
        log_entry(
            path=video_path,
            output=downsampled_video_path,
            status="skipped",
            reason="output_already_exists",
            elapsed_seconds=0.0,
        )
        continue

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-filter:v', f'fps=fps={TARGET_FPS}',
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'veryfast',
        downsampled_video_path,
    ]

    t_start = time.perf_counter()
    try:
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # capture stderr so we can log it on failure
        )
        elapsed = round(time.perf_counter() - t_start, 3)

        if result.returncode != 0:
            stderr_tail = result.stderr.decode(errors='replace').strip().splitlines()
            error_msg   = "\n".join(stderr_tail[-10:])   # last 10 lines are most informative
            tqdm.write(f"[ERROR] ffmpeg failed for {filename} (rc={result.returncode})")
            log_entry(
                path=video_path,
                output=downsampled_video_path,
                status="error",
                reason="ffmpeg_nonzero_exit",
                returncode=result.returncode,
                ffmpeg_stderr=error_msg,
                elapsed_seconds=elapsed,
            )
        else:
            tqdm.write(f"[OK] {filename}  ({elapsed}s)")
            log_entry(
                path=video_path,
                output=downsampled_video_path,
                status="ok",
                elapsed_seconds=elapsed,
                output_size_bytes=os.path.getsize(downsampled_video_path)
                    if os.path.exists(downsampled_video_path) else None,
            )

    except FileNotFoundError:
        elapsed = round(time.perf_counter() - t_start, 3)
        tqdm.write("[FATAL] ffmpeg not found — is it installed and on PATH?")
        log_entry(
            path=video_path,
            status="error",
            reason="ffmpeg_not_found",
            elapsed_seconds=elapsed,
        )
        break   # no point continuing if ffmpeg is missing

    except Exception as exc:
        elapsed = round(time.perf_counter() - t_start, 3)
        tqdm.write(f"[ERROR] Unexpected error for {filename}: {exc}")
        log_entry(
            path=video_path,
            status="error",
            reason="unexpected_exception",
            exception=str(exc),
            elapsed_seconds=elapsed,
        )

print(f"\n--- BATCH COMPLETE ---")
print(f"Log written to: {LOG_FILE}")