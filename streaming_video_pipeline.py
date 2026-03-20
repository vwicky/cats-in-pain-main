import argparse
import glob
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import joblib
import librosa
import numpy as np
import torch
from pydub import AudioSegment
from scenedetect import ContentDetector, detect
from tqdm import tqdm
from ultralytics import YOLO
from yt_dlp import YoutubeDL

from data.class_names import pre_classifier_class_names


# -----------------------------
# Config defaults
# -----------------------------

MODEL_PATH = "yolo11m.pt"
PRECLASSIFIER_PATH = "models/audio_preclassifier/voting_classifier_with-add-data.pkl"

GEMINI_LABELS_FILE = "gemini_labeled_videos.jsonl"
METADATA_FILE = "metadata.jsonl"
CRAWLED_CANDIDATES_FILE = "logs/crawled_cat_candidates.jsonl"
PIPELINE_LOG_FILE = "logs/streaming_pipeline_log.jsonl"

SNIPPETS_OUTPUT_DIR = "downloaded_snippets"
WORK_DIR = "downloads/streaming_pipeline_workdir"

CAT_CLASS_ID = 15
SAMPLE_RATE = 22050
N_MFCC = 20

MIN_SEGMENT_DURATION = 3.0
MIN_CHUNK_DURATION = 3.0
MAX_CHUNK_DURATION = 7.0
FRAME_STRIDE = 3
YOLO_IMGSZ = 640
YOLO_CONF = 0.4
# Match original downloader behavior for fetch filtering.
MAX_DOWNLOAD_DURATION_SECONDS = 1800
# Keep original processing constraint from your processor.
MAX_PROCESS_DURATION_SECONDS = 300
MAX_VIDEO_FILESIZE_BYTES = 500 * 1024 * 1024


# -----------------------------
# Globals loaded once
# -----------------------------

yolo_model: YOLO | None = None
preclass_model = None
cat_audio_class_id = None
thread_local = threading.local()


def normalize_title(value: str) -> str:
    stem = Path(value).stem
    if stem.endswith("_video"):
        stem = stem[: -len("_video")]
    return stem.strip().lower()


def read_jsonl(path: str) -> Iterable[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_processed_index(metadata_file: str) -> tuple[set[str], set[str]]:
    processed_ids: set[str] = set()
    processed_titles: set[str] = set()

    for row in read_jsonl(metadata_file):
        video_id = row.get("video_id")
        original_video = row.get("original_video")
        if isinstance(video_id, str) and video_id:
            processed_ids.add(video_id)
        if isinstance(original_video, str) and original_video:
            processed_titles.add(normalize_title(original_video))

    return processed_ids, processed_titles


def load_video_title_index(candidates_file: str) -> dict[str, str]:
    id_to_title: dict[str, str] = {}
    for row in read_jsonl(candidates_file):
        video_id = row.get("video_id")
        video_title = row.get("video_title")
        if isinstance(video_id, str) and isinstance(video_title, str) and video_id and video_title:
            id_to_title.setdefault(video_id, video_title)
    return id_to_title


def load_kept_video_ids(gemini_labels_file: str) -> list[str]:
    ordered_unique_ids: list[str] = []
    seen: set[str] = set()

    for row in read_jsonl(gemini_labels_file):
        keep_flag = row.get("keep", row.get("kept", False))
        video_id = row.get("video_id")
        if keep_flag is True and isinstance(video_id, str) and video_id and video_id not in seen:
            seen.add(video_id)
            ordered_unique_ids.append(video_id)

    return ordered_unique_ids


def load_logged_video_ids(pipeline_log_file: str) -> set[str]:
    """
    IDs already attempted in this pipeline (success/skipped/error) should not be queued again.
    """
    seen_ids: set[str] = set()
    for row in read_jsonl(pipeline_log_file):
        video_id = row.get("video_id")
        status = row.get("status")
        if isinstance(video_id, str) and video_id and status in {"success", "skipped", "error"}:
            seen_ids.add(video_id)
    return seen_ids


def build_todo_video_ids(
    gemini_labels_file: str,
    metadata_file: str,
    crawled_candidates_file: str,
    pipeline_log_file: str,
) -> list[str]:
    processed_ids, processed_titles = load_processed_index(metadata_file)
    logged_ids = load_logged_video_ids(pipeline_log_file)
    id_to_title = load_video_title_index(crawled_candidates_file)
    kept_ids = load_kept_video_ids(gemini_labels_file)

    todo: list[str] = []
    # Reverse order: process most recently appended kept IDs first.
    for video_id in reversed(kept_ids):
        if video_id in processed_ids:
            continue
        if video_id in logged_ids:
            continue
        title = id_to_title.get(video_id)
        if title and normalize_title(title) in processed_titles:
            continue
        todo.append(video_id)
    return todo


def init_pipeline_models(model_path: str, preclassifier_path: str) -> None:
    global yolo_model, preclass_model, cat_audio_class_id
    if yolo_model is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        yolo_model = YOLO(model_path).to(device)
    if preclass_model is None:
        preclass_model = joblib.load(preclassifier_path)
    if cat_audio_class_id is None:
        cat_audio_class_id = pre_classifier_class_names.index("cat")


def get_thread_models() -> tuple[YOLO, Any, int]:
    """
    Lazily initializes one model bundle per thread when running parallel workers.
    """
    if not hasattr(thread_local, "yolo_model"):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        thread_local.yolo_model = YOLO(MODEL_PATH).to(device)
        thread_local.preclass_model = joblib.load(PRECLASSIFIER_PATH)
        thread_local.cat_audio_class_id = pre_classifier_class_names.index("cat")
    return thread_local.yolo_model, thread_local.preclass_model, thread_local.cat_audio_class_id


def extract_features_pydub(audio_segment: AudioSegment) -> np.ndarray | None:
    try:
        if audio_segment.frame_rate != SAMPLE_RATE:
            audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE)
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)

        audio_array = np.array(audio_segment.get_array_of_samples())
        audio_float = audio_array.astype(np.float32) / (2**15)
        stft = np.abs(librosa.stft(audio_float))
        mfccs = librosa.feature.mfcc(y=audio_float, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

        # librosa delta requires enough time frames.
        if mfccs.shape[1] < 9:
            return None

        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
        centroid = librosa.feature.spectral_centroid(S=stft, sr=SAMPLE_RATE)
        flatness = librosa.feature.spectral_flatness(S=stft)
        rolloff = librosa.feature.spectral_rolloff(S=stft, sr=SAMPLE_RATE)

        return np.hstack(
            [
                np.mean(mfccs.T, axis=0),
                np.mean(mfccs_delta.T, axis=0),
                np.mean(mfccs_delta2.T, axis=0),
                np.mean(centroid.T, axis=0),
                np.mean(flatness.T, axis=0),
                np.mean(rolloff.T, axis=0),
            ]
        )
    except Exception:
        return None


def audio_preclassifier(
    features: np.ndarray,
    prob_threshold: float = 0.5,
    model_override: Any | None = None,
    cat_class_override: int | None = None,
) -> tuple[bool, float]:
    model = model_override if model_override is not None else preclass_model
    cat_class_idx = cat_class_override if cat_class_override is not None else cat_audio_class_id
    if model is None or cat_class_idx is None:
        raise RuntimeError("Audio preclassifier is not initialized.")
    features = features.reshape(1, -1)
    prediction_proba = float(model.predict_proba(features)[0][cat_class_idx])
    return prediction_proba >= prob_threshold, prediction_proba


def cut_video_ffmpeg(input_path: str, output_path: str, start_time: float, duration: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-t",
        str(duration),
        "-i",
        input_path,
        "-c:v",
        "copy",
        "-an",
        "-loglevel",
        "error",
        output_path,
    ]
    subprocess.run(cmd, check=False)


def get_single_cat_segments(video_path: str, model_override: YOLO | None = None) -> list[dict[str, float]]:
    model = model_override if model_override is not None else yolo_model
    if model is None:
        raise RuntimeError("YOLO model is not initialized.")

    scene_list = detect(video_path, ContentDetector())
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return []

    video_segments: list[dict[str, float]] = []

    for scene_idx, (start_time, end_time) in enumerate(scene_list):
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames()
        track_history: dict[int, dict[str, Any]] = {}

        # Sequential access is much faster than random cap.set seeks per frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        while frame_idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.track(
                frame,
                persist=True,
                verbose=False,
                conf=YOLO_CONF,
                imgsz=YOLO_IMGSZ,
                classes=[CAT_CLASS_ID],
            )[0]

            if results.boxes.id is None:
                continue

            ids = results.boxes.id.int().cpu().tolist()
            if len(ids) > 1:
                for tid in ids:
                    track_history.setdefault(tid, {"frames": [], "valid": False})
                    track_history[tid]["valid"] = False
            else:
                tid = ids[0]
                track = track_history.setdefault(tid, {"frames": [], "valid": True})
                if track["valid"]:
                    track["frames"].append(frame_idx / fps)

            # Skip unsampled frames without full decode.
            frames_to_skip = min(FRAME_STRIDE - 1, end_frame - frame_idx - 1)
            skipped = 0
            while skipped < frames_to_skip:
                if not cap.grab():
                    break
                skipped += 1
            frame_idx += skipped + 1

        for tid, data in track_history.items():
            if not data["valid"] or not data["frames"]:
                continue
            seg_start = data["frames"][0]
            seg_end = data["frames"][-1]
            duration = seg_end - seg_start
            if duration >= MIN_SEGMENT_DURATION:
                video_segments.append(
                    {
                        "scene": float(scene_idx),
                        "start": round(seg_start, 2),
                        "end": round(seg_end, 2),
                        "duration": round(duration, 2),
                    }
                )

    cap.release()
    return video_segments


def resolve_single_file(glob_pattern: str) -> str | None:
    matches = glob.glob(glob_pattern)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def build_auth_opts(cookie_file: str | None, cookies_from_browser: str | None) -> dict[str, Any]:
    auth_opts: dict[str, Any] = {}
    if cookie_file:
        auth_opts["cookiefile"] = cookie_file
    if cookies_from_browser:
        # yt-dlp expects a tuple like ("chrome",)
        auth_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return auth_opts


def download_video_assets(
    video_id: str,
    output_folder: str,
    cookie_file: str | None = None,
    cookies_from_browser: str | None = None,
) -> dict[str, Any]:
    os.makedirs(output_folder, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    base_opts = {
        "quiet": True,
        # Match original notebook behavior.
        "js_runtimes": {"node": {}},
        **build_auth_opts(cookie_file=cookie_file, cookies_from_browser=cookies_from_browser),
    }
    try:
        with YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        duration = info.get("duration")
        filesize = info.get("filesize") or info.get("filesize_approx")
        if duration is None or duration > MAX_DOWNLOAD_DURATION_SECONDS:
            return {
                "video_id": video_id,
                "error": "video too long",
                "duration_seconds": duration,
            }
        if filesize and filesize > MAX_VIDEO_FILESIZE_BYTES:
            return {
                "video_id": video_id,
                "error": "video too large",
                "duration_seconds": duration,
                "filesize_bytes": filesize,
            }

        title = info.get("title") or video_id

        video_opts = {
            **base_opts,
            "format": "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]/bestvideo/best",
            "outtmpl": os.path.join(output_folder, "%(id)s_video.%(ext)s"),
            "merge_output_format": "mp4",
            "max_filesize": MAX_VIDEO_FILESIZE_BYTES,
            "noplaylist": True,
            "retries": 3,
            "fragment_retries": 3,
        }
        with YoutubeDL(video_opts) as ydl:
            ydl.download([url])

        audio_opts = {
            **base_opts,
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": os.path.join(output_folder, "%(id)s_audio.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "noplaylist": True,
            "retries": 3,
            "fragment_retries": 3,
        }
        with YoutubeDL(audio_opts) as ydl:
            ydl.download([url])

        video_path = resolve_single_file(os.path.join(output_folder, f"{video_id}_video.*"))
        audio_path = resolve_single_file(os.path.join(output_folder, f"{video_id}_audio.*"))
        if not video_path or not audio_path:
            return {"video_id": video_id, "error": "downloaded files not found"}

        return {
            "video_id": video_id,
            "title": title,
            "video_path": video_path,
            "audio_path": audio_path,
        }
    except Exception as exc:
        return {"video_id": video_id, "error": str(exc)}


def process_downloaded_video(
    video_id: str,
    title: str,
    video_path: str,
    audio_path: str,
    snippets_output_dir: str,
    show_chunk_progress: bool = False,
) -> dict[str, Any] | None:
    os.makedirs(snippets_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    cap.release()
    if duration > MAX_PROCESS_DURATION_SECONDS or duration == 0:
        return None

    model_override = None
    preclass_override = None
    cat_class_override = None
    if yolo_model is None or preclass_model is None or cat_audio_class_id is None:
        model_override, preclass_override, cat_class_override = get_thread_models()

    raw_segments = get_single_cat_segments(video_path, model_override=model_override)
    if not raw_segments:
        return None

    try:
        full_audio = AudioSegment.from_file(audio_path)
    except Exception:
        return None

    valid_chunks: list[dict[str, float]] = []
    for seg in raw_segments:
        curr_start = seg["start"]
        seg_end = seg["end"]
        while (seg_end - curr_start) >= MIN_CHUNK_DURATION:
            chunk_dur = min(MAX_CHUNK_DURATION, seg_end - curr_start)
            valid_chunks.append({"start": curr_start, "duration": chunk_dur})
            curr_start += chunk_dur

    if not valid_chunks:
        return None

    base_name = Path(video_path).stem.replace("_video", "")
    record = {
        "video_id": video_id,
        "original_video": f"{title}.mp4",
        "processed_at": datetime.now().isoformat(),
        "total_chunks_analyzed": len(valid_chunks),
        "saved_chunks_count": 0,
        "snippets": [],
    }

    chunk_iter = valid_chunks
    if show_chunk_progress:
        chunk_iter = tqdm(valid_chunks, desc=f"chunks:{video_id}", leave=False, unit="chunk")

    for i, chunk in enumerate(chunk_iter):
        start_ms = int(chunk["start"] * 1000)
        end_ms = int((chunk["start"] + chunk["duration"]) * 1000)
        audio_chunk = full_audio[start_ms:end_ms]

        features = extract_features_pydub(audio_chunk)
        if features is None:
            continue

        prediction, proba = audio_preclassifier(
            features,
            prob_threshold=0.5,
            model_override=preclass_override,
            cat_class_override=cat_class_override,
        )
        if prediction is True:
            record["saved_chunks_count"] += 1
            snippet_id = f"{base_name}_snip_{i}"
            video_snippet_path = os.path.join(snippets_output_dir, f"{snippet_id}.mp4")
            audio_snippet_path = os.path.join(snippets_output_dir, f"{snippet_id}.mp3")
            cut_video_ffmpeg(video_path, video_snippet_path, chunk["start"], chunk["duration"])
            audio_chunk.export(audio_snippet_path, format="mp3")
            record["snippets"].append(
                {
                    "id": snippet_id,
                    "audio_proba": round(float(proba), 4),
                    "timestamp_range": [
                        round(chunk["start"], 2),
                        round(chunk["start"] + chunk["duration"], 2),
                    ],
                    "duration": round(chunk["duration"], 2),
                }
            )

    return record if record["snippets"] else None


def cleanup_paths(*paths: str) -> None:
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


def process_one_video_id(
    video_id: str,
    work_dir: str,
    snippets_output_dir: str,
    cookie_file: str | None,
    cookies_from_browser: str | None,
    show_chunk_progress: bool,
) -> dict[str, Any]:
    started_at = time.monotonic()
    download_started_at = time.monotonic()
    downloaded = download_video_assets(video_id, work_dir, cookie_file, cookies_from_browser)
    download_seconds = round(time.monotonic() - download_started_at, 2)

    if "error" in downloaded:
        return {
            "video_id": video_id,
            "status": "error",
            "stage": "download",
            "error": downloaded["error"],
            "duration_seconds": downloaded.get("duration_seconds"),
            "filesize_bytes": downloaded.get("filesize_bytes"),
            "download_seconds": download_seconds,
            "total_seconds": round(time.monotonic() - started_at, 2),
        }

    title = downloaded["title"]
    video_path = downloaded["video_path"]
    audio_path = downloaded["audio_path"]
    try:
        process_started_at = time.monotonic()
        record = process_downloaded_video(
            video_id=video_id,
            title=title,
            video_path=video_path,
            audio_path=audio_path,
            snippets_output_dir=snippets_output_dir,
            show_chunk_progress=show_chunk_progress,
        )
        process_seconds = round(time.monotonic() - process_started_at, 2)
        if record is None:
            return {
                "video_id": video_id,
                "status": "skipped",
                "stage": "process",
                "reason": "no valid snippets",
                "download_seconds": download_seconds,
                "process_seconds": process_seconds,
                "total_seconds": round(time.monotonic() - started_at, 2),
            }
        return {
            "video_id": video_id,
            "status": "success",
            "record": record,
            "download_seconds": download_seconds,
            "process_seconds": process_seconds,
            "total_seconds": round(time.monotonic() - started_at, 2),
        }
    finally:
        cleanup_paths(video_path, audio_path)


def print_result_line(video_id: str, result: dict[str, Any], done: int, total: int) -> None:
    parts = [f"[{done}/{total}] {video_id}: {result.get('status', 'unknown')}"]
    if result.get("status") == "error":
        parts.append(f"stage={result.get('stage', 'unknown')}")
        parts.append(result.get("error", "unknown error"))
    elif result.get("status") == "skipped":
        parts.append(result.get("reason", "no reason"))
    if result.get("duration_seconds") is not None:
        parts.append(f"duration={round(float(result['duration_seconds']), 2)}s")
    if result.get("filesize_bytes") is not None:
        parts.append(f"size={int(result['filesize_bytes'])}B")
    if "download_seconds" in result:
        parts.append(f"download={result['download_seconds']}s")
    if "process_seconds" in result:
        parts.append(f"process={result['process_seconds']}s")
    if "total_seconds" in result:
        parts.append(f"total={result['total_seconds']}s")
    tqdm.write(" | ".join(parts))


def run_pipeline(
    gemini_labels_file: str,
    metadata_file: str,
    crawled_candidates_file: str,
    snippets_output_dir: str,
    work_dir: str,
    pipeline_log_file: str,
    prefetch_downloads: int,
    cookie_file: str | None,
    cookies_from_browser: str | None,
    limit: int | None,
    video_workers: int,
    chunk_progress: bool,
    frame_stride: int,
    yolo_imgsz: int,
    yolo_conf: float,
) -> None:
    global FRAME_STRIDE, YOLO_IMGSZ, YOLO_CONF
    FRAME_STRIDE = max(1, int(frame_stride))
    YOLO_IMGSZ = max(160, int(yolo_imgsz))
    YOLO_CONF = max(0.05, min(0.95, float(yolo_conf)))

    os.makedirs(snippets_output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(pipeline_log_file), exist_ok=True)

    workers = max(1, int(video_workers))
    if workers == 1:
        init_pipeline_models(MODEL_PATH, PRECLASSIFIER_PATH)

    todo_video_ids = build_todo_video_ids(
        gemini_labels_file=gemini_labels_file,
        metadata_file=metadata_file,
        crawled_candidates_file=crawled_candidates_file,
        pipeline_log_file=pipeline_log_file,
    )
    if isinstance(limit, int) and limit > 0:
        todo_video_ids = todo_video_ids[:limit]
    print(f"Found {len(todo_video_ids)} kept+unprocessed videos to handle.")
    if not todo_video_ids:
        return
    total = len(todo_video_ids)
    if workers > 1:
        print(f"Using parallel video workers: {workers}")
        print("Each worker runs full pipeline: download -> process -> delete.")
        print(f"Perf params: frame_stride={FRAME_STRIDE}, yolo_imgsz={YOLO_IMGSZ}, yolo_conf={YOLO_CONF}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending: dict[Any, str] = {}
            todo_iter = iter(todo_video_ids)

            # Keep a small rolling queue of tasks to reduce overhead and memory use.
            queue_size = workers * 2
            for _ in range(min(queue_size, total)):
                next_video = next(todo_iter, None)
                if next_video is None:
                    break
                fut = executor.submit(
                    process_one_video_id,
                    next_video,
                    work_dir,
                    snippets_output_dir,
                    cookie_file,
                    cookies_from_browser,
                    False,
                )
                pending[fut] = next_video

            done = 0
            with tqdm(total=total, desc="videos", unit="video") as pbar:
                while pending:
                    completed_future = next(as_completed(pending))
                    video_id = pending.pop(completed_future)
                    next_video = next(todo_iter, None)
                    if next_video is not None:
                        fut = executor.submit(
                            process_one_video_id,
                            next_video,
                            work_dir,
                            snippets_output_dir,
                            cookie_file,
                            cookies_from_browser,
                            False,
                        )
                        pending[fut] = next_video

                    result = completed_future.result()
                    done += 1
                    if result.get("status") == "success":
                        append_jsonl(metadata_file, result["record"])
                    append_jsonl(pipeline_log_file, result)
                    print_result_line(video_id, result, done, total)
                    pbar.update(1)
        return

    if prefetch_downloads > 1:
        print("Note: --prefetch-downloads is ignored when --video-workers=1.")
    print(f"Perf params: frame_stride={FRAME_STRIDE}, yolo_imgsz={YOLO_IMGSZ}, yolo_conf={YOLO_CONF}")

    with tqdm(total=total, desc="videos", unit="video") as pbar:
        for done, video_id in enumerate(todo_video_ids, start=1):
            result = process_one_video_id(
                video_id=video_id,
                work_dir=work_dir,
                snippets_output_dir=snippets_output_dir,
                cookie_file=cookie_file,
                cookies_from_browser=cookies_from_browser,
                show_chunk_progress=chunk_progress,
            )
            if result.get("status") == "success":
                append_jsonl(metadata_file, result["record"])
            append_jsonl(pipeline_log_file, result)
            print_result_line(video_id, result, done, total)
            pbar.update(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Disk-friendly streaming pipeline: download -> process -> save snippets -> delete."
    )
    parser.add_argument("--gemini-labels", default=GEMINI_LABELS_FILE)
    parser.add_argument("--metadata-file", default=METADATA_FILE)
    parser.add_argument("--crawled-candidates", default=CRAWLED_CANDIDATES_FILE)
    parser.add_argument("--snippets-output-dir", default=SNIPPETS_OUTPUT_DIR)
    parser.add_argument("--work-dir", default=WORK_DIR)
    parser.add_argument("--pipeline-log-file", default=PIPELINE_LOG_FILE)
    parser.add_argument(
        "--prefetch-downloads",
        type=int,
        default=2,
        help="Legacy flag; currently ignored (use --video-workers for parallelism).",
    )
    parser.add_argument(
        "--cookie-file",
        default=None,
        help="Path to Netscape format cookies.txt used by yt-dlp.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        choices=["chrome", "chromium", "brave", "edge", "firefox", "safari", "opera", "vivaldi"],
        help="Read cookies directly from a local browser profile.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N candidate videos (debugging/smoke tests).",
    )
    parser.add_argument(
        "--video-workers",
        type=int,
        default=1,
        help="Run full video pipelines in parallel (download+process+cleanup).",
    )
    parser.add_argument(
        "--chunk-progress",
        action="store_true",
        help="Show per-chunk tqdm for currently processed video (single-worker mode).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=3,
        help="Sample every N-th frame for visual tracking (higher = faster).",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=640,
        help="YOLO inference image size (lower = faster).",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.4,
        help="YOLO confidence threshold (higher may reduce detections).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        gemini_labels_file=args.gemini_labels,
        metadata_file=args.metadata_file,
        crawled_candidates_file=args.crawled_candidates,
        snippets_output_dir=args.snippets_output_dir,
        work_dir=args.work_dir,
        pipeline_log_file=args.pipeline_log_file,
        prefetch_downloads=args.prefetch_downloads,
        cookie_file=args.cookie_file,
        cookies_from_browser=args.cookies_from_browser,
        limit=args.limit,
        video_workers=args.video_workers,
        chunk_progress=args.chunk_progress,
        frame_stride=args.frame_stride,
        yolo_imgsz=args.yolo_imgsz,
        yolo_conf=args.yolo_conf,
    )
