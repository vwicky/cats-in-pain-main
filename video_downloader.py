import os
import json
import cv2
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pydub import AudioSegment
from ultralytics import YOLO
from scenedetect import detect, ContentDetector

# =========================
# CONFIG
# =========================

MODEL_PATH = "yolo11m.pt"
CAT_CLASS_ID = 15
MIN_SEGMENT_DURATION = 3.0
FRAME_STRIDE = 3

OUTPUT_DIR = "dataset_snippets"
METADATA_FILE = "metadata.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model (loaded once per worker)
model = None

# =========================
# WORKER INITIALIZER
# =========================

def init_worker():
    global model
    model = YOLO(MODEL_PATH).to("mps")

# =========================
# HELPER FUNCTIONS
# =========================

def load_processed_videos():
    processed = set()
    if not os.path.exists(METADATA_FILE):
        return processed
    with open(METADATA_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add(data["original_video"])
            except Exception:
                continue
    return processed

def get_single_cat_segments(video_path):
    global model
    scene_list = detect(video_path, ContentDetector())
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_segments = []

    for scene_idx, (start_time, end_time) in enumerate(scene_list):
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames()
        track_history = {}

        # REMOVE: model.tracker.reset()

        for frame_idx in range(start_frame, end_frame, FRAME_STRIDE):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.4,
                classes=[CAT_CLASS_ID]
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

        for tid, data in track_history.items():
            if data["valid"] and data["frames"]:
                seg_start = data["frames"][0]
                seg_end = data["frames"][-1]
                duration = seg_end - seg_start
                if duration >= MIN_SEGMENT_DURATION:
                    video_segments.append({
                        "scene": scene_idx,
                        "start": round(seg_start, 2),
                        "end": round(seg_end, 2),
                        "duration": round(duration, 2)
                    })

    cap.release()
    return video_segments

def process_single_video(args):
    directory, filename = args
    base_name = filename.replace("_video.mp4", "")
    video_path = os.path.join(directory, f"{base_name}_video.mp4")
    audio_path = os.path.join(directory, f"{base_name}_audio.mp3")

    if not os.path.exists(audio_path):
        return None

    # duration check
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    cap.release()
    if duration > 300 or duration == 0:
        return None

    # visual stage
    raw_segments = get_single_cat_segments(video_path)
    if not raw_segments:
        return None

    # audio load
    try:
        full_audio = AudioSegment.from_file(audio_path)
    except Exception:
        print(f"⚠️ Corrupted audio skipped: {audio_path}")
        return None

    # chunking
    valid_chunks = []
    for seg in raw_segments:
        curr_start = seg["start"]
        seg_end = seg["end"]
        while (seg_end - curr_start) >= 3.0:
            chunk_dur = min(7.0, seg_end - curr_start)
            valid_chunks.append({"start": curr_start, "duration": chunk_dur})
            curr_start += chunk_dur

    if not valid_chunks:
        return None

    video_record = {
        "original_video": filename,
        "processed_at": datetime.now().isoformat(),
        "total_chunks_analyzed": len(valid_chunks),
        "saved_chunks_count": 0,
        "snippets": []
    }

    # audio classification stage
    for i, chunk in enumerate(valid_chunks):
        start_ms = int(chunk["start"] * 1000)
        end_ms = int((chunk["start"] + chunk["duration"]) * 1000)
        audio_chunk = full_audio[start_ms:end_ms]

        features = extract_features_pydub(audio_chunk)
        if features is None:
            continue

        prediction, proba = audio_preclassifier(features, prob_threshold=0.5)
        if prediction == 1:
            video_record["saved_chunks_count"] += 1
            snip_base = f"{base_name}_snip_{i}"
            vid_snip_path = os.path.join(OUTPUT_DIR, f"{snip_base}.mp4")
            aud_snip_path = os.path.join(OUTPUT_DIR, f"{snip_base}.mp3")
            cut_video_ffmpeg(video_path, vid_snip_path, chunk["start"], chunk["duration"])
            audio_chunk.export(aud_snip_path, format="mp3")
            video_record["snippets"].append({
                "id": snip_base,
                "audio_proba": round(float(proba), 4),
                "timestamp_range": [
                    round(chunk["start"], 2),
                    round(chunk["start"] + chunk["duration"], 2)
                ],
                "duration": round(chunk["duration"], 2)
            })

    if video_record["snippets"]:
        return video_record
    return None

def process_directory(directory):
    video_files = [f for f in os.listdir(directory) if f.endswith("_video.mp4")]
    processed = load_processed_videos()
    video_files = [v for v in video_files if v not in processed]

    print(f"🚀 Found {len(video_files)} videos to process")
    tasks = [(directory, f) for f in video_files]
    workers = max(cpu_count() - 1, 1)

    with Pool(workers, initializer=init_worker) as pool, open(METADATA_FILE, "a") as f_metadata:
        for result in tqdm(pool.imap_unordered(process_single_video, tasks),
                           total=len(tasks),
                           desc="Overall Progress",
                           unit="video"):
            if result:
                f_metadata.write(json.dumps(result) + "\n")
                f_metadata.flush()

    print(f"\n✅ Done. Metadata saved to {METADATA_FILE}")


# =========================
# RUN SCRIPT
# =========================
if __name__ == "__main__":
    process_directory("crawled_downloads")