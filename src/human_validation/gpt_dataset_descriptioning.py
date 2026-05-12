import argparse
import base64
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = """
You are labeling a cat-video dataset for a paper.
Your task is to inspect 3 frames from the same video (beginning, middle, end) and assign coarse, visually grounded dataset attributes.

Rules:
- Use only what is visible in the frames.
- Do not guess.
- Prefer "uncertain" or "unknown" over speculation.
- Do not infer emotion.
- Do not produce prose or explanations.
- Return only a JSON object matching the schema.
- If a breed is not obvious, do NOT force an exact breed name.
- For breed/type, use coarse visual categories only.

Cross-field consistency rules (you must follow these):
- If other_animals_present is "no", other_animals must be [].
- If other_animals_present is "yes", other_animals must contain at least one entry.
- If other_animals_present is "uncertain", other_animals must be [].
- If cat_present is "no", num_cats must be "uncertain".

Interpretation guidance:
- environment: indoor / outdoor / mixed / uncertain
- scene_type: home / street / nature_or_garden / vehicle / shelter_or_vet / other / uncertain
- lighting: bright_daylight / normal_indoor / low_light / night / backlit / uncertain
- occlusion: none / partial / heavy / uncertain
- visual_clarity: clear / slightly_blurry / blurry / highly_compressed / uncertain
- other_animals: list the visible animal types only; use unknown if not clear
- cat_visual_type: coarse appearance class, not exact breed
- overall_confidence: high = clear frames and nearly all fields are certain;
                      medium = one or two fields are uncertain or visibility is partial;
                      low = poor visibility, heavy occlusion, or the majority of fields are uncertain
""".strip()


JSON_SCHEMA = {
    "name": "cat_video_dataset_audit",
    "schema": {
        "type": "object",
        "properties": {
            "cat_present": {
                "type": "string",
                "enum": ["yes", "no", "uncertain"],
            },
            "num_cats": {
                "type": "string",
                "enum": ["one", "multiple", "uncertain"],
            },
            "environment": {
                "type": "string",
                "enum": ["indoor", "outdoor", "mixed", "uncertain"],
            },
            "scene_type": {
                "type": "string",
                "enum": [
                    "home",
                    "street",
                    "nature_or_garden",
                    "vehicle",
                    "shelter_or_vet",
                    "other",
                    "uncertain",
                ],
            },
            "lighting": {
                "type": "string",
                "enum": [
                    "bright_daylight",
                    "normal_indoor",
                    "low_light",
                    "night",
                    "backlit",
                    "uncertain",
                ],
            },
            "occlusion": {
                "type": "string",
                "enum": ["none", "partial", "heavy", "uncertain"],
            },
            "visual_clarity": {
                "type": "string",
                "enum": [
                    "clear",
                    "slightly_blurry",
                    "blurry",
                    "highly_compressed",
                    "uncertain",
                ],
            },
            "human_presence": {
                "type": "string",
                "enum": ["yes", "no", "uncertain"],
            },
            "other_animals_present": {
                "type": "string",
                "enum": ["yes", "no", "uncertain"],
            },
            "other_animals": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "another_cat",
                        "dog",
                        "bird",
                        "rodent",
                        "livestock",
                        "wild_animal",
                        "unknown",
                    ],
                },
            },
            "cat_visual_type": {
                "type": "string",
                "enum": [
                    "domestic_shorthair",
                    "domestic_longhair",
                    "siamese_like",
                    "tabby",
                    "calico",
                    "tuxedo",
                    "ginger",
                    "black",
                    "white",
                    "mixed_or_multiple_cats",
                    "unknown",
                ],
            },
            "overall_confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
        },
        "required": [
            "cat_present",
            "num_cats",
            "environment",
            "scene_type",
            "lighting",
            "occlusion",
            "visual_clarity",
            "human_presence",
            "other_animals_present",
            "other_animals",
            "cat_visual_type",
            "overall_confidence",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}


def _frame_to_data_url(frame_bgr) -> str:
    """Convert an OpenCV BGR frame to a base64 JPEG data URL."""
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("Could not encode frame as JPEG.")
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def sample_three_frames(video_path: str) -> List[Any]:
    """
    Sample 3 frames from the beginning, middle, and end of a video.
    Returns a list of OpenCV BGR frames.

    Uses time-based seek first, then frame-index seek, then sequential decode.
    Many MP4/H.264 files report an inflated FRAME_COUNT or fail random seeks;
    sequential read returns the last successfully decoded frame when the file
    is shorter than the metadata claims.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps is None or fps < 1e-3:
            fps = 30.0

        total_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_reported <= 0:
            raise ValueError(f"Could not determine frame count for: {video_path}")

        last_idx = max(0, total_reported - 1)
        idx_begin = int(round(last_idx * 0.01))
        idx_mid = int(round(last_idx * 0.50))
        idx_end = int(round(last_idx * 0.99))

        duration_ms = (total_reported / fps) * 1000.0
        if duration_ms <= 0:
            duration_ms = 1.0

        def read_at_msec(msec: float):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, msec))
            ok, fr = cap.read()
            return fr if ok and fr is not None else None

        def read_at_frame_index(idx: int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            return fr if ok and fr is not None else None

        def read_sequential_upto(idx: int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            last_good = None
            for _ in range(idx + 1):
                ok, fr = cap.read()
                if not ok or fr is None:
                    return last_good
                last_good = fr
            return last_good

        def sample_one(idx: int, frac: float):
            # 1) Seek by timestamp (often works when frame-index seek fails)
            msec = min(max(0.0, duration_ms * frac), duration_ms - 1e-3)
            fr = read_at_msec(msec)
            if fr is None and msec > 1.0:
                fr = read_at_msec(max(0.0, msec - 50.0))
            if fr is None:
                fr = read_at_frame_index(idx)
            if fr is None:
                fr = read_sequential_upto(idx)
            return fr

        frames = [
            sample_one(idx_begin, 0.01),
            sample_one(idx_mid, 0.50),
            sample_one(idx_end, 0.99),
        ]
        if any(f is None for f in frames):
            raise ValueError(f"Could not decode enough frames from: {video_path}")
        return frames
    finally:
        cap.release()


def analyze_video_with_gpt_54(video_path: str) -> Union[Dict[str, Any], str]:
    """
    Sample 3 frames from a video and label them with GPT-5.4.
    Returns:
        - dict on success
        - raw string on failure to parse JSON
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    try:
        frames = sample_three_frames(video_path)
        image_urls = [_frame_to_data_url(f) for f in frames]

        user_content = [
            {
                "type": "input_text",
                "text": (
                    "Analyze these three frames from the same cat video in order: "
                    "beginning, middle, end. Return a single video-level JSON record."
                ),
            }
        ]

        for idx, url in enumerate(image_urls, start=1):
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": url,
                    "detail": "low",
                }
            )

        # Structured outputs: Responses API uses `text.format`, not `response_format`.
        response = client.responses.create(
            model="gpt-5.4",
            instructions=SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    **JSON_SCHEMA,
                }
            },
            temperature=0,
            max_output_tokens=300,
        )

        raw_text = getattr(response, "output_text", None)
        if not raw_text:
            # Fallback if the SDK shape changes
            raw_text = str(response)

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text

    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}


def _validation_root_default() -> Path:
    return Path(__file__).resolve().parent / "video_audio_human_validation"


def discover_videos(validation_root: Path) -> List[Tuple[Path, str]]:
    """
    Return (absolute_path, folder_label) for each video under cat/ and non-cat/.
    folder_label is \"cat\" or \"non-cat\".
    """
    out: List[Tuple[Path, str]] = []
    for folder_label, sub in (("cat", "cat"), ("non-cat", "non-cat")):
        d = validation_root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                out.append((p.resolve(), folder_label))
    return out


def load_recorded_video_paths(jsonl_path: Path) -> Set[str]:
    """Paths already present in the jsonl (for --skip-existing)."""
    seen: Set[str] = set()
    if not jsonl_path.is_file():
        return seen
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                vp = obj.get("video_path")
                if isinstance(vp, str):
                    seen.add(vp)
            except json.JSONDecodeError:
                continue
    return seen


def _gpt_success(result: Union[Dict[str, Any], str]) -> bool:
    return isinstance(result, dict)

def _validate_consistency(result: Dict[str, Any]) -> List[str]:
    """Returns a list of consistency violations, empty if clean."""
    violations = []

    oap = result.get("other_animals_present")
    oa = result.get("other_animals", [])

    if oap == "no" and oa:
        violations.append(f"other_animals_present=no but other_animals={oa}")
    if oap == "yes" and not oa:
        violations.append("other_animals_present=yes but other_animals=[]")
    if oap == "uncertain" and oa:
        violations.append(f"other_animals_present=uncertain but other_animals={oa}")
    if result.get("cat_present") == "no" and result.get("num_cats") != "uncertain":
        violations.append(
            f"cat_present=no but num_cats={result.get('num_cats')} (expected uncertain)"
        )

    return violations

def process_single_video(
    video_path: Path,
    folder_label: str,
    validation_root: Path,
    output_jsonl: Path,
    write_lock: threading.Lock,
) -> None:
    """Analyze one video and append one JSON line to the shared jsonl file."""
    analyzed_at = datetime.now(timezone.utc).isoformat()
    result = analyze_video_with_gpt_54(str(video_path))

    vr = validation_root.resolve()
    try:
        rel = str(video_path.relative_to(vr))
    except ValueError:
        rel = str(video_path)

    parse_ok = _gpt_success(result)
    violations = _validate_consistency(result) if parse_ok else []

    record: Dict[str, Any] = {
        "video_path": str(video_path),
        "folder_label": folder_label,
        "relative_to_validation_root": rel,
        "analyzed_at_utc": analyzed_at,
        "model": "gpt-5.4",
        "gpt_response": result,
        "parse_ok": parse_ok,
        "consistency_violations": violations,   # [] means clean
    }

    line = json.dumps(record, ensure_ascii=False) + "\n"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with write_lock:
        with open(output_jsonl, "a", encoding="utf-8") as f:
            f.write(line)


def _round_robin_chunks(
    items: List[Tuple[Path, str]], n_workers: int
) -> List[List[Tuple[Path, str]]]:
    if not items:
        return []
    n_workers = max(1, min(n_workers, len(items)))
    buckets: List[List[Tuple[Path, str]]] = [[] for _ in range(n_workers)]
    for i, item in enumerate(items):
        buckets[i % n_workers].append(item)
    return buckets


def _worker_run(
    worker_id: int,
    tasks: List[Tuple[Path, str]],
    validation_root: Path,
    output_jsonl: Path,
    write_lock: threading.Lock,
) -> None:
    desc = f"worker {worker_id}"
    for video_path, folder_label in tqdm(
        tasks,
        position=worker_id,
        desc=desc,
        leave=True,
        dynamic_ncols=True,
    ):
        process_single_video(
            video_path,
            folder_label,
            validation_root,
            output_jsonl,
            write_lock,
        )


def batch_describe_videos(
    validation_root: Optional[Path] = None,
    output_jsonl: Optional[Path] = None,
    num_workers: int = 4,
    skip_existing: bool = False,
) -> None:
    """
    Walk cat/ and non-cat/ under validation_root, call analyze_video_with_gpt_54
    for each video in parallel (threads). Appends one JSON object per line to a
    single jsonl file.
    """
    vr = (validation_root or _validation_root_default()).resolve()
    out_path = output_jsonl or (vr / "gpt_dataset_descriptions.jsonl")
    out_path = out_path.resolve()

    tasks = discover_videos(vr)
    if not tasks:
        print(f"No videos found under {vr / 'cat'} or {vr / 'non-cat'}")
        return

    if skip_existing:
        done = load_recorded_video_paths(out_path)
        before = len(tasks)
        tasks = [(p, lab) for p, lab in tasks if str(p) not in done]
        skipped = before - len(tasks)
        if skipped:
            print(f"Skipping {skipped} videos already in {out_path}")
        if not tasks:
            print("Nothing left to process.")
            return

    chunks = _round_robin_chunks(tasks, num_workers)
    write_lock = threading.Lock()
    threads: List[threading.Thread] = []
    for wid, chunk in enumerate(chunks):
        if not chunk:
            continue
        t = threading.Thread(
            target=_worker_run,
            args=(wid, chunk, vr, out_path, write_lock),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Done. Appended {len(tasks)} lines to {out_path}.")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Label cat/non-cat videos with GPT and append results to one jsonl file."
    )
    p.add_argument(
        "--validation-root",
        type=Path,
        default=None,
        help="Folder containing cat/ and non-cat/ (default: video_audio_human_validation next to this script).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        dest="output_jsonl",
        help="Path to the output .jsonl file (default: <validation-root>/gpt_dataset_descriptions.jsonl).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (each has its own tqdm bar).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos whose video_path already appears in the output jsonl.",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    batch_describe_videos(
        validation_root=args.validation_root,
        output_jsonl=args.output_jsonl,
        num_workers=args.workers,
        skip_existing=args.skip_existing,
    )