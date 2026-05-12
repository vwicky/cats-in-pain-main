#!/usr/bin/env python3
"""
Render ViTPose overlays on a video using the same flow as easy_ViTPose's inference.py:
VitInference(..., is_video=True), frame loop, model.inference(RGB), model.draw().

Default output is a fixed file under the system temp directory so each run overwrites
the previous render (no (1), (2) suffixes).

Usage (from project root):
  python scripts/render_pose_video.py path/to/video.mp4
  python scripts/render_pose_video.py path/to/video.mp4 -o /tmp/my_overlay.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
from tqdm import tqdm

# Project root = video/easyViTPose/  (parent of extraction/)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from easy_ViTPose import VitInference  # noqa: E402
from easy_ViTPose.vit_utils.inference import VideoReader  # noqa: E402


def default_output_path() -> Path:
    out_dir = Path(tempfile.gettempdir()) / "cats_in_pain_vitpose"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "pose_overlay.mp4"


def _open_writer(path: Path, fps: float, frame_size_wh: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Overwrite: VideoWriter truncates when creating a new file at the same path
    for tag in ("mp4v", "avc1", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*tag)
        w = cv2.VideoWriter(str(path), fourcc, float(fps), frame_size_wh)
        if w.isOpened():
            return w
    raise RuntimeError(f"Could not open VideoWriter for {path}")


def render_pose_video(
    input_video: str | Path,
    output_video: str | Path | None = None,
    *,
    model: str = "./models/pose_est/vitpose-l-ap10k.pth",
    yolo: str = "./yolo11m.pt",
    model_name: str = "l",
    dataset: str = "ap10k",
    det_class: str = "cat",
    yolo_size: int = 320,
    yolo_step: int = 1,
    single_pose: bool = False,
    show_yolo: bool = True,
    conf_threshold: float = 0.5,
    rotate: int = 0,
) -> Path:
    input_video = Path(input_video).resolve()
    if not input_video.is_file():
        raise FileNotFoundError(input_video)

    out = Path(output_video) if output_video else default_output_path()
    out = out.resolve()

    _orig = os.getcwd()
    try:
        os.chdir(_ROOT)
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first = cap.read()
        cap.release()
        if not ret or first is None:
            raise RuntimeError(f"Could not read first frame from {input_video}")
        h, w = first.shape[:2]
        frame_size = (w, h)

        model_obj = VitInference(
            str(model),
            str(yolo),
            model_name=model_name,
            dataset=dataset,
            det_class=det_class,
            yolo_size=yolo_size,
            is_video=True,
            single_pose=single_pose,
            yolo_step=yolo_step,
        )
        model_obj.reset()

        writer = _open_writer(out, fps, frame_size)

        reader = VideoReader(str(input_video), rotate=rotate)
        for img in tqdm(reader, total=total if total > 0 else None, desc="pose overlay"):
            model_obj.inference(img)
            # draw() returns RGB; VideoWriter expects BGR (same as upstream inference.py)
            bgr = model_obj.draw(
                show_yolo=show_yolo,
                show_raw_yolo=False,
                confidence_threshold=conf_threshold,
            )[..., ::-1]
            if bgr.shape[1] != w or bgr.shape[0] != h:
                bgr = cv2.resize(bgr, frame_size)
            writer.write(bgr)

        writer.release()
    finally:
        os.chdir(_orig)

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Render ViTPose on a video; default output overwrites each run.")
    p.add_argument("input_video", type=str, help="Path to input .mp4 (or similar)")
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=f"Output path (default: {default_output_path()})",
    )
    p.add_argument("--model", type=str, default="./models/pose_est/vitpose-l-ap10k.pth")
    p.add_argument("--yolo", type=str, default="./yolo11m.pt")
    p.add_argument("--model-name", type=str, default="l", choices=["s", "b", "l", "h"])
    p.add_argument("--dataset", type=str, default="ap10k")
    p.add_argument("--det-class", type=str, default="cat")
    p.add_argument("--yolo-size", type=int, default=320)
    p.add_argument("--yolo-step", type=int, default=1)
    p.add_argument("--single-pose", action="store_true", help="Skip SORT tracker (faster, single subject)")
    p.add_argument("--no-yolo-boxes", action="store_true", help="Do not draw YOLO boxes")
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    args = p.parse_args()

    out = render_pose_video(
        args.input_video,
        args.output,
        model=args.model,
        yolo=args.yolo,
        model_name=args.model_name,
        dataset=args.dataset,
        det_class=args.det_class,
        yolo_size=args.yolo_size,
        yolo_step=args.yolo_step,
        single_pose=args.single_pose,
        show_yolo=not args.no_yolo_boxes,
        conf_threshold=args.conf_threshold,
        rotate=args.rotate,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
