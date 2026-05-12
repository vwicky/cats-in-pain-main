#!/usr/bin/env python3
"""
Merge per-snippet outputs from steps 01–04 into final_dataset_v2.jsonl.

Run from repository root:

  python dataset_construction/05_final_dataset.py
  python dataset_construction/05_final_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".webm")

SNIPPET_PATH_KEYS = (
    "path",
    "video_path",
    "file",
    "snippet_path",
    "video_file",
    "local_path",
)

# Same 10-class order as 04_audio_classification.py / prob_* columns
CLASS_ORDER_10: list[str] = [
    "Angry",
    "Defence",
    "Fighting",
    "Happy",
    "HuntingMind",
    "Mating",
    "MotherCall",
    "Paining",
    "Resting",
    "Warning",
]

PROB_FIELD_NAMES: list[str] = [
    "prob_Angry",
    "prob_Defence",
    "prob_Fighting",
    "prob_Happy",
    "prob_HuntingMind",
    "prob_Mating",
    "prob_MotherCall",
    "prob_Paining",
    "prob_Resting",
    "prob_Warning",
]

OUTPUT_ROW_KEYS: list[str] = [
    "snippet_id",
    "video_id",
    "platform",
    "channel",
    "behavioral_category",
    "duration_sec",
    "start_sec",
    "end_sec",
    "video_path",
    "audio_path",
    "audio_label_10",
    "audio_label_5",
    "audio_label_binary",
    "audio_confidence",
    "audio_high_confidence",
    "label_source",
    "final_label_5",
    "final_label_binary",
    "cat_id",
    "cat_id_method",
    "cluster_size",
    "bbox_found",
    "clip_similarity_to_nearest",
    "gpt_verified",
    "gpt_resolution",
    "gpt_blur",
    "gpt_lighting",
    "gpt_occlusion",
    "gpt_camera_motion",
    "gpt_is_ai_generated",
    "gpt_n_cats_visible",
    "gpt_breed_guess",
    "gpt_coat_color",
    "gpt_coat_pattern",
    "gpt_age_guess",
    "gpt_setting",
    "gpt_location_type",
    "gpt_human_interaction",
    "gpt_primary_behavior",
    "gpt_face_clearly_visible",
    "gpt_pain_indicators_visible",
    "gpt_vet_clinic_confirmed",
    "gpt_suitable_for_training",
    "gpt_exclusion_reason",
    "suitable_for_training",
    "unsuitable_reason",
] + PROB_FIELD_NAMES


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_manifest_records(path: Path):
    """
    Parse JSONL records; tolerate embedded newlines inside string fields (multi-line rows)
    and skip trailing comment lines (e.g. ``# {...}``).
    """
    buf = ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not buf and line.lstrip().startswith("#"):
                continue
            buf += line
            chunk = buf.strip()
            if not chunk:
                buf = ""
                continue
            try:
                rec = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            buf = ""
            yield rec


def _candidate_paths_from_snippet(snippet: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k in SNIPPET_PATH_KEYS:
        v = snippet.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    for k, v in snippet.items():
        if k in SNIPPET_PATH_KEYS or k == "id":
            continue
        if isinstance(v, str) and ("/" in v or "\\" in v):
            low = v.lower()
            if any(low.endswith(ext) for ext in VIDEO_EXTS):
                out.append(v.strip())
    return out


def resolve_snippet_video(
    repo_root: Path,
    snippet: dict[str, Any],
    snippets_dirs: list[Path],
) -> str | None:
    sid = snippet.get("id")
    if not isinstance(sid, str) or not sid:
        return None

    for raw in _candidate_paths_from_snippet(snippet):
        p = Path(raw)
        if p.is_file():
            return str(p.resolve())
        q = repo_root / raw
        if q.is_file():
            return str(q.resolve())

    for base in snippets_dirs:
        for ext in VIDEO_EXTS:
            cand = base / f"{sid}{ext}"
            if cand.is_file():
                return str(cand.resolve())
    return None


def relpath_under_repo(abs_path: str | None, repo_root: Path) -> str | None:
    if not abs_path:
        return None
    try:
        return str(Path(abs_path).resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return abs_path


def load_audio_predictions(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """stem -> row (last wins); return duplicate stem count."""
    by_stem: dict[str, dict[str, Any]] = {}
    dup = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fp = row.get("filepath") or row.get("relpath")
            if not fp:
                continue
            stem = Path(str(fp)).stem
            if stem in by_stem:
                dup += 1
            by_stem[stem] = row
    return by_stem, dup


def load_gpt_descriptions_aggregate(path: Path) -> dict[str, float]:
    """Totals for merge report."""
    total_in = 0.0
    total_out = 0.0
    cost = 0.0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            total_in += float(r.get("input_tokens") or 0)
            total_out += float(r.get("output_tokens") or 0)
            cost += float(r.get("cost_usd") or 0)
    return {
        "n_lines": n,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": cost,
    }


def margin_top1_top2(probs: list[float]) -> float:
    if len(probs) < 2:
        return 0.0
    s = sorted(probs, reverse=True)
    return float(s[0] - s[1])


def flatten_gpt(gpt: dict[str, Any] | None) -> dict[str, Any]:
    if not gpt or not isinstance(gpt, dict):
        return {
            "gpt_resolution": None,
            "gpt_blur": None,
            "gpt_lighting": None,
            "gpt_occlusion": None,
            "gpt_camera_motion": None,
            "gpt_is_ai_generated": None,
            "gpt_n_cats_visible": None,
            "gpt_breed_guess": None,
            "gpt_coat_color": None,
            "gpt_coat_pattern": None,
            "gpt_age_guess": None,
            "gpt_setting": None,
            "gpt_location_type": None,
            "gpt_human_interaction": None,
            "gpt_primary_behavior": None,
            "gpt_face_clearly_visible": None,
            "gpt_pain_indicators_visible": None,
            "gpt_vet_clinic_confirmed": None,
            "gpt_suitable_for_training": None,
            "gpt_exclusion_reason": None,
        }
    vq = gpt.get("video_quality") or {}
    cats = gpt.get("cats") or {}
    primary = (cats.get("primary_cat") or {}) if isinstance(cats.get("primary_cat"), dict) else {}
    env = gpt.get("environment") or {}
    beh = gpt.get("behavior") or {}
    flags = gpt.get("dataset_flags") or {}
    coat_color = primary.get("coat_color")
    if isinstance(coat_color, list):
        coat_color_str = ", ".join(str(x) for x in coat_color)
    else:
        coat_color_str = coat_color
    return {
        "gpt_resolution": vq.get("resolution"),
        "gpt_blur": vq.get("blur"),
        "gpt_lighting": vq.get("lighting"),
        "gpt_occlusion": vq.get("occlusion"),
        "gpt_camera_motion": vq.get("camera_motion"),
        "gpt_is_ai_generated": vq.get("is_ai_generated"),
        "gpt_n_cats_visible": cats.get("n_cats_visible"),
        "gpt_breed_guess": primary.get("breed_guess"),
        "gpt_coat_color": coat_color_str,
        "gpt_coat_pattern": primary.get("coat_pattern"),
        "gpt_age_guess": primary.get("age_guess"),
        "gpt_setting": env.get("setting"),
        "gpt_location_type": env.get("location_type"),
        "gpt_human_interaction": env.get("human_interaction"),
        "gpt_primary_behavior": beh.get("primary_behavior"),
        "gpt_face_clearly_visible": beh.get("face_clearly_visible"),
        "gpt_pain_indicators_visible": flags.get("pain_indicators_visible"),
        "gpt_vet_clinic_confirmed": flags.get("vet_clinic_confirmed"),
        "gpt_suitable_for_training": flags.get("suitable_for_training"),
        "gpt_exclusion_reason": flags.get("exclusion_reason"),
    }


def cat_id_method_blocked(method: str | None) -> bool:
    if not method or not isinstance(method, str):
        return False
    return method.startswith("no_cat_") or method.startswith("unsuitable_")


def compute_unsuitable_reason_and_flags(
    gpt: dict[str, Any] | None,
    cat_id_method: str | None,
    audio_join_hit: bool,
    audio_high_conf: bool | None,
) -> tuple[str | None, list[str]]:
    """
    Primary reason (first failing check in order). Also return list of all failed checks
    for the 'multiple conditions' report bucket.
    """
    failed: list[str] = []
    gpt = gpt or {}
    flags = gpt.get("dataset_flags") or {}
    vq = gpt.get("video_quality") or {}

    if flags.get("suitable_for_training") is not True:
        failed.append("gpt_not_suitable")
    if vq.get("is_ai_generated") is True:
        failed.append("gpt_ai_generated")
    if cat_id_method_blocked(cat_id_method):
        failed.append("cat_id_blocked")
    if not audio_join_hit:
        failed.append("audio_missing")
    elif audio_high_conf is not True:
        failed.append("audio_low_confidence")

    if not failed:
        return None, []

    order = [
        "gpt_not_suitable",
        "gpt_ai_generated",
        "cat_id_blocked",
        "audio_missing",
        "audio_low_confidence",
    ]
    primary = next((k for k in order if k in failed), failed[0])
    label_map = {
        "gpt_not_suitable": "gpt_not_suitable_for_training",
        "gpt_ai_generated": "gpt_ai_generated",
        "cat_id_blocked": "cat_id_unsuitable",
        "audio_missing": "audio_missing",
        "audio_low_confidence": "audio_low_confidence",
    }
    return label_map.get(primary, primary), failed


def suitable_from_flags(
    gpt: dict[str, Any] | None,
    cat_id_method: str | None,
    audio_high_confidence: bool | None,
) -> bool:
    gpt = gpt or {}
    flags = gpt.get("dataset_flags") or {}
    vq = gpt.get("video_quality") or {}
    if flags.get("suitable_for_training") is not True:
        return False
    if vq.get("is_ai_generated") is True:
        return False
    if cat_id_method_blocked(cat_id_method):
        return False
    if audio_high_confidence is not True:
        return False
    return True


def pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.0"
    return f"{100.0 * part / total:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge snippet manifests into final_dataset_v2.jsonl.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print counts only (no output files).",
    )
    args = parser.parse_args()

    cfg = load_config()
    fd = cfg.get("final_dataset") or {}
    sources = cfg.get("sources") or {}

    conf_thr = float(fd.get("confidence_threshold", 0.65))
    margin_thr = float(fd.get("margin_threshold", 0.15))
    label_10_to_5: dict[str, str] = dict(fd.get("label_map_10_to_5") or {})
    binary_map: dict[str, str] = dict(fd.get("binary_map") or {})

    input_manifest = REPO_ROOT / fd["input_manifest"]
    assignments_csv = REPO_ROOT / fd["assignments_csv"]
    gpt_desc_path = REPO_ROOT / fd["gpt_descriptions_jsonl"]
    audio_pred_path = REPO_ROOT / fd["audio_predictions_jsonl"]
    out_manifest = REPO_ROOT / fd["output_manifest"]
    report_txt = REPO_ROOT / fd["merge_report_txt"]
    merge_log = REPO_ROOT / fd["merge_log_jsonl"]

    for p, name in [
        (input_manifest, "input_manifest"),
        (assignments_csv, "assignments_csv"),
        (gpt_desc_path, "gpt_descriptions_jsonl"),
        (audio_pred_path, "audio_predictions_jsonl"),
    ]:
        if not p.is_file():
            print(f"ERROR: missing {name}: {p}", file=sys.stderr)
            return 1

    snippets_dirs = [REPO_ROOT / x for x in sources.get("snippets_dirs", [])]

    print("Loading audio predictions…")
    audio_by_stem, audio_dup_stems = load_audio_predictions(audio_pred_path)
    print(f"✅ Loaded {len(audio_by_stem)} audio prediction rows ({audio_dup_stems} duplicate stems overwritten)")

    print("Loading cat_id assignments…")
    assign_df = pd.read_csv(assignments_csv)
    assign_df = assign_df.set_index("snippet_id", drop=False)
    print(f"✅ Loaded {len(assign_df)} assignment rows")

    print("Loading GPT descriptions aggregate…")
    gpt_agg = load_gpt_descriptions_aggregate(gpt_desc_path)
    print(
        f"✅ GPT descriptions: {gpt_agg['n_lines']} lines, "
        f"tokens≈{gpt_agg['input_tokens'] + gpt_agg['output_tokens']:.0f}, "
        f"cost_usd≈{gpt_agg['cost_usd']:.2f}"
    )

    # First pass: count snippets and optionally dry-run
    snippet_records: list[dict[str, Any]] = []
    n_videos = 0
    for rec in iter_manifest_records(input_manifest):
        n_videos += 1
        snippets = rec.get("snippets") or []
        for sn in snippets:
            if not isinstance(sn, dict):
                continue
            sid = sn.get("id")
            if not sid:
                continue
            snippet_records.append({"video": rec, "snippet": sn})

    print(f"✅ Parsed {len(snippet_records)} snippets from {n_videos} video records in metadata manifest")

    if args.dry_run:
        would_audio = sum(
            1 for r in snippet_records if Path(str(r["snippet"].get("id"))).name in audio_by_stem
        )
        print(
            f"Dry run: would join audio for {would_audio} / {len(snippet_records)} snippets; "
            f"outputs: {out_manifest.name}, {report_txt.name}, {merge_log.name}"
        )
        return 0

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    report_txt.parent.mkdir(parents=True, exist_ok=True)

    tmp_out = out_manifest.with_suffix(out_manifest.suffix + ".tmp")
    tmp_log = merge_log.with_suffix(merge_log.suffix + ".tmp")

    audio_hits = 0
    audio_miss = 0
    cat_hits = 0
    cat_miss = 0
    high_conf = 0
    low_conf = 0
    no_audio = 0

    # Unsuitable breakdown (non-exclusive)
    br_gpt_not = 0
    br_ai = 0
    br_cat = 0
    br_audio_low = 0
    br_audio_miss = 0
    br_multi = 0

    # Label distributions (high-conf only for final section)
    cnt_10_high: Counter[str] = Counter()
    cnt_5_high: Counter[str] = Counter()
    cnt_bin_high: Counter[str] = Counter()

    suitable_n = 0
    rows_written = 0

    def process_row(
        video: dict[str, Any],
        sn: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal audio_hits, audio_miss, cat_hits, cat_miss
        nonlocal high_conf, low_conf, no_audio
        nonlocal br_gpt_not, br_ai, br_cat, br_audio_low, br_audio_miss, br_multi
        nonlocal cnt_10_high, cnt_5_high, cnt_bin_high, suitable_n

        sid = str(sn["id"])
        gpt = sn.get("gpt_description")
        flat_gpt = flatten_gpt(gpt if isinstance(gpt, dict) else None)

        ts = sn.get("timestamp_range")
        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            start_sec = float(ts[0])
            end_sec = float(ts[1])
        else:
            start_sec = None
            end_sec = None
        dur = sn.get("duration")
        duration_sec = float(dur) if dur is not None else None

        abs_vid = resolve_snippet_video(REPO_ROOT, sn, snippets_dirs)
        video_path = relpath_under_repo(abs_vid, REPO_ROOT)

        ar = audio_by_stem.get(sid)
        audio_join_hit = ar is not None
        if audio_join_hit:
            audio_hits += 1
        else:
            audio_miss += 1

        audio_label_10 = ar.get("predicted_class") if ar else None
        audio_confidence = ar.get("confidence") if ar else None
        probs_flat: dict[str, float | None] = {k: None for k in PROB_FIELD_NAMES}
        margin = 0.0
        if ar:
            plist: list[float] = []
            for cn in CLASS_ORDER_10:
                key = f"prob_{cn}"
                v = ar.get(key)
                if v is not None:
                    plist.append(float(v))
                probs_flat[key] = float(v) if v is not None else None
            margin = margin_top1_top2(plist) if plist else 0.0
            conf_f = float(audio_confidence) if audio_confidence is not None else 0.0
            audio_high_confidence = conf_f >= conf_thr and margin >= margin_thr
        else:
            audio_high_confidence = None

        if not audio_join_hit:
            no_audio += 1
            label_source = "no_audio"
        elif audio_high_confidence:
            high_conf += 1
            label_source = "audio_high_conf"
        else:
            low_conf += 1
            label_source = "audio_low_conf"

        if audio_label_10 and isinstance(audio_label_10, str):
            audio_label_5 = label_10_to_5.get(audio_label_10)
        else:
            audio_label_5 = None

        if audio_label_5 and audio_label_5 in binary_map:
            audio_label_binary = binary_map[audio_label_5]
        else:
            audio_label_binary = None

        final_label_5 = audio_label_5 if audio_high_confidence else None
        final_label_binary = audio_label_binary if audio_high_confidence else None

        if audio_high_confidence and audio_label_10:
            cnt_10_high[str(audio_label_10)] += 1
        if final_label_5:
            cnt_5_high[str(final_label_5)] += 1
        if final_label_binary:
            cnt_bin_high[str(final_label_binary)] += 1

        # Cat CSV
        row_csv = assign_df.loc[sid] if sid in assign_df.index else None
        if row_csv is not None:
            cat_hits += 1
            cluster_size = row_csv.get("cluster_size")
            breed_guess = row_csv.get("breed_guess")
            coat_pattern = row_csv.get("coat_pattern")
            channel = row_csv.get("channel")
            platform = row_csv.get("platform")
        else:
            cat_miss += 1
            cluster_size = None
            breed_guess = None
            coat_pattern = None
            channel = None
            platform = None

        if platform is None or (isinstance(platform, float) and pd.isna(platform)):
            # fallback infer from video record
            src = (video.get("source") or "").strip().lower()
            if src == "tiktok_metadata":
                platform = "TikTok"
            else:
                platform = "YouTube"
            abs_p = abs_vid or ""
            lowp = abs_p.lower()
            if "tiktok_snippets" in lowp or "/tiktok/" in lowp:
                platform = "TikTok"
            elif "dailymotion" in lowp:
                platform = "DailyMotion"

        if channel is None or (isinstance(channel, float) and pd.isna(channel)):
            channel = video.get("video_id")

        cat_id = sn.get("cat_id")
        cat_id_method = sn.get("cat_id_method")

        audio_relp = None
        if ar and isinstance(ar.get("relpath"), str):
            audio_relp = ar["relpath"]
        elif audio_join_hit and ar:
            fp = ar.get("filepath")
            if fp:
                audio_relp = relpath_under_repo(str(fp), REPO_ROOT)

        primary_reason, failed_list = compute_unsuitable_reason_and_flags(
            gpt if isinstance(gpt, dict) else None,
            str(cat_id_method) if cat_id_method is not None else None,
            audio_join_hit,
            audio_high_confidence,
        )
        suitable = suitable_from_flags(
            gpt if isinstance(gpt, dict) else None,
            str(cat_id_method) if cat_id_method is not None else None,
            audio_high_confidence,
        )
        if suitable:
            suitable_n += 1
            unsuitable_reason = None
        else:
            unsuitable_reason = primary_reason

        # Non-exclusive breakdown
        if not suitable:
            if "gpt_not_suitable" in failed_list:
                br_gpt_not += 1
            if "gpt_ai_generated" in failed_list:
                br_ai += 1
            if "cat_id_blocked" in failed_list:
                br_cat += 1
            if "audio_low_confidence" in failed_list:
                br_audio_low += 1
            if "audio_missing" in failed_list:
                br_audio_miss += 1
            if len(failed_list) > 1:
                br_multi += 1

        out: dict[str, Any] = {
            "snippet_id": sid,
            "video_id": video.get("video_id"),
            "platform": platform,
            "channel": channel,
            "behavioral_category": video.get("behavioral_category"),
            "duration_sec": duration_sec,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "video_path": video_path,
            "audio_path": audio_relp,
            "audio_label_10": audio_label_10,
            "audio_label_5": audio_label_5,
            "audio_label_binary": audio_label_binary,
            "audio_confidence": float(audio_confidence) if audio_confidence is not None else None,
            "audio_high_confidence": audio_high_confidence,
            "label_source": label_source,
            "final_label_5": final_label_5,
            "final_label_binary": final_label_binary,
            "cat_id": cat_id,
            "cat_id_method": cat_id_method,
            "cluster_size": int(cluster_size) if cluster_size is not None and not pd.isna(cluster_size) else None,
            "bbox_found": sn.get("bbox_found"),
            "clip_similarity_to_nearest": sn.get("clip_similarity_to_nearest"),
            "gpt_verified": sn.get("gpt_verified"),
            **flat_gpt,
            "suitable_for_training": suitable,
            "unsuitable_reason": unsuitable_reason,
        }
        out.update(probs_flat)

        log_line = {
            "snippet_id": sid,
            "audio_join": "hit" if audio_join_hit else "miss",
            "cat_join": "hit" if row_csv is not None else "miss",
            "final_label_5": final_label_5,
            "suitable_for_training": suitable,
            "unsuitable_reason": unsuitable_reason,
        }
        return out, log_line

    # Build ordered output objects
    ordered_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    for item in snippet_records:
        o, lg = process_row(item["video"], item["snippet"])
        ordered = {k: o.get(k) for k in OUTPUT_ROW_KEYS}
        ordered_rows.append(ordered)
        log_rows.append(lg)

    print("✅ Derived labels and join stats in memory")

    n_snip = len(snippet_records)
    labeled_any = audio_hits

    try:
        with open(tmp_out, "w", encoding="utf-8") as fo, open(tmp_log, "w", encoding="utf-8") as fl:
            for ordered, lg in zip(ordered_rows, log_rows, strict=True):
                fo.write(json.dumps(ordered, ensure_ascii=False) + "\n")
                fl.write(json.dumps(lg, ensure_ascii=False) + "\n")
                rows_written += 1
    except KeyboardInterrupt:
        print(f"\nInterrupted after writing {rows_written} rows (partial files not finalized).")
        if tmp_out.is_file():
            print(f"Partial output may exist at: {tmp_out}")
        return 130

    os.replace(tmp_out, out_manifest)
    os.replace(tmp_log, merge_log)
    print(f"✅ Wrote {rows_written} rows to {out_manifest.relative_to(REPO_ROOT)}")
    print(f"✅ Wrote merge log to {merge_log.relative_to(REPO_ROOT)}")

    # Report
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("════════════════════════════════════════════════════════")
    lines.append("FINAL DATASET MERGE REPORT")
    lines.append(f"Generated: {now}")
    lines.append("════════════════════════════════════════════════════════")
    lines.append("")
    lines.append("INPUT SOURCES")
    lines.append("─────────────")
    lines.append(f"Snippets from metadata_clean_03:    {n_snip}")
    lines.append(
        f"Audio predictions joined:           {audio_hits}  ({pct(audio_hits, n_snip)}%)  — {audio_miss} missed"
    )
    lines.append(f"Cat ID assignments joined:          {cat_hits}  ({pct(cat_hits, n_snip)}%)")
    lines.append("")
    lines.append("LABEL PIPELINE")
    lines.append("──────────────")
    lines.append(f"Audio labels available:             {labeled_any}")
    lines.append(
        f"High-confidence labels (≥{conf_thr}, margin≥{margin_thr}):"
    )
    lines.append(f"                                    {high_conf}  ({pct(high_conf, labeled_any)} of labeled)")
    lines.append(f"Low-confidence labels:              {low_conf}  ({pct(low_conf, n_snip)}%)")
    lines.append(f"No audio label:                     {no_audio}  ({pct(no_audio, n_snip)}%)")
    lines.append("")
    lines.append("FINAL LABEL DISTRIBUTION (high-confidence only)")
    lines.append("────────────────────────────────────────────────")
    lines.append("10-class:")
    for cls in CLASS_ORDER_10:
        c = cnt_10_high.get(cls, 0)
        lines.append(f"  {cls:12} | {c:5} | {pct(c, high_conf)}%")
    lines.append("")
    lines.append("5-class merged:")
    for name in ["Paining", "Positive_Baseline", "Agonistic", "Vocalizing", "HuntingMind"]:
        c = cnt_5_high.get(name, 0)
        lines.append(f"  {name:18} {c:5}  ({pct(c, high_conf)}%)")
    lines.append("")
    lines.append("Binary:")
    pain_c = cnt_bin_high.get("Pain", 0)
    nop_c = cnt_bin_high.get("No_Pain", 0)
    ratio = f"1:{max(1, nop_c) // max(1, pain_c)}" if pain_c else "n/a"
    lines.append(f"  Pain:    {pain_c:5}  ({pct(pain_c, high_conf)}%)")
    lines.append(f"  No_Pain: {nop_c:5}  ({pct(nop_c, high_conf)}%)")
    lines.append(f"  Ratio:   {ratio}")
    lines.append("")
    lines.append("SUITABILITY FOR TRAINING")
    lines.append("────────────────────────")
    lines.append(f"Suitable (all conditions met):   {suitable_n}  ({pct(suitable_n, n_snip)}%)")
    lines.append("Unsuitable — breakdown:")
    lines.append(f"  GPT: not suitable:             {br_gpt_not}")
    lines.append(f"  GPT: AI generated:             {br_ai}")
    lines.append(f"  Cat ID: no cat / unsuitable:   {br_cat}")
    lines.append(f"  Audio: low confidence:         {br_audio_low}")
    lines.append(f"  Audio: missing:                {br_audio_miss}")
    lines.append(f"  Multiple conditions:           {br_multi}")
    lines.append("")
    lines.append("OUTPUT")
    lines.append("──────")
    lines.append(f"final_dataset_v2.jsonl:  {rows_written} rows")
    lines.append(f"Training-ready rows:     {suitable_n} rows")
    lines.append("")
    lines.append("GPT descriptions file (aggregate)")
    lines.append(f"  Lines: {gpt_agg['n_lines']}, cost USD (sum): {gpt_agg['cost_usd']:.4f}")
    lines.append("════════════════════════════════════════════════════════")

    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ Wrote report to {report_txt.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
