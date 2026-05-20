/** API response shapes (subset; full `raw` is pipeline JSON). */
export type JobStatus = "queued" | "running" | "done" | "failed";

export interface JobProgress {
  stage?: string;
  window_idx?: number;
  n_windows?: number | null;
  percent?: number;
  events?: Array<Record<string, unknown>>;
}

export interface JobResponse {
  id: string;
  status: JobStatus;
  /** Echo of POST form / job queue parameters (e.g. multicat_video_only). */
  params?: Record<string, unknown>;
  progress: JobProgress;
  error_type?: string | null;
  error_message?: string | null;
}

export interface ClipAudioProbe {
  duration_sec?: number;
  rms?: number;
  likely_silent?: boolean;
}

/** Per-cat row (multicat video mode, window-level or legacy). */
export interface NormalizedCatRow {
  local_track_id?: number;
  window_index?: number | null;
  detection_rate_sampled?: number;
  /** same as detection_rate_sampled */
  track_coverage?: number;
  n_detected_frames?: number;
  p_pain?: number | null;
  decision?: string;
  artifacts?: Record<string, unknown>;
  probabilities?: Record<string, unknown>;
  pose_video_url?: string | null;
}

/** Top-level peer result for multicat single-video runs (normalize API). */
export interface NormalizedCatResult extends NormalizedCatRow {
  cat_index?: number;
  track_id?: number | null;
}

/** Row in ``normalized.windows`` (split or single). */
export interface NormalizedWindowRow {
  window_index?: number;
  start_sec?: number;
  end_sec?: number;
  branch?: string;
  p_cat?: number;
  p_pain?: number | null;
  decision?: string;
  audio_emotion_label?: string;
  clip_audio_probe?: ClipAudioProbe | null;
  clip_video?: string;
  window_run_dir?: string;
  artifacts?: Record<string, unknown>;
  probabilities?: Record<string, unknown>;
  cats?: NormalizedCatRow[] | null;
  multicat_headline?: Record<string, unknown> | null;
  multicat_empty_reason?: string | null;
  multicat_prevalence_fraction?: number | null;
  multicat_cats_above_threshold?: number | null;
  multicat_cats_total?: number | null;
  multicat_prevalence_label?: string | null;
}

export interface NormalizedResult {
  summary: Record<string, unknown>;
  windows: NormalizedWindowRow[];
  artifacts: Record<string, string[]>;
  raw: Record<string, unknown>;
  /** Present when ``multicat_params.multicat_video_only`` on a single-clip (non-split) run. */
  multicat_video_only?: boolean;
  multicat_cat_count?: number | null;
  cats?: NormalizedCatResult[];
  final_decision?: unknown;
  p_pain_max?: unknown;
  p_pain_mean?: unknown;
  p_pain_headline?: unknown;
  meta?: Record<string, unknown>;
}
