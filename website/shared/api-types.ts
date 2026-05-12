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
  progress: JobProgress;
  error_type?: string | null;
  error_message?: string | null;
}

export interface ClipAudioProbe {
  duration_sec?: number;
  rms?: number;
  likely_silent?: boolean;
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
}

export interface NormalizedResult {
  summary: Record<string, unknown>;
  windows: NormalizedWindowRow[];
  artifacts: Record<string, string[]>;
  raw: Record<string, unknown>;
  final_decision?: unknown;
  p_pain_max?: unknown;
  p_pain_mean?: unknown;
  meta?: Record<string, unknown>;
}
