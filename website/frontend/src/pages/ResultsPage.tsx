import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import type { NormalizedResult, NormalizedWindowRow } from "../../../shared/api-types";
import { artifactUrl, fetchArtifacts, fetchResult } from "../api";

function pad3(n: unknown): string {
  const x = typeof n === "number" ? n : parseInt(String(n), 10);
  if (Number.isNaN(x)) return "000";
  return String(x).padStart(3, "0");
}

function toProbMap(value: unknown): Record<string, number> {
  if (!value || typeof value !== "object") return {};
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    const n = Number(v);
    if (!Number.isNaN(n)) out[k] = n;
  }
  return out;
}

function decisionBadge(decision: string): string {
  if (decision === "pain") return "bg-red-500/15 border border-red-400/40 text-red-700 dark:text-red-300";
  if (decision === "non_pain") return "bg-emerald-500/15 border border-emerald-400/40 text-emerald-700 dark:text-emerald-300";
  return "bg-slate-500/10 border border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300";
}

function branchBadge(branch: string): string {
  return branch === "audio"
    ? "bg-sky-500/15 border border-sky-400/40 text-sky-700 dark:text-sky-300"
    : "bg-violet-500/15 border border-violet-400/40 text-violet-700 dark:text-violet-300";
}

function formatPairwiseLabel(label: string): string {
  const parts = label.split("|");
  if (parts.length !== 2) return label;
  const [left, right] = parts;
  if (left === "Paining") return `P(Paining over ${right})`;
  if (right === "Paining") return `P(Paining over ${left})`;
  return `P(${left} over ${right})`;
}

function metaClassLabelClass(label: string): string {
  const low = label.toLowerCase();
  if (low.includes("non_pain")) {
    return "text-emerald-700 dark:text-emerald-300";
  }
  if (low.includes("pain")) {
    return "text-rose-700 dark:text-rose-300";
  }
  return "";
}

function metaWeightPillClass(w: number): string {
  const a = Math.abs(w);
  if (a < 0.05) {
    return "bg-slate-500/15 text-slate-700 dark:text-slate-300 border border-slate-400/35";
  }
  if (w > 0) {
    return "bg-rose-500/20 text-rose-800 dark:text-rose-200 border border-rose-400/45";
  }
  return "bg-emerald-500/15 text-emerald-800 dark:text-emerald-200 border border-emerald-400/40";
}

function toFriendlyMetaClassProbabilities(
  probs: Record<string, number>,
  pPain: number | null,
): Record<string, number> {
  const entries = Object.entries(probs);
  if (entries.length === 0) return probs;

  let painClassKey: string | null =
    entries.find(([k]) => k.toLowerCase().includes("pain"))?.[0] ?? null;

  // Some meta models expose numeric classes (e.g. 0/1). In that case infer which
  // one is "pain" by matching the class probability closest to window p_pain.
  if (!painClassKey && pPain != null) {
    let best: { key: string; dist: number } | null = null;
    for (const [k, v] of entries) {
      const dist = Math.abs(v - pPain);
      if (!best || dist < best.dist) best = { key: k, dist };
    }
    painClassKey = best?.key ?? null;
  }

  const out: Record<string, number> = {};
  const isBinary = entries.length === 2;
  for (const [k, v] of entries) {
    const friendly =
      painClassKey && k === painClassKey
        ? `pain (class ${k})`
        : isBinary && painClassKey
          ? `non_pain (class ${k})`
          : `class ${k}`;
    out[friendly] = v;
  }
  return out;
}

function ProbabilityBars({
  title,
  probs,
  colorClass,
  limit = 999,
  formatLabel = (label: string) => label,
  wrapLabels = false,
  formatRightMeta = () => null,
  accentRightColumn = false,
  rightPctClassName,
}: {
  title: string;
  probs: Record<string, number>;
  colorClass: string;
  limit?: number;
  formatLabel?: (label: string, p: number) => ReactNode;
  wrapLabels?: boolean;
  formatRightMeta?: (label: string, p: number) => ReactNode;
  accentRightColumn?: boolean;
  rightPctClassName?: (label: string, p: number) => string;
}) {
  const [animateIn, setAnimateIn] = useState(false);
  const entries = Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit);
  if (entries.length === 0) return null;

  useEffect(() => {
    setAnimateIn(false);
    const id = requestAnimationFrame(() => setAnimateIn(true));
    return () => cancelAnimationFrame(id);
  }, [title, probs, limit]);

  return (
    <div className="space-y-2">
      <h4 className="text-[11px] uppercase tracking-[0.16em] text-slate-600 dark:text-slate-500">{title}</h4>
      <div className="space-y-1.5">
        {entries.map(([label, p]) => {
          const pct = Math.max(0, Math.min(100, p * 100));
          const rightMeta = formatRightMeta(label, p);
          const pctExtra = rightPctClassName?.(label, p) ?? "";
          const rightColTone =
            pctExtra ||
            (accentRightColumn ? "text-violet-700 dark:text-violet-300" : "text-slate-600 dark:text-slate-400");
          return (
            <div key={label} className="grid grid-cols-[1fr_minmax(4.75rem,auto)] items-center gap-2">
              <div className="min-w-0">
                <div className="flex items-center justify-between text-[11px] text-slate-600 dark:text-slate-400 leading-none">
                  <span className={wrapLabels ? "whitespace-normal break-words pr-2 leading-snug" : "truncate"}>
                    {formatLabel(label, p)}
                  </span>
                  <span
                    className={`font-mono text-slate-700 dark:text-slate-300 tabular-nums transition-opacity duration-500 ${
                      animateIn ? "opacity-100" : "opacity-0"
                    }`}
                  >
                    {p.toFixed(4)}
                  </span>
                </div>
                <div className="mt-1 h-1.5 bg-slate-200 dark:bg-slate-800/95 rounded overflow-hidden">
                  <div
                    className={`h-full ${colorClass} transition-[width] duration-700 ease-out`}
                    style={{ width: animateIn ? `${pct}%` : "0%" }}
                  />
                </div>
              </div>
              <div
                className={`text-[10px] font-mono text-right tabular-nums leading-tight min-w-[4.5rem] ${rightColTone}`}
              >
                <div className="font-medium">{pct.toFixed(1)}%</div>
                {rightMeta != null && rightMeta !== "" && (
                  <div className="mt-0.5 flex justify-end">{rightMeta}</div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ClipPainOverview({ windows }: { windows: NormalizedWindowRow[] }) {
  const [animateIn, setAnimateIn] = useState(false);
  useEffect(() => {
    setAnimateIn(false);
    const id = requestAnimationFrame(() => setAnimateIn(true));
    return () => cancelAnimationFrame(id);
  }, [windows]);

  return (
    <section className="rounded-2xl border border-slate-200 dark:border-slate-700/70 bg-white dark:bg-slate-900/80 p-5 shadow-[0_0_0_1px_rgba(15,23,42,0.08)] dark:shadow-[0_0_0_1px_rgba(15,23,42,0.4)]">
      <div className="text-[10px] uppercase tracking-[0.16em] text-slate-600 dark:text-slate-500 mb-3">
        Clip overview · P(pain) by clip
      </div>
      <p className="text-[11px] text-slate-600 dark:text-slate-500 mb-4">
        Bar length is per-window pain probability. Color indicates routing: audio branch (YAMNet P(cat) ≥ threshold)
        vs video branch. Low RMS / “silent” refers to waveform energy in the extracted clip, not YAMNet.
      </p>
      <div className="flex flex-wrap gap-5 mb-4 text-[11px] text-slate-600 dark:text-slate-400">
        <span className="inline-flex items-center gap-2">
          <span className="h-2.5 w-6 rounded-sm bg-sky-400" aria-hidden />
          Audio branch
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-2.5 w-6 rounded-sm bg-violet-400" aria-hidden />
          Video branch
        </span>
      </div>
      <div className="space-y-3">
        {windows.map((w, i) => {
          const branch = String(w.branch ?? "");
          const pPain = w.p_pain != null ? Number(w.p_pain) : null;
          const start = w.start_sec != null ? Number(w.start_sec) : null;
          const end = w.end_sec != null ? Number(w.end_sec) : null;
          const idx = w.window_index != null ? Number(w.window_index) : i;
          const label =
            start != null && end != null && !Number.isNaN(start) && !Number.isNaN(end)
              ? `#${idx + 1} [${start.toFixed(2)}–${end.toFixed(2)}]s`
              : `#${idx + 1}`;
          const pct =
            pPain == null || Number.isNaN(pPain) ? null : Math.max(0, Math.min(100, pPain * 100));
          const barColor =
            branch === "audio" ? "bg-sky-400" : branch === "video" ? "bg-violet-400" : "bg-slate-400";
          const probe = w.clip_audio_probe;
          const silent = probe?.likely_silent === true;
          const pCat = w.p_cat != null ? Number(w.p_cat) : null;
          return (
            <div key={i} className="grid grid-cols-[minmax(7rem,1fr)_minmax(0,4fr)_auto] gap-3 items-center">
              <div className="min-w-0">
                <div className="text-[11px] font-medium text-slate-800 dark:text-slate-200 truncate">{label}</div>
                <div className="text-[10px] space-x-2 text-slate-500 dark:text-slate-500 mt-0.5">
                  <span className={`text-[10px] px-1.5 py-0.5 rounded ${branchBadge(branch)}`}>{branch}</span>
                  {pCat != null && !Number.isNaN(pCat) && (
                    <span className="font-mono tabular-nums">P(cat)={pCat.toFixed(3)}</span>
                  )}
                  {silent && (
                    <span className="text-amber-700 dark:text-amber-300/90" title="Low waveform RMS on extracted WAV">
                      low audio level
                    </span>
                  )}
                </div>
              </div>
              <div className="min-w-0 h-2.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                <div
                  className={`h-full ${barColor} transition-[width] duration-700 ease-out`}
                  style={{ width: animateIn ? `${pct ?? 0}%` : "0%" }}
                />
              </div>
              <div className="text-[11px] font-mono text-slate-700 dark:text-slate-300 tabular-nums text-right min-w-[3.5rem]">
                {pPain != null && !Number.isNaN(pPain) ? pPain.toFixed(4) : "—"}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default function ResultsPage() {
  const { id = "" } = useParams();
  const [res, setRes] = useState<NormalizedResult | null>(null);
  const [arts, setArts] = useState<Record<string, string[]> | null>(null);
  const [showRaw, setShowRaw] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    async function load() {
      try {
        const [r, a] = await Promise.all([fetchResult(id), fetchArtifacts(id)]);
        if (!stop) {
          setRes(r);
          setArts(a);
        }
      } catch (e) {
        if (!stop) setErr(e instanceof Error ? e.message : String(e));
      }
    }
    void load();
    return () => {
      stop = true;
    };
  }, [id]);

  if (err) return <p className="text-red-500 dark:text-red-400">{err}</p>;
  if (!res) return <p className="text-slate-600 dark:text-slate-400">Loading results…</p>;

  const summary = res.summary as Record<string, unknown>;
  const split = summary.split as Record<string, unknown> | undefined;
  const timing = (res.raw as Record<string, unknown>).timing_seconds as
    | Record<string, unknown>
    | undefined;
  const totalSec = timing?.total;

  const finalDecision = String(res.final_decision ?? "—");
  const pPainMax = res.p_pain_max != null ? Number(res.p_pain_max) : null;
  const pPainMean = res.p_pain_mean != null ? Number(res.p_pain_mean) : null;
  const confidence =
    pPainMax == null ? null : finalDecision === "non_pain" ? 1 - pPainMax : pPainMax;
  const confidencePct = confidence == null ? null : Math.max(0, Math.min(100, confidence * 100));
  const riskLabel =
    confidencePct == null ? "unknown" : confidencePct >= 75 ? "low risk" : confidencePct >= 50 ? "medium risk" : "high risk";

  const windows = (res.windows ?? []) as NormalizedWindowRow[];
  const isSplit = (res.raw as Record<string, unknown>)?.mode === "split_sliding_windows";
  const showClipOverview = windows.length > 1 || isSplit;

  return (
    <div className="space-y-6">
      <section className="text-[11px] tracking-[0.16em] uppercase text-slate-600 dark:text-slate-500">A · Summary</section>
      <section className="rounded-2xl border border-slate-200 dark:border-slate-700/70 bg-white dark:bg-slate-900/80 p-5 shadow-[0_0_0_1px_rgba(15,23,42,0.08)] dark:shadow-[0_0_0_1px_rgba(15,23,42,0.4)]">
        <div className="grid md:grid-cols-[2fr_1fr_1fr] gap-5">
          <div className="border-r border-slate-200 dark:border-slate-800 pr-4">
            <div className="text-[10px] uppercase tracking-[0.16em] text-slate-600 dark:text-slate-500 mb-2">Final verdict</div>
            <div className="flex items-center gap-3">
              <div className={`text-4xl font-semibold ${finalDecision === "pain" ? "text-red-300" : "text-emerald-300"}`}>
                {finalDecision}
              </div>
              <span className={`text-[11px] px-3 py-1 rounded-full ${decisionBadge(finalDecision)}`}>
                {riskLabel}
              </span>
            </div>
            <div className="mt-3 text-[11px] text-slate-600 dark:text-slate-500">
              Pain confidence — {confidencePct != null ? `${confidencePct.toFixed(1)}%` : "—"}
            </div>
            <div className="mt-1 h-1.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden max-w-56">
              <div className="h-full bg-emerald-400" style={{ width: `${confidencePct ?? 0}%` }} />
            </div>
          </div>

          <div>
            <div className="text-[10px] uppercase tracking-[0.16em] text-slate-600 dark:text-slate-500">Max P(pain)</div>
            <div className="mt-2 text-3xl font-semibold text-emerald-600 dark:text-emerald-300 font-mono tabular-nums">
              {pPainMax != null ? pPainMax.toFixed(4) : "—"}
            </div>
          </div>

          <div>
            <div className="text-[10px] uppercase tracking-[0.16em] text-slate-600 dark:text-slate-500">Mean P(pain)</div>
            <div className="mt-2 text-3xl font-semibold text-emerald-600 dark:text-emerald-300 font-mono tabular-nums">
              {pPainMean != null ? pPainMean.toFixed(4) : "—"}
            </div>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-4 text-[11px] text-slate-600 dark:text-slate-500">
          <span>
            Total runtime: <span className="text-slate-700 dark:text-slate-300 font-mono">{totalSec != null ? `${Number(totalSec).toFixed(2)}s` : "—"}</span>
          </span>
          {split && (
            <span>
              split windows: <span className="text-slate-700 dark:text-slate-300 font-mono">{JSON.stringify(split)}</span>
            </span>
          )}
        </div>
      </section>

      {showClipOverview && <ClipPainOverview windows={windows} />}

      <section className="text-[11px] tracking-[0.16em] uppercase text-slate-600 dark:text-slate-500">B · Timeline</section>
      <section className="overflow-x-auto rounded-2xl border border-slate-200 dark:border-slate-700/70 bg-white dark:bg-slate-900/80">
        <table className="w-full text-sm">
          <thead className="text-left text-slate-600 dark:text-slate-500 uppercase text-[10px] tracking-[0.16em]">
            <tr className="border-b border-slate-200 dark:border-slate-800">
              <th className="px-3 py-2">#</th>
              <th className="px-3 py-2">start</th>
              <th className="px-3 py-2">end</th>
              <th className="px-3 py-2">branch</th>
              <th className="px-3 py-2">p_cat</th>
              <th className="px-3 py-2">p_pain</th>
              <th className="px-3 py-2">clip audio</th>
              <th className="px-3 py-2">decision</th>
            </tr>
          </thead>
          <tbody>
            {windows.map((w, i) => {
              const decision = String(w.decision ?? "—");
              const branch = String(w.branch ?? "—");
              return (
                <tr key={i} className="border-t border-slate-200 dark:border-slate-800/70 hover:bg-slate-100 dark:hover:bg-slate-800/35">
                  <td className="px-3 py-2 font-mono text-slate-500 dark:text-slate-500 tabular-nums">
                    {w.window_index != null ? Number(w.window_index) + 1 : i + 1}
                  </td>
                  <td className="px-3 py-2 font-mono tabular-nums">
                    {w.start_sec != null && !Number.isNaN(Number(w.start_sec))
                      ? Number(w.start_sec).toFixed(2)
                      : "—"}
                  </td>
                  <td className="px-3 py-2 font-mono tabular-nums">
                    {w.end_sec != null && !Number.isNaN(Number(w.end_sec))
                      ? Number(w.end_sec).toFixed(2)
                      : "—"}
                  </td>
                  <td className="px-3 py-2">
                    <span className={`text-[11px] px-2 py-0.5 rounded-full ${branchBadge(branch)}`}>
                      {branch}
                    </span>
                  </td>
                  <td className="px-3 py-2 font-mono tabular-nums">
                    {w.p_cat != null ? Number(w.p_cat).toFixed(4) : "—"}
                  </td>
                  <td className="px-3 py-2 font-mono text-emerald-600 dark:text-emerald-300 tabular-nums">
                    {w.p_pain != null ? Number(w.p_pain).toFixed(4) : "—"}
                  </td>
                  <td className="px-3 py-2 text-[11px] text-slate-600 dark:text-slate-400">
                    {w.clip_audio_probe != null ? (
                      <div className="space-y-0.5">
                        <div className="font-mono tabular-nums">
                          rms {typeof w.clip_audio_probe.rms === "number" ? w.clip_audio_probe.rms.toExponential(2) : "—"}
                        </div>
                        {w.clip_audio_probe.likely_silent === true && (
                          <span className="text-amber-700 dark:text-amber-300/90">low level</span>
                        )}
                      </div>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className={`text-[11px] px-2 py-0.5 rounded-full ${decisionBadge(decision)}`}>
                        {decision}
                      </span>
                      {branch === "audio" && typeof w.audio_emotion_label === "string" && w.audio_emotion_label && (
                        <span className="text-[11px] px-2 py-0.5 rounded-full bg-sky-500/10 border border-sky-400/30 text-sky-700 dark:text-sky-200">
                          {w.audio_emotion_label}
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <section className="text-[11px] tracking-[0.16em] uppercase text-slate-600 dark:text-slate-500">C · Evidence & Probabilities</section>
      <section className="space-y-4">
        {windows.map((w, i) => {
          const branch = String(w.branch ?? "");
          const artsRow = (w.artifacts ?? {}) as Record<string, unknown>;
          const sep = artsRow.separated_audio as string | undefined;
          const orig = artsRow.original_video as string | undefined;
          const probsRow = (w.probabilities ?? {}) as Record<string, unknown>;
          const audioSoftmax = toProbMap(probsRow.audio_softmax);
          const videoPairwise = toProbMap(probsRow.video_pairwise);
          let videoPairwiseWeights = toProbMap(probsRow.video_pairwise_weights);
          if (branch === "video" && Object.keys(videoPairwiseWeights).length === 0) {
            // Fallback for older/stale normalized responses: recover weights from raw.
            const rawRoot = (res.raw ?? {}) as Record<string, unknown>;
            const rawWindows = (rawRoot.windows ?? []) as Array<Record<string, unknown>>;
            const rawWindow = rawWindows[i] ?? {};
            const rawResult = (rawWindow.result ?? {}) as Record<string, unknown>;
            const rawMeta = (rawResult.meta_result ?? {}) as Record<string, unknown>;
            videoPairwiseWeights = toProbMap(rawMeta.meta_feature_weights);
          }
          const videoMetaRaw = toProbMap(probsRow.video_meta_class_probs);
          const pPainWindow = w.p_pain != null ? Number(w.p_pain) : null;
          const videoMeta = toFriendlyMetaClassProbabilities(videoMetaRaw, pPainWindow);

          const wi = w.window_index ?? i;
          const relPose = isSplit ? `window_${pad3(wi)}/pose_video.mp4` : "pose_video.mp4";
          const relAudio = isSplit ? `window_${pad3(wi)}/separated_audio.wav` : "separated_audio.wav";

          return (
            <div key={i} className="rounded-2xl border border-slate-200 dark:border-slate-700/70 bg-white dark:bg-slate-900/80 p-4 space-y-4">
              <div className="flex items-center gap-2 text-[11px]">
                <span className="px-2 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 font-medium">
                  window {Number(wi) + 1}
                </span>
                <span className={`px-2 py-0.5 rounded-full ${branchBadge(branch)}`}>{branch}</span>
              </div>

              {branch === "video" && (
                <div>
                  <div className="text-[11px] text-slate-600 dark:text-slate-500 mb-1">pose video</div>
                  <video
                    src={artifactUrl(id, relPose)}
                    controls
                    className="w-full max-w-lg rounded-lg border border-slate-300 dark:border-slate-700"
                  />
                </div>
              )}
              {branch === "audio" && sep && (
                <div>
                  <div className="text-[11px] text-slate-600 dark:text-slate-500 mb-1">separated audio</div>
                  <audio controls src={artifactUrl(id, relAudio)} className="w-full" />
                </div>
              )}

              {orig && <div className="text-[11px] text-slate-500 dark:text-slate-600 truncate">original: {orig}</div>}

              {branch === "audio" && (
                <ProbabilityBars
                  title={`audio class probabilities (${Object.keys(audioSoftmax).length} classes)`}
                  probs={audioSoftmax}
                  colorClass="bg-sky-400"
                  limit={10}
                />
              )}
              {branch === "video" && (
                <div className="grid md:grid-cols-2 gap-5">
                  <ProbabilityBars
                    title="pairwise sub-model probs (each is P(Paining))"
                    probs={videoPairwise}
                    colorClass="bg-violet-400"
                    formatLabel={(label) => formatPairwiseLabel(label)}
                    wrapLabels
                    accentRightColumn
                    formatRightMeta={(label) => {
                      const w = videoPairwiseWeights[label];
                      if (w == null || Number.isNaN(w)) return null;
                      const sign = w >= 0 ? "+" : "";
                      return (
                        <span
                          className={`inline-block rounded px-1.5 py-0.5 font-mono text-[10px] tabular-nums ${metaWeightPillClass(w)}`}
                          title="Meta-model coefficient for this pairwise feature (sign: pushes toward pain vs non-pain log-odds)"
                        >
                          w={sign}
                          {w.toFixed(3)}
                        </span>
                      );
                    }}
                  />
                  <ProbabilityBars
                    title="meta class probs (final classifier)"
                    probs={videoMeta}
                    colorClass="bg-emerald-400"
                    formatLabel={(label) => (
                      <span className={metaClassLabelClass(label)}>{label}</span>
                    )}
                    wrapLabels
                    rightPctClassName={(label) => {
                      const low = label.toLowerCase();
                      if (low.includes("non_pain")) {
                        return "text-emerald-700 dark:text-emerald-300 font-semibold";
                      }
                      if (low.includes("pain")) {
                        return "text-rose-700 dark:text-rose-300 font-semibold";
                      }
                      return "";
                    }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </section>

      {arts && (
        <>
          <section className="text-[11px] tracking-[0.16em] uppercase text-slate-600 dark:text-slate-500">Artifacts</section>
          <section className="grid md:grid-cols-2 gap-3 text-xs font-mono">
            {(["video", "audio", "json", "other"] as const).map((k) => (
              <div key={k} className="rounded-xl border border-slate-200 dark:border-slate-700/70 bg-white dark:bg-slate-900/80 p-3">
                <div className="text-slate-600 dark:text-slate-500 mb-1 uppercase tracking-wider text-[10px]">{k} files</div>
                <ul className="max-h-36 overflow-y-auto text-slate-700 dark:text-slate-300 space-y-0.5">
                  {(arts[k] ?? []).slice(0, 40).map((p) => (
                    <li key={p}>
                      <a
                        className="hover:underline text-sky-700 dark:text-sky-300"
                        href={artifactUrl(id, p)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {p}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </section>
        </>
      )}

      <section>
        <button
          type="button"
          onClick={() => setShowRaw(!showRaw)}
          className="text-sm text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200"
        >
          D · Raw pipeline JSON {showRaw ? "▼" : "▶"}
        </button>
        {showRaw && (
          <pre className="mt-2 p-3 text-xs bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded max-h-96 overflow-auto text-slate-700 dark:text-slate-500">
            {JSON.stringify(res.raw, null, 2)}
          </pre>
        )}
      </section>
    </div>
  );
}
