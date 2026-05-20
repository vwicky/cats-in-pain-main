import { Fragment, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import type {
  NormalizedCatResult,
  NormalizedCatRow,
  NormalizedResult,
  NormalizedWindowRow,
} from "../../../shared/api-types";
import { artifactUrl, fetchArtifacts, fetchResult } from "../api";
import { CatResultPanel, ProbabilityBars } from "../components/CatResultPanel";

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
  const d = decision.toLowerCase();
  if (d === "pain") return "bg-red-500/15 border border-red-400/40 text-red-700 dark:text-red-300";
  if (d === "non_pain") return "bg-emerald-500/15 border border-emerald-400/40 text-emerald-700 dark:text-emerald-300";
  return "bg-slate-500/10 border border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300";
}

/** Human-readable verdict for UI (backend still uses pain / non_pain). */
function decisionLabelForDisplay(decision: string): string {
  const d = decision.toLowerCase();
  if (d === "pain") return "Pain";
  if (d === "non_pain") return "No pain";
  return decision;
}

function formatSplitSummary(split: Record<string, unknown> | undefined): string | null {
  if (!split || typeof split !== "object") return null;
  const parts: string[] = [];
  for (const [k, v] of Object.entries(split)) {
    if (v === undefined || v === null) continue;
    const label = k.replace(/_/g, " ");
    parts.push(`${label}: ${String(v)}`);
  }
  return parts.length ? parts.join(" · ") : null;
}

/** Left accent for per-cat verdict rows (Section A). */
function verdictStripeClass(decision: string): string {
  const d = decision.toLowerCase();
  if (d === "pain") return "border-l-rose-500 dark:border-l-rose-400";
  if (d === "non_pain") return "border-l-emerald-500 dark:border-l-emerald-400";
  return "border-l-slate-400 dark:border-l-slate-500";
}

/** Fill for Summary per-cat P(pain) micro-bar. */
function pPainBarFillClass(decision: string): string {
  const d = decision.toLowerCase();
  if (d === "pain") return "bg-rose-400 dark:bg-rose-500";
  if (d === "non_pain") return "bg-emerald-400 dark:bg-emerald-500";
  return "bg-slate-400 dark:bg-slate-500";
}

function SectionLabel({
  id,
  title,
  kicker,
}: {
  id: string;
  title: string;
  /** Short line under the title; omit for a tighter heading. */
  kicker?: string | null;
}) {
  return (
    <div className="flex items-center gap-3 py-1">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-slate-200/90 text-[11px] font-bold text-slate-700 dark:bg-slate-800 dark:text-slate-200">
        {id}
      </span>
      <div className="min-w-0">
        <h2 className="text-sm font-semibold tracking-tight text-slate-900 dark:text-slate-100">{title}</h2>
        {kicker ? (
          <p className="text-[11px] leading-snug text-slate-500 dark:text-slate-400 mt-0.5">{kicker}</p>
        ) : null}
      </div>
    </div>
  );
}

function branchBadge(branch: string): string {
  return branch === "audio"
    ? "bg-sky-500/15 border border-sky-400/40 text-sky-700 dark:text-sky-300"
    : "bg-violet-500/15 border border-violet-400/40 text-violet-700 dark:text-violet-300";
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

  const windows = (res.windows ?? []) as NormalizedWindowRow[];
  const isSplit = (res.raw as Record<string, unknown>)?.mode === "split_sliding_windows";
  const rawRoot = res.raw as Record<string, unknown>;
  const mparamsRaw = (rawRoot.multicat_params ?? {}) as Record<string, unknown>;
  const isMulticatJob =
    Boolean(res.multicat_video_only) || Boolean(mparamsRaw.multicat_video_only);
  const multicatSingle = Boolean(res.multicat_video_only) && !isSplit;
  const topCats = (res.cats ?? []) as NormalizedCatResult[];

  /** Section A rows: single-video uses ``topCats``; split uses all ``windows[].cats``. */
  const sectionACatRows: {
    cat: NormalizedCatRow;
    key: string;
    /** Split multicat: subheader label for first cat in each window. */
    windowHeaderText: string | null;
    /** 1-based cat index within the clip (single) or within the window (split). */
    catSlot: number;
  }[] = (() => {
    if (multicatSingle && topCats.length > 0) {
      return topCats.map((cat, i) => ({
        cat,
        key: `single-${i}`,
        windowHeaderText: null,
        catSlot: i + 1,
      }));
    }
    if (isSplit && isMulticatJob) {
      const out: {
        cat: NormalizedCatRow;
        key: string;
        windowHeaderText: string | null;
        catSlot: number;
      }[] = [];
      let serial = 0;
      windows.forEach((w, wi) => {
        const wcats = w.cats;
        if (!wcats?.length) return;
        const wIdx = w.window_index != null ? Number(w.window_index) : wi;
        const start = w.start_sec != null ? Number(w.start_sec) : null;
        const end = w.end_sec != null ? Number(w.end_sec) : null;
        const timePart =
          start != null && end != null && !Number.isNaN(start) && !Number.isNaN(end)
            ? ` · [${start.toFixed(2)}–${end.toFixed(2)}]s`
            : "";
        const headerText = `Window ${wIdx + 1}${timePart}`;
        wcats.forEach((cat, ci) => {
          const c = cat as NormalizedCatRow;
          out.push({
            cat: c,
            key: `w${wIdx}-c${ci}-${serial++}`,
            windowHeaderText: ci === 0 ? headerText : null,
            catSlot: ci + 1,
          });
        });
      });
      return out;
    }
    return [];
  })();

  const mcCount =
    res.multicat_cat_count != null && !Number.isNaN(Number(res.multicat_cat_count))
      ? Number(res.multicat_cat_count)
      : topCats.length;

  const catPainValues = sectionACatRows
    .map(({ cat }) => (cat.p_pain != null ? Number(cat.p_pain) : NaN))
    .filter((p) => !Number.isNaN(p));
  const mcPainMaxFromCats = catPainValues.length ? Math.max(...catPainValues) : null;
  const mcPainMeanFromCats = catPainValues.length
    ? catPainValues.reduce((a, b) => a + b, 0) / catPainValues.length
    : null;
  const showPerCatVerdicts = sectionACatRows.length > 0;

  const verdictDisplay =
    multicatSingle || (isSplit && isMulticatJob) ? "—" : String(res.final_decision ?? "—");
  const finalDecision = verdictDisplay;
  const pPainMax = res.p_pain_max != null ? Number(res.p_pain_max) : null;
  const pPainMean = res.p_pain_mean != null ? Number(res.p_pain_mean) : null;
  const pPainHeadline = res.p_pain_headline != null ? Number(res.p_pain_headline) : null;
  const prevLabel = summary.multicat_clip_prevalence_label
    ? String(summary.multicat_clip_prevalence_label)
    : null;
  const mStrat = (summary.multicat_params as Record<string, unknown> | undefined)?.multicat_summary_strategy;
  const summaryScalar = pPainHeadline != null ? pPainHeadline : pPainMax;
  const confidence =
    showPerCatVerdicts || summaryScalar == null
      ? null
      : finalDecision === "non_pain"
        ? 1 - summaryScalar
        : summaryScalar;
  const confidencePct = confidence == null ? null : Math.max(0, Math.min(100, confidence * 100));
  const riskLabel =
    confidencePct == null ? "unknown" : confidencePct >= 75 ? "low risk" : confidencePct >= 50 ? "medium risk" : "high risk";

  const splitSummaryLine = formatSplitSummary(split);
  const mcVerdictPainCount = showPerCatVerdicts
    ? sectionACatRows.filter(({ cat }) => String(cat.decision ?? "").toLowerCase() === "pain").length
    : 0;
  const mcVerdictNonPainCount = showPerCatVerdicts
    ? sectionACatRows.filter(({ cat }) => String(cat.decision ?? "").toLowerCase() === "non_pain").length
    : 0;
  const mcVerdictOtherCount = showPerCatVerdicts
    ? sectionACatRows.length - mcVerdictPainCount - mcVerdictNonPainCount
    : 0;

  const verdictDisplayFriendly =
    verdictDisplay === "—" ? "—" : decisionLabelForDisplay(String(verdictDisplay));
  const confidenceBarClass =
    String(finalDecision).toLowerCase() === "pain"
      ? "bg-rose-400 dark:bg-rose-500"
      : String(finalDecision).toLowerCase() === "non_pain"
        ? "bg-emerald-400 dark:bg-emerald-500"
        : "bg-slate-400 dark:bg-slate-500";

  const showClipOverview = windows.length > 1 || isSplit;

  return (
    <div className="space-y-6">
      <SectionLabel
        id="A"
        title="Summary"
        kicker="Verdict, pain scores, and how long the run took."
      />
      {isMulticatJob && (
        <div
          className="flex gap-3 rounded-xl border border-sky-200/90 dark:border-sky-500/40 bg-gradient-to-r from-sky-50/95 to-sky-50/40 dark:from-sky-950/50 dark:to-sky-950/25 px-4 py-3 text-[13px] leading-snug text-sky-950 dark:text-sky-100 shadow-sm"
          role="status"
        >
          <span
            className="mt-0.5 h-2 w-2 shrink-0 rounded-full bg-sky-500 dark:bg-sky-400"
            aria-hidden
          />
          <div>
            {isSplit
              ? `Multicat mode · ${windows.length} window(s)${
                  sectionACatRows.length ? ` · ${sectionACatRows.length} cat track(s) scored` : ""
                }`
              : `${mcCount} cat${mcCount === 1 ? "" : "s"} in this clip (multicat mode).`}
          </div>
        </div>
      )}
      <section
        className="rounded-2xl border border-slate-200 dark:border-slate-700/70 bg-white dark:bg-slate-900/80 p-5 shadow-[0_0_0_1px_rgba(15,23,42,0.08)] dark:shadow-[0_0_0_1px_rgba(15,23,42,0.4)]"
        aria-labelledby="summary-card-title"
      >
        <h3 id="summary-card-title" className="sr-only">
          Run summary
        </h3>
        <div className="grid gap-6 md:gap-6 md:grid-cols-[minmax(0,1.55fr)_minmax(0,0.85fr)_minmax(0,0.85fr)]">
          <div className="space-y-0 md:border-r md:border-slate-200 dark:md:border-slate-800 md:pr-6 pb-6 md:pb-0 border-b md:border-b-0 border-slate-200 dark:border-slate-800">
            {showPerCatVerdicts ? (
              <>
                <div className="text-[10px] uppercase tracking-[0.16em] text-slate-600 dark:text-slate-500 mb-1">
                  Verdicts by cat
                </div>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 mb-3 leading-relaxed">
                  One meta-model label per tracked cat. There is no single combined clip verdict—compare tracks
                  and windows below.
                </p>
                <div className="flex flex-wrap gap-1.5 mb-3" aria-label="Verdict counts">
                  <span className="inline-flex items-center rounded-md bg-slate-100 dark:bg-slate-800/90 px-2 py-0.5 text-[10px] font-semibold text-slate-700 dark:text-slate-200 tabular-nums">
                    {sectionACatRows.length} scored
                  </span>
                  {mcVerdictPainCount > 0 ? (
                    <span className="inline-flex items-center rounded-md bg-red-500/10 border border-red-400/25 dark:border-red-500/30 px-2 py-0.5 text-[10px] font-medium text-red-800 dark:text-red-200 tabular-nums">
                      {mcVerdictPainCount} pain
                    </span>
                  ) : null}
                  {mcVerdictNonPainCount > 0 ? (
                    <span className="inline-flex items-center rounded-md bg-emerald-500/10 border border-emerald-400/25 dark:border-emerald-500/30 px-2 py-0.5 text-[10px] font-medium text-emerald-800 dark:text-emerald-200 tabular-nums">
                      {mcVerdictNonPainCount} no pain
                    </span>
                  ) : null}
                  {mcVerdictOtherCount > 0 ? (
                    <span className="inline-flex items-center rounded-md bg-slate-500/10 border border-slate-300/40 dark:border-slate-600/50 px-2 py-0.5 text-[10px] font-medium text-slate-600 dark:text-slate-300 tabular-nums">
                      {mcVerdictOtherCount} other
                    </span>
                  ) : null}
                </div>
                <div className="rounded-lg border border-slate-200 dark:border-slate-700/70 overflow-x-auto">
                  <table className="w-full min-w-[28rem] text-sm border-collapse">
                    <caption className="sr-only">Per-cat meta-model verdicts and P(pain)</caption>
                    <thead>
                      <tr className="border-b border-slate-200 dark:border-slate-800 text-left text-slate-600 dark:text-slate-500 uppercase text-[10px] tracking-[0.16em]">
                        <th className="pl-2 pr-1 py-2 font-medium w-10">#</th>
                        <th className="px-2 py-2 font-medium w-11">Cat</th>
                        <th className="px-2 py-2 font-medium min-w-[4.5rem]">Track</th>
                        <th className="px-2 py-2 font-medium">Verdict</th>
                        <th className="px-2 py-2 font-medium text-right min-w-[4.5rem]">P(pain)</th>
                        <th className="px-2 py-2 font-medium w-24 min-w-[5.5rem]">
                          <span className="sr-only">P(pain) bar</span>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {sectionACatRows.map((row, ri) => {
                        const { cat, key, windowHeaderText, catSlot } = row;
                        const dec = String(cat.decision ?? "—");
                        const decShow = decisionLabelForDisplay(dec);
                        const pp = cat.p_pain != null ? Number(cat.p_pain) : null;
                        const ppPct =
                          pp != null && !Number.isNaN(pp) ? Math.max(0, Math.min(100, pp * 100)) : null;
                        const track = String(
                          (cat as NormalizedCatResult).track_id ?? cat.local_track_id ?? "—",
                        );
                        return (
                          <Fragment key={key}>
                            {windowHeaderText ? (
                              <tr className="bg-slate-100/95 dark:bg-slate-800/55 border-t border-slate-200 dark:border-slate-700/80">
                                <td
                                  colSpan={6}
                                  className="px-2 py-1.5 text-[10px] font-semibold tracking-wide text-slate-700 dark:text-slate-300"
                                >
                                  {windowHeaderText}
                                </td>
                              </tr>
                            ) : null}
                            <tr className="border-t border-slate-200 dark:border-slate-800/70 hover:bg-slate-100 dark:hover:bg-slate-800/35">
                              <td
                                className={`border-l-4 pl-2 pr-1 py-1.5 font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums ${verdictStripeClass(dec)}`}
                              >
                                {ri + 1}
                              </td>
                              <td className="px-2 py-1.5 font-mono text-[11px] text-slate-700 dark:text-slate-300 tabular-nums">
                                {catSlot}
                              </td>
                              <td
                                className="px-2 py-1.5 font-mono text-[11px] text-slate-800 dark:text-slate-200 tabular-nums max-w-[7rem] truncate"
                                title={track}
                              >
                                {track}
                              </td>
                              <td className="px-2 py-1.5">
                                <span
                                  className={`inline-block text-[10px] px-2 py-0.5 rounded-full font-medium whitespace-nowrap ${decisionBadge(dec)}`}
                                  title={dec}
                                >
                                  {decShow}
                                </span>
                              </td>
                              <td className="px-2 py-1.5 font-mono text-[11px] text-emerald-700 dark:text-emerald-300 tabular-nums text-right">
                                {pp != null && !Number.isNaN(pp) ? pp.toFixed(4) : "—"}
                              </td>
                              <td className="px-2 py-1.5 w-24 min-w-[5.5rem]">
                                <div
                                  className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden"
                                  title={
                                    pp != null && !Number.isNaN(pp)
                                      ? `P(pain) ${pp.toFixed(4)} (${((pp * 100).toFixed(1))}%)`
                                      : undefined
                                  }
                                >
                                  <div
                                    className={`h-full min-w-0 ${pPainBarFillClass(dec)}`}
                                    style={{ width: `${ppPct ?? 0}%` }}
                                  />
                                </div>
                              </td>
                            </tr>
                          </Fragment>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <>
                <div className="text-[10px] uppercase tracking-[0.16em] text-slate-600 dark:text-slate-500 mb-1">
                  Final verdict
                </div>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 mb-3 leading-relaxed">
                  Aggregate decision for this job. Confidence reflects how strongly the headline score supports
                  that label.
                </p>
                <div className="flex flex-wrap items-center gap-3">
                  <div
                    className={`text-4xl font-semibold tracking-tight ${
                      isMulticatJob && !showPerCatVerdicts
                        ? "text-slate-400 dark:text-slate-500"
                        : finalDecision === "pain"
                          ? "text-rose-400 dark:text-rose-300"
                          : "text-emerald-400 dark:text-emerald-300"
                    }`}
                  >
                    {verdictDisplay === "—" ? "—" : verdictDisplayFriendly}
                  </div>
                  {(!isMulticatJob || (isSplit && isMulticatJob && !showPerCatVerdicts)) && (
                    <span className={`text-[11px] px-3 py-1 rounded-full font-medium ${decisionBadge(finalDecision)}`}>
                      {riskLabel}
                    </span>
                  )}
                </div>
                {isMulticatJob && !showPerCatVerdicts ? (
                  <p className="mt-3 text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">
                    No per-cat rows match the current filters (track threshold or window). Check the timeline and
                    evidence sections for each window.
                  </p>
                ) : null}
                <div className="mt-3 text-[11px] text-slate-600 dark:text-slate-500">
                  <span className="font-medium text-slate-700 dark:text-slate-300">Confidence</span>
                  {" — "}
                  {confidencePct != null ? `${confidencePct.toFixed(1)}%` : "—"}
                  {!isMulticatJob && mStrat === "majority_above_threshold" && (
                    <span className="block mt-1.5 text-amber-800/90 dark:text-amber-200/90">
                      Headline scalar is prevalence (fraction of cats above threshold), not mean P(pain).
                    </span>
                  )}
                </div>
                <div className="mt-2 h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden max-w-xs">
                  <div
                    className={`h-full ${confidenceBarClass} transition-[width] duration-500 ease-out`}
                    style={{ width: `${confidencePct ?? 0}%` }}
                  />
                </div>
              </>
            )}
          </div>

          <div className="md:pt-1">
            <div className="text-[10px] uppercase tracking-[0.16em] text-slate-600 dark:text-slate-500">
              {showPerCatVerdicts ? "Peak P(pain)" : "Headline / max P(pain)"}
            </div>
            <div className="mt-1.5 text-3xl font-semibold text-emerald-700 dark:text-emerald-300 font-mono tabular-nums">
              {showPerCatVerdicts
                ? mcPainMaxFromCats != null
                  ? mcPainMaxFromCats.toFixed(4)
                  : "—"
                : pPainHeadline != null
                  ? pPainHeadline.toFixed(4)
                  : pPainMax != null
                    ? pPainMax.toFixed(4)
                    : "—"}
            </div>
            <p className="mt-2 text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
              {showPerCatVerdicts
                ? "Largest meta-model pain probability among the scored tracks listed at left."
                : "Primary value shown in the clip header. If headline differs from max cat, both are shown below."}
            </p>
            {!showPerCatVerdicts &&
              pPainHeadline != null &&
              pPainMax != null &&
              Math.abs(pPainHeadline - pPainMax) > 1e-6 && (
                <div className="text-[10px] text-slate-500 dark:text-slate-400 mt-2 font-mono">
                  Max per cat {pPainMax.toFixed(4)}
                </div>
              )}
          </div>

          <div className="md:pt-1">
            <div className="text-[10px] uppercase tracking-[0.16em] text-slate-600 dark:text-slate-500">
              {showPerCatVerdicts ? "Average P(pain)" : "Mean P(pain)"}
            </div>
            <div className="mt-1.5 text-3xl font-semibold text-emerald-700 dark:text-emerald-300 font-mono tabular-nums">
              {showPerCatVerdicts
                ? mcPainMeanFromCats != null
                  ? mcPainMeanFromCats.toFixed(4)
                  : "—"
                : pPainMean != null
                  ? pPainMean.toFixed(4)
                  : "—"}
            </div>
            <p className="mt-2 text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
              {showPerCatVerdicts
                ? "Simple mean of P(pain) over the same scored tracks—not a pooled model probability."
                : "Average pain probability across cats or windows for this job."}
            </p>
          </div>
        </div>
        <div className="mt-5 pt-4 border-t border-slate-200 dark:border-slate-800 flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:gap-x-6 sm:gap-y-2 text-[11px] text-slate-600 dark:text-slate-500">
          <span>
            <span className="text-slate-500 dark:text-slate-500">Runtime</span>{" "}
            <span className="text-slate-800 dark:text-slate-200 font-mono tabular-nums">
              {totalSec != null ? `${Number(totalSec).toFixed(2)}s` : "—"}
            </span>
          </span>
          {splitSummaryLine ? (
            <span className="min-w-0">
              <span className="text-slate-500 dark:text-slate-500">Split</span>{" "}
              <span className="text-slate-800 dark:text-slate-200">{splitSummaryLine}</span>
            </span>
          ) : null}
          {!isMulticatJob && prevLabel ? (
            <span className="text-slate-800 dark:text-slate-200">{prevLabel}</span>
          ) : null}
        </div>
      </section>

      {showClipOverview && <ClipPainOverview windows={windows} />}

      <SectionLabel id="B" title="Timeline" />
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

      <SectionLabel id="C" title="Evidence & Probabilities" />
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

              {branch === "video" && !multicatSingle && (
                <div>
                  <div className="text-[11px] text-slate-600 dark:text-slate-500 mb-1">pose video</div>
                  <video
                    src={artifactUrl(id, relPose)}
                    controls
                    className="w-full max-w-lg rounded-lg border border-slate-300 dark:border-slate-700"
                  />
                </div>
              )}
              {branch === "video" && multicatSingle && topCats.length > 0 && (
                <p className="text-[11px] text-slate-600 dark:text-slate-500">
                  Cropped pose videos are shown with each cat below.
                </p>
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
              {branch === "video" && (!w.cats || w.cats.length === 0) && !multicatSingle && (
                <CatResultPanel
                  label="Result"
                  pairwiseProbs={videoPairwise}
                  pairwiseWeights={videoPairwiseWeights}
                  metaClassProbs={videoMeta}
                />
              )}

              {branch === "video" && Array.isArray(w.cats) && w.cats.length > 0 && !multicatSingle && (
                <div className="space-y-4">
                  <div>
                    <CatResultPanel
                      label="Summary (coverage-weighted)"
                      header={{
                        decision: String(w.decision ?? "—"),
                        pPain: pPainWindow,
                      }}
                      pairwiseProbs={videoPairwise}
                      pairwiseWeights={videoPairwiseWeights}
                      metaClassProbs={toFriendlyMetaClassProbabilities(videoMetaRaw, pPainWindow)}
                    />
                    {!multicatSingle &&
                      typeof w.multicat_prevalence_label === "string" &&
                      w.multicat_prevalence_label ? (
                      <p className="text-xs text-amber-800 dark:text-amber-200/90 mt-2">{w.multicat_prevalence_label}</p>
                    ) : null}
                  </div>
                  {w.cats.map((cat: NormalizedCatRow, ci: number) => {
                    const cprobs = (cat.probabilities ?? {}) as Record<string, unknown>;
                    const vpCat = toProbMap(cprobs.video_pairwise);
                    const vwCat = toProbMap(cprobs.video_pairwise_weights);
                    const catMetaRaw = toProbMap(cprobs.video_meta_class_probs);
                    const pCatPain = cat.p_pain != null ? Number(cat.p_pain) : null;
                    const pv = cat.pose_video_url as string | undefined;
                    return (
                      <CatResultPanel
                        key={ci}
                        label={`Cat ${ci + 1} (track ${cat.local_track_id}, ${((cat.detection_rate_sampled ?? 0) * 100).toFixed(0)}% coverage)`}
                        header={{
                          decision: String(cat.decision ?? "—"),
                          pPain: pCatPain,
                        }}
                        pairwiseProbs={vpCat}
                        pairwiseWeights={vwCat}
                        metaClassProbs={toFriendlyMetaClassProbabilities(catMetaRaw, pCatPain)}
                        poseVideoSrc={pv ? artifactUrl(id, pv) : null}
                      />
                    );
                  })}
                </div>
              )}

              {branch === "video" && multicatSingle && i === 0 && topCats.length > 0 && (
                <div className="space-y-6 border-t border-slate-200 dark:border-slate-800 pt-4 mt-2">
                  {topCats.map((cat, ci) => {
                    const cprobs = (cat.probabilities ?? {}) as Record<string, unknown>;
                    const vpCat = toProbMap(cprobs.video_pairwise);
                    const vwCat = toProbMap(cprobs.video_pairwise_weights);
                    const catMetaRaw = toProbMap(cprobs.video_meta_class_probs);
                    const pCatPain = cat.p_pain != null ? Number(cat.p_pain) : null;
                    const tid = cat.track_id ?? cat.local_track_id;
                    const poseRel = cat.pose_video_url as string | undefined;
                    return (
                      <div
                        key={`mc-${ci}`}
                        className="space-y-3 pb-6 border-b border-slate-200 dark:border-slate-800 last:border-0 last:pb-0"
                      >
                        <div className="text-[13px] font-medium text-slate-800 dark:text-slate-200">
                          Cat {ci + 1} · track {tid} · {((cat.detection_rate_sampled ?? 0) * 100).toFixed(0)}% frame coverage
                        </div>
                        <CatResultPanel
                          label={`Track ${tid}`}
                          header={{
                            decision: String(cat.decision ?? "—"),
                            pPain: pCatPain,
                          }}
                          pairwiseProbs={vpCat}
                          pairwiseWeights={vwCat}
                          metaClassProbs={toFriendlyMetaClassProbabilities(catMetaRaw, pCatPain)}
                          poseVideoSrc={poseRel ? artifactUrl(id, poseRel) : null}
                        />
                      </div>
                    );
                  })}
                </div>
              )}

              {w.multicat_empty_reason && (
                <p className="text-sm text-amber-800 dark:text-amber-200/90 rounded-md bg-amber-500/10 border border-amber-500/30 p-3">
                  No cat met the tracking threshold in this segment:{" "}
                  <span className="font-mono">{w.multicat_empty_reason}</span>
                </p>
              )}
            </div>
          );
        })}
      </section>

      {arts && (
        <>
          <SectionLabel id="D" title="Artifacts" />
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
