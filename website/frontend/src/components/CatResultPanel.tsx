import type { ReactNode } from "react";
import { useEffect, useState } from "react";

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

function decisionBadge(decision: string): string {
  if (decision === "pain") return "bg-red-500/15 border border-red-400/40 text-red-700 dark:text-red-300";
  if (decision === "non_pain") return "bg-emerald-500/15 border border-emerald-400/40 text-emerald-700 dark:text-emerald-300";
  return "bg-slate-500/10 border border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300";
}

export function ProbabilityBars({
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

export type CatResultPanelProps = {
  label: string;
  header?: { decision: string; pPain: number | null };
  pairwiseProbs: Record<string, number>;
  pairwiseWeights: Record<string, number>;
  metaClassProbs: Record<string, number>;
  /** Run-relative pose clip path (resolved via ``artifactUrl``). */
  poseVideoSrc?: string | null;
};

function VideoPairwiseMetaGrid({
  pairwiseProbs,
  pairwiseWeights,
  metaClassProbs,
}: {
  pairwiseProbs: Record<string, number>;
  pairwiseWeights: Record<string, number>;
  metaClassProbs: Record<string, number>;
}) {
  return (
    <div className="grid md:grid-cols-2 gap-5">
      <ProbabilityBars
        title="pairwise sub-model probs (each is P(Paining))"
        probs={pairwiseProbs}
        colorClass="bg-violet-400"
        formatLabel={(label) => formatPairwiseLabel(label)}
        wrapLabels
        accentRightColumn
        formatRightMeta={(label) => {
          const wgt = pairwiseWeights[label];
          if (wgt == null || Number.isNaN(wgt)) return null;
          const sign = wgt >= 0 ? "+" : "";
          return (
            <span
              className={`inline-block rounded px-1.5 py-0.5 font-mono text-[10px] tabular-nums ${metaWeightPillClass(wgt)}`}
              title="Meta-model coefficient for this pairwise feature (sign: pushes toward pain vs non-pain log-odds)"
            >
              w={sign}
              {wgt.toFixed(3)}
            </span>
          );
        }}
      />
      <ProbabilityBars
        title="meta class probs (final classifier)"
        probs={metaClassProbs}
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
  );
}

export function CatResultPanel({
  label,
  header,
  pairwiseProbs,
  pairwiseWeights,
  metaClassProbs,
  poseVideoSrc,
}: CatResultPanelProps) {
  const showSrOnlyLabel = label === "Result" && header == null;

  if (showSrOnlyLabel) {
    return (
      <>
        <span className="sr-only">{label}</span>
        <VideoPairwiseMetaGrid
          pairwiseProbs={pairwiseProbs}
          pairwiseWeights={pairwiseWeights}
          metaClassProbs={metaClassProbs}
        />
        {poseVideoSrc ? (
          <div className="mt-4">
            <div className="text-[11px] text-slate-600 dark:text-slate-500 mb-1">pose video</div>
            <video
              src={poseVideoSrc}
              controls
              className="w-full max-w-lg rounded-lg border border-slate-300 dark:border-slate-700"
            />
          </div>
        ) : null}
      </>
    );
  }

  return (
    <div className="space-y-3">
      <div className="text-[11px] font-medium text-slate-600 dark:text-slate-400">{label}</div>
      {header && (
        <div className="flex flex-wrap gap-2 text-xs items-center">
          <span className={`px-2 py-0.5 rounded-full ${decisionBadge(String(header.decision ?? "—"))}`}>
            {header.decision ?? "—"}
          </span>
          <span className="font-mono text-emerald-600 dark:text-emerald-300">
            p_pain {header.pPain != null && !Number.isNaN(header.pPain) ? header.pPain.toFixed(4) : "—"}
          </span>
        </div>
      )}
      <VideoPairwiseMetaGrid
        pairwiseProbs={pairwiseProbs}
        pairwiseWeights={pairwiseWeights}
        metaClassProbs={metaClassProbs}
      />
      {poseVideoSrc ? (
        <div>
          <div className="text-[11px] text-slate-600 dark:text-slate-500 mb-1">pose video</div>
          <video
            src={poseVideoSrc}
            controls
            className="w-full max-w-lg rounded-lg border border-slate-300 dark:border-slate-700"
          />
        </div>
      ) : null}
    </div>
  );
}
