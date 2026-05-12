import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import type { JobResponse } from "../../../shared/api-types";
import { fetchJob } from "../api";

export default function JobPage() {
  const { id = "" } = useParams();
  const [job, setJob] = useState<JobResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    async function tick() {
      try {
        const j = await fetchJob(id);
        if (!stop) {
          setJob(j);
          setErr(null);
        }
      } catch (e) {
        if (!stop) setErr(e instanceof Error ? e.message : String(e));
      }
    }
    void tick();
    const t = setInterval(() => void tick(), 2500);
    return () => {
      stop = true;
      clearInterval(t);
    };
  }, [id]);

  if (err && !job) return <p className="text-red-500 dark:text-red-400">{err}</p>;
  if (!job) return <p className="text-slate-600 dark:text-slate-400">Loading job…</p>;

  const pct = job.progress.percent ?? 0;
  const uiWindow = (job.progress.window_idx ?? 0) + 1;
  const nWin = job.progress.n_windows;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-bold">Job {job.id}</h1>
        <span
          className={`text-xs px-2 py-0.5 rounded font-mono ${
            job.status === "done"
              ? "bg-emerald-900 text-emerald-200"
              : job.status === "failed"
                ? "bg-red-900 text-red-200"
                : "bg-amber-900 text-amber-200"
          }`}
        >
          {job.status}
        </span>
        {job.status === "done" && (
          <Link
            to={`/job/${id}/results`}
            className="text-sm text-emerald-600 dark:text-emerald-400 underline"
          >
            Open results
          </Link>
        )}
      </div>

      <div className="bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 rounded-lg p-4 space-y-3 max-w-xl">
        <div className="text-sm text-slate-600 dark:text-slate-400">Progress</div>
        <div className="h-2 bg-slate-200 dark:bg-slate-800 rounded overflow-hidden">
          <div
            className="h-full bg-emerald-600 transition-all"
            style={{ width: `${Math.min(100, pct)}%` }}
          />
        </div>
        <div className="text-sm text-slate-700 dark:text-slate-300">
          <div>
            Stage: <code className="text-emerald-600 dark:text-emerald-300">{job.progress.stage ?? "—"}</code>
          </div>
          <div>
            Window (UI 1-based):{" "}
            <code className="text-emerald-600 dark:text-emerald-300">
              {uiWindow}
              {nWin != null ? ` / ${nWin}` : ""}
            </code>
          </div>
        </div>
      </div>

      <details className="text-sm max-w-3xl">
        <summary className="cursor-pointer text-slate-600 dark:text-slate-400">Concise structured events</summary>
        <pre className="mt-2 p-3 bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded overflow-x-auto text-xs text-slate-700 dark:text-slate-400">
          {JSON.stringify(job.progress.events ?? [], null, 2)}
        </pre>
      </details>

      {job.status === "failed" && (
        <div className="text-red-300 text-sm">
          <div className="font-mono text-xs">{job.error_type}</div>
          <div>{job.error_message}</div>
        </div>
      )}
    </div>
  );
}
