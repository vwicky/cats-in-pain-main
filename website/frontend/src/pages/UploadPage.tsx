import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createJobUpload } from "../api";

function slidingWindows(
  durationSec: number,
  windowSec: number,
  stepSec: number,
): [number, number][] {
  if (durationSec <= 0 || windowSec <= 0 || stepSec <= 0) return [];
  const out: [number, number][] = [];
  let t = 0;
  while (t < durationSec - 1e-9) {
    const end = Math.min(t + windowSec, durationSec);
    out.push([t, end]);
    if (end >= durationSec - 1e-9) break;
    t += stepSec;
  }
  return out;
}

export default function UploadPage() {
  const nav = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [mode, setMode] = useState<"upload" | "local_path">("upload");
  const [localPath, setLocalPath] = useState("");
  const [device, setDevice] = useState("auto");
  const [catThreshold, setCatThreshold] = useState(0.5);
  const [winSec, setWinSec] = useState(6);
  const [stepSec, setStepSec] = useState(3);
  const [multicatVideoOnly, setMulticatVideoOnly] = useState(false);
  const [multicatAdvOpen, setMulticatAdvOpen] = useState(false);
  const [multicatMaxCats, setMulticatMaxCats] = useState(8);
  const [multicatMinCoverage, setMulticatMinCoverage] = useState(0.15);
  const [multicatPainTh, setMulticatPainTh] = useState(0.5);
  const [multicatStrategy, setMulticatStrategy] = useState("coverage_weighted_mean");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const previews = useMemo(() => {
    if (duration == null || winSec <= 0 || stepSec <= 0) return [];
    return slidingWindows(duration, winSec, stepSec);
  }, [duration, winSec, stepSec]);

  function onFile(f: File | null) {
    setFile(f);
    setDuration(null);
    if (!f) return;
    const url = URL.createObjectURL(f);
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => {
      setDuration(Number.isFinite(v.duration) ? v.duration : null);
      URL.revokeObjectURL(url);
    };
    v.onerror = () => {
      setDuration(null);
      URL.revokeObjectURL(url);
    };
    v.src = url;
  }

  async function submit() {
    setErr(null);
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("mode", mode);
      fd.append("device", device);
      fd.append("cat_threshold", String(catThreshold));
      fd.append("split_window_sec", String(winSec));
      fd.append("split_step_sec", String(stepSec));
      console.log(
        "PRE-APPEND multicat state:",
        multicatVideoOnly,
        "fd entries:",
        [...fd.entries()].filter(([k]) => k === "multicat_video_only"),
      );
      fd.append("multicat_video_only", multicatVideoOnly ? "true" : "false");
      fd.append("multicat_max_cats", String(multicatMaxCats));
      fd.append("multicat_min_track_coverage", String(multicatMinCoverage));
      fd.append("multicat_decision_threshold", String(multicatPainTh));
      fd.append("multicat_summary_strategy", multicatStrategy);
      if (mode === "upload") {
        if (!file) throw new Error("Choose a video file");
        fd.append("file", file);
      } else {
        fd.append("video_path", localPath);
      }
      const res = await createJobUpload(fd);
      nav(`/job/${res.id}`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-1">New run</h1>
        <p className="text-slate-600 dark:text-slate-400 text-sm max-w-2xl">
          Upload routes through the same pipeline as CLI. Parameters affect sliding windows and branch
          thresholding; preview explains how the video is segmented before you spend GPU time.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="space-y-4 bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 rounded-lg p-4">
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Input mode</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as "upload" | "local_path")}
            className="w-full bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm"
          >
            <option value="upload">upload (browser → server)</option>
            <option value="local_path">local_path (server filesystem; requires ALLOW_LOCAL_PATHS)</option>
          </select>

          {mode === "upload" ? (
            <div>
              <label className="block text-sm text-slate-600 dark:text-slate-400 mb-1">Video file</label>
              <input
                type="file"
                accept="video/*"
                onChange={(e) => onFile(e.target.files?.[0] ?? null)}
                className="text-sm w-full"
              />
            </div>
          ) : (
            <div>
              <label className="block text-sm text-slate-600 dark:text-slate-400 mb-1">Absolute path on server</label>
              <input
                value={localPath}
                onChange={(e) => setLocalPath(e.target.value)}
                placeholder="/path/to/video.mp4"
                className="w-full bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-sm"
              />
            </div>
          )}

          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={multicatVideoOnly}
              onChange={(e) => setMulticatVideoOnly(e.target.checked)}
              className="mt-1 rounded border-slate-400"
            />
            <span>
              <span className="block text-sm font-medium text-slate-800 dark:text-slate-200">
                Multiple cats — video-only pipeline
              </span>
              <span className="block text-xs text-slate-600 dark:text-slate-500 mt-0.5">
                Forces the video (pose + ST-GCN) branch for every clip and scores each SORT track separately. YAMNet
                P(cat) is still logged but does not choose the audio branch. Unchecked = default single-pose routing
                (audio or video).
              </span>
            </span>
          </label>
          <p className="text-xs text-red-500">multicat state: {String(multicatVideoOnly)}</p>

          {multicatVideoOnly && (
            <div className="border border-slate-200 dark:border-slate-700 rounded-md p-3 space-y-3">
              <button
                type="button"
                onClick={() => setMulticatAdvOpen((o) => !o)}
                className="text-xs text-emerald-600 dark:text-emerald-400 hover:underline"
              >
                {multicatAdvOpen ? "Hide" : "Show"} multicat advanced parameters
              </button>
              {multicatAdvOpen && (
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-slate-600 dark:text-slate-500">multicat_max_cats</label>
                    <input
                      type="number"
                      min={1}
                      max={32}
                      value={multicatMaxCats}
                      onChange={(e) => setMulticatMaxCats(parseInt(e.target.value, 10) || 8)}
                      className="w-full mt-1 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded px-2 py-1.5 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-slate-600 dark:text-slate-500">min_track_coverage</label>
                    <input
                      type="number"
                      step="0.05"
                      value={multicatMinCoverage}
                      onChange={(e) => setMulticatMinCoverage(parseFloat(e.target.value))}
                      className="w-full mt-1 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded px-2 py-1.5 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-slate-600 dark:text-slate-500">pain decision threshold</label>
                    <input
                      type="number"
                      step="0.05"
                      value={multicatPainTh}
                      onChange={(e) => setMulticatPainTh(parseFloat(e.target.value))}
                      className="w-full mt-1 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded px-2 py-1.5 text-sm"
                    />
                  </div>
                  <div className="col-span-2">
                    <label className="text-xs text-slate-600 dark:text-slate-500">summary strategy</label>
                    <select
                      value={multicatStrategy}
                      onChange={(e) => setMulticatStrategy(e.target.value)}
                      className="w-full mt-1 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded px-2 py-1.5 text-sm"
                    >
                      <option value="coverage_weighted_mean">coverage_weighted_mean (default)</option>
                      <option value="mean">mean</option>
                      <option value="max">max</option>
                      <option value="majority_above_threshold">majority_above_threshold (prevalence)</option>
                    </select>
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-slate-600 dark:text-slate-500">device</label>
              <select
                value={device}
                onChange={(e) => setDevice(e.target.value)}
                className="w-full mt-1 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded px-2 py-1.5 text-sm"
              >
                <option value="auto">auto</option>
                <option value="cpu">cpu</option>
                <option value="cuda">cuda</option>
                <option value="mps">mps</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-slate-600 dark:text-slate-500">cat_threshold</label>
              <input
                type="number"
                step="0.05"
                value={catThreshold}
                onChange={(e) => setCatThreshold(parseFloat(e.target.value))}
                className="w-full mt-1 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded px-2 py-1.5 text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-slate-600 dark:text-slate-500">split_window_sec</label>
              <input
                type="number"
                step="0.5"
                value={winSec}
                onChange={(e) => setWinSec(parseFloat(e.target.value))}
                className="w-full mt-1 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded px-2 py-1.5 text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-slate-600 dark:text-slate-500">split_step_sec</label>
              <input
                type="number"
                step="0.5"
                value={stepSec}
                onChange={(e) => setStepSec(parseFloat(e.target.value))}
                className="w-full mt-1 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded px-2 py-1.5 text-sm"
              />
            </div>
          </div>

          {err && <p className="text-red-400 text-sm whitespace-pre-wrap">{err}</p>}

          <button
            type="button"
            disabled={busy}
            onClick={() => void submit()}
            className="w-full py-2 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 font-medium"
          >
            {busy ? "Starting…" : "Start job"}
          </button>
        </div>

        <div className="bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 rounded-lg p-4 space-y-3">
          <h2 className="font-semibold text-slate-900 dark:text-slate-200">Window preview</h2>
          <p className="text-xs text-slate-600 dark:text-slate-500">
            Uses the same sliding rule as the pipeline. Backend worker also precomputes{" "}
            <code className="text-emerald-600 dark:text-emerald-400">n_windows</code> for the progress bar.
          </p>
          <div className="text-sm text-slate-700 dark:text-slate-300">
            <div>Duration: {duration != null ? `${duration.toFixed(3)}s` : "— (select a video)"}</div>
            <div>
              Window: {winSec}s · Step: {stepSec}s
            </div>
            <div>Windows: {previews.length}</div>
          </div>
          <ul className="text-xs font-mono text-emerald-700 dark:text-emerald-300/90 max-h-48 overflow-y-auto space-y-1">
            {previews.map(([a, b], i) => (
              <li key={i}>
                [{a.toFixed(2)}–{b.toFixed(2)}]
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
