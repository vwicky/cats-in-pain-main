import type { JobResponse, NormalizedResult } from "../../shared/api-types";

const API = "";

export async function createJobUpload(form: FormData): Promise<{ id: string }> {
  const r = await fetch(`${API}/jobs`, { method: "POST", body: form });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchJob(id: string): Promise<JobResponse> {
  const r = await fetch(`${API}/jobs/${id}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchResult(id: string): Promise<NormalizedResult> {
  const r = await fetch(`${API}/jobs/${id}/result`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchArtifacts(id: string): Promise<Record<string, string[]>> {
  const r = await fetch(`${API}/jobs/${id}/artifacts`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function artifactUrl(jobId: string, relPath: string): string {
  const enc = relPath.split("/").map(encodeURIComponent).join("/");
  return `${API}/jobs/${jobId}/artifacts/${enc}`;
}
