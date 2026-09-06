export const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function unwrap(res) {
  if (res.ok) return res.json();
  let detail = `Request failed (${res.status})`;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    /* server returned a non-JSON body */
  }
  throw new Error(detail);
}

export async function createJob({ prompt, video, referenceImage, flip = false, girth = 1 }) {
  const form = new FormData();
  form.append("prompt", prompt);
  form.append("video", video);
  if (referenceImage) form.append("reference_image", referenceImage);
  form.append("flip", String(flip));
  form.append("girth", String(girth));
  return unwrap(await fetch(`${API}/jobs`, { method: "POST", body: form }));
}

export async function getStatus(jobId) {
  return unwrap(await fetch(`${API}/jobs/${jobId}/status`));
}

export const outputUrl = (jobId) => `${API}/jobs/${jobId}/output`;

/**
 * Poll a job until it reaches a terminal stage. Returns a cancel function.
 *
 * A single failed poll does not fail the job. The server restarts on file save
 * and is CPU-saturated while a job runs, so transient errors are expected;
 * only a sustained run of failures is treated as the job being gone.
 */
export function pollJob(jobId, { onUpdate, intervalMs = 1200, maxStrikes = 5 }) {
  let stopped = false;
  let timer = null;
  let strikes = 0;

  const tick = async () => {
    if (stopped) return;
    try {
      const status = await getStatus(jobId);
      if (stopped) return;
      strikes = 0;
      onUpdate(status);
      if (status.stage === "done" || status.stage === "failed") return;
    } catch (err) {
      if (stopped) return;
      strikes += 1;
      if (strikes >= maxStrikes) {
        onUpdate({
          job_id: jobId,
          stage: "failed",
          error: `Lost contact with the server: ${err.message}`,
        });
        return;
      }
      onUpdate({ job_id: jobId, reconnecting: true });
    }
    timer = setTimeout(tick, intervalMs);
  };

  tick();
  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
  };
}