import { useEffect, useMemo, useRef, useState } from "react";
import { createJob, outputUrl, pollJob } from "./api";

const STAGES = [
  ["parsing_instruction", "Read the instruction"],
  ["reading_video", "Decode the video"],
  ["tracking_object", "Find and track the object"],
  ["encoding_video", "Encode the result"],
];

const TERMINAL = ["done", "failed"];

function FilePicker({ label, hint, accept, file, onPick }) {
  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  return (
    <div className="field">
      <label>
        <span className="field-name">{label}</span>
        {hint && <span className="hint"> {hint}</span>}
        <input type="file" accept={accept} onChange={(e) => onPick(e.target.files[0] ?? null)} />
      </label>
      {preview &&
        (accept.startsWith("video") ? (
          <video className="thumb" src={preview} muted controls />
        ) : (
          <img className="thumb" src={preview} alt="" />
        ))}
    </div>
  );
}

export default function App() {
  const [video, setVideo] = useState(null);
  const [reference, setReference] = useState(null);
  const [prompt, setPrompt] = useState("Replace the Pepsi bottle with a Fanta bottle");
  const [flip, setFlip] = useState(false);
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const cancel = useRef(null);

  useEffect(() => () => cancel.current?.(), []);

  const running = job && !TERMINAL.includes(job.stage);
  const done = job?.stage === "done";
  const currentIndex = STAGES.findIndex(([s]) => s === job?.stage);

  const percent =
    job?.total_frames && job?.frames_done
      ? Math.round((job.frames_done / job.total_frames) * 100)
      : 0;

  async function start() {
    setError(null);
    cancel.current?.();
    try {
      const { job_id } = await createJob({
        prompt,
        video,
        referenceImage: reference,
        flip,
      });
      setJob({ job_id, stage: "queued", frames_done: 0, total_frames: null });
      // Merge rather than replace: a reconnect notice is a partial update and
      // must not wipe the stage, plan or progress already on screen.
      cancel.current = pollJob(job_id, {
        onUpdate: (update) => setJob((prev) => ({ ...prev, ...update })),
      });
    } catch (e) {
      setError(e.message);
    }
  }

  function stageClass(index) {
    if (done || (currentIndex > -1 && currentIndex > index)) return "done";
    if (currentIndex === index) return "active";
    return "todo";
  }

  return (
    <main>
      <header>
        <h1>Video object editor</h1>
        <p className="lede">
          Upload a clip and describe the change in plain language. The backend reads the
          instruction, finds the object, tracks it across every frame, and replaces it.
        </p>
      </header>

      <section className="panel">
        <FilePicker label="Video" accept="video/*" file={video} onPick={setVideo} />

        <FilePicker
          label="Reference image"
          hint="optional — a transparent PNG of the replacement works best"
          accept="image/*"
          file={reference}
          onPick={setReference}
        />

        <div className="field">
          <label>
            <span className="field-name">Instruction</span>
            <textarea
              rows={2}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Replace the bottle with a can"
            />
          </label>
        </div>

        <div className="field">
          <label className="checkbox">
            <input type="checkbox" checked={flip} onChange={(e) => setFlip(e.target.checked)} />
            <span>
              Flip the replacement end to end
              <span className="hint">
                {" "}
                — use this if the object comes out facing the wrong way
              </span>
            </span>
          </label>
        </div>

        <button onClick={start} disabled={!video || !prompt.trim() || running}>
          {running ? "Processing…" : "Start editing"}
        </button>
      </section>

      {error && <p className="error panel">{error}</p>}

      {job && (
        <section className="panel">
          <ol className="stages">
            {STAGES.map(([id, label], i) => (
              <li key={id} className={stageClass(i)}>
                <span>{label}</span>
                {id === "tracking_object" && job.total_frames ? (
                  <span className="count">
                    {job.frames_done} of {job.total_frames} frames
                  </span>
                ) : null}
              </li>
            ))}
          </ol>

          {job.stage === "tracking_object" && (
            <div className="bar" role="progressbar" aria-valuenow={percent}>
              <div style={{ width: `${percent}%` }} />
            </div>
          )}

          {job.reconnecting && !TERMINAL.includes(job.stage) && (
            <p className="hint">Waiting for the server to respond…</p>
          )}

          {job.plan && (
            <dl className="plan">
              <dt>Operation</dt>
              <dd>{job.plan.operation}</dd>
              <dt>Target</dt>
              <dd>{job.plan.target}</dd>
              <dt>Detector query</dt>
              <dd>{job.plan.detector_query}</dd>
              <dt>Replacement</dt>
              <dd>{job.plan.replacement ?? "—"}</dd>
            </dl>
          )}

          {job.stage === "failed" && (
            <p className="error">{job.error ?? "The job failed. Check the server log."}</p>
          )}

          {done && (
            <div className="result">
              {job.misses > 0 && (
                <p className="hint">
                  The object was not found in {job.misses}{" "}
                  {job.misses === 1 ? "frame" : "frames"}; those were left unchanged.
                </p>
              )}
              <video src={outputUrl(job.job_id)} controls autoPlay loop playsInline />
              <a href={outputUrl(job.job_id)} download={`${job.job_id}.mp4`}>
                Download the edited video
              </a>
            </div>
          )}
        </section>
      )}
    </main>
  );
}