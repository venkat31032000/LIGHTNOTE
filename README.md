# Video object editor

Replace or remove an object in a short video using a natural-language instruction.

Upload a clip, optionally supply a reference image of the replacement, and type
something like `Replace the Pepsi bottle with a Fanta bottle`. The backend reads the
instruction, locates the object, segments it, tracks it across every frame, erases the
original and composites the replacement in its place.

Detection and segmentation run locally. Only the instruction parsing and one
orientation check need an API, so the repository runs with a single OpenAI key and
no GPU.

---

## Architecture

```
React (Vite)                     FastAPI
─────────────                    ───────
POST /jobs        ──────────►    accept upload, queue BackgroundTask, return job_id
GET  /status      ◄─────────►    poll stage + frames_done
GET  /output      ◄──────────    serve the encoded mp4

                                 pipeline (background)
                                 ─────────────────────
                                 1. parse instruction      gpt-4o-mini, JSON mode
                                 2. decode video           OpenCV
                                 3. resolve orientation    gpt-4o-mini, vision
                                 4. per frame:
                                      detect               OWLv2
                                      segment              SAM
                                      erase                cv2.inpaint (Telea)
                                      composite            perspective warp + alpha
                                 5. encode                 VideoWriter → ffmpeg H.264
```

The frontend holds no processing logic — it posts a multipart form, polls for status,
and plays the result. Job state lives in an in-memory dict, which is deliberate for a
prototype and the first thing to replace in production.

---

## AI and models

| Step | Model | Why |
| --- | --- | --- |
| Instruction understanding | `gpt-4o-mini`, `response_format=json_object` | Guaranteed parseable output in one call. Returns `operation`, `target`, `detector_query` and `replacement` as separate fields. |
| Object detection | `google/owlv2-base-patch16-ensemble` | Open-vocabulary, so any noun works with no retraining. Runs locally on CPU. |
| Segmentation | `facebook/sam-vit-base` | Prompted with the detector's box; gives a pixel-accurate mask rather than a rectangle. |
| Orientation | `gpt-4o-mini`, vision | Decides which way round the replacement goes. One call per job. |
| Object removal | `cv2.inpaint` (Telea) | Classical and instant. Enough because the replacement covers most of the erased region. |

### Why `detector_query` is separate from `target`

Object detectors score poorly on brand names and well on plain nouns. The parser
returns both: `target` keeps the user's phrasing (`"Pepsi bottle"`) for display and
intent, while `detector_query` is the generic noun (`"bottle"`) that actually goes to
OWLv2. The query is also wrapped as `"a photo of a {query}"` — OWL-family models are
trained on caption-like text and score noticeably higher on a phrase than a bare noun.

### Why Grounding DINO was dropped

Grounding DINO was the first choice. Its published checkpoint failed to load the
box-prediction head against the installed `transformers` version, so those weights came
back randomly initialised — producing plausible-looking but meaningless boxes before
crashing outright on the next call. OWLv2 loads cleanly and does the same job.

Worth noting how that surfaced: the first detection returned a box scoring 0.29, which
looked like a weak detection. It was random output from an uninitialised layer.
Rendering the box onto the frame is what exposed it, which is why every stage of this
pipeline has a visual debug endpoint.

---

## Tracking

Detection and segmentation are re-run independently on every frame rather than using a
video object tracker such as SAM 2.

For clips of one to three seconds this is the right trade: simple, no temporal state to
drift, automatic recovery if the object is briefly lost, no extra model. Measured on the
demo clip — 34 of 34 frames detected, zero misses, mask area stable throughout.

The cost is no object identity across frames, which matters with more than one instance
present. See Limitations.

---

## Replacement

With a reference image supplied:

1. **Key the background.** A PNG's own alpha is used when present. Otherwise near-white,
   low-saturation pixels are treated as background, cleaned with a morphological open,
   reduced to the largest connected region — which drops reflections, shadows and JPEG
   speckle — and cropped to the alpha bounding box.
2. **Erase the original.** The mask is dilated and inpainted, so nothing of the original
   shows through gaps in the replacement's silhouette.
3. **Fit to the object's pose.** `cv2.minAreaRect` gives the object's oriented bounding
   box. The reference is warped onto it with a perspective transform, scaled uniformly
   so its aspect ratio survives rather than being stretched to fill the box.
4. **Blend.** Alpha-composited over the inpainted frame with a feathered edge.

Without a reference the object is recoloured by shifting hue within the mask. Not a real
replacement, but it demonstrates the mask is correct and gives the endpoint a sensible
default.

### Orientation

This turned out to be the hardest part, and it has two independent failure modes.

**Which end is the top.** `minAreaRect` returns an axis, not a direction, so a bottle can
land cap-first or base-first. The first approach compared pixel mass — an asymmetric
object carries less mass at its narrow end — which works for bottles and fails for cans:
the demo can measured 28,389 pixels in its top half against 28,629 in its bottom, a ratio
of 0.992. That is a coin flip.

The current approach sends the object cropped from the frame and the reference to a
vision model and asks whether the replacement needs turning around. It is decided **once**,
on the first frame where the object is found, and reused for the whole clip — deciding
per frame lets consecutive frames disagree, which shows up as the object turning over
mid-clip. The mass heuristic remains as the fallback when the call fails, and a `flip`
parameter overrides either.

**Direction continuity.** No fixed rule for choosing a direction on an axis is continuous
over a full rotation; any rule breaks somewhere, and at that angle the replacement swaps
end for end between consecutive frames. Rotating a test mask through 360° in 2° steps, a
fixed rule moved the reference's cap by up to **160 px** in a single frame. Each frame now
picks the sign that agrees with the previous frame's axis, with the fixed rule used only
for frame one. The same test then gives a largest step of **3.3 px**, which is the mask's
own jitter.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness and job count |
| `POST` | `/jobs` | `prompt`, `video`, optional `reference_image`, `flip`, `girth`. Returns a `job_id` immediately. |
| `GET` | `/jobs/{id}/status` | Stage, `frames_done` / `total_frames`, the parsed plan, `misses`, `orientation_source` |
| `GET` | `/jobs/{id}/output` | The encoded mp4 |
| `POST` | `/parse` | Instruction parsing alone |

Stages: `queued`, `parsing_instruction`, `reading_video`, `tracking_object`,
`encoding_video`, `done`, `failed`.

### Debug endpoints

Every failure in this pipeline is easier to see than to read in a traceback.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/debug/upload` | Store a video and get a job id, no processing |
| `POST` | `/debug/{id}/masks` | Detect and segment every frame; per-frame score and mask area |
| `GET` | `/debug/{id}/sheet` | Contact sheet of every frame with its mask overlaid; a yellow diagonal marks a frame where detection failed |
| `GET` | `/debug/reference/{id}` | The reference after keying, over a grid, so a background that failed to key is obvious |

---

## Setup

Requires Python 3.11, Node 18+, and `ffmpeg` on `PATH`. Without ffmpeg the output stays
mp4v, which most browsers will not play.

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `backend/.env` from `.env.example`:

```
OPENAI_API_KEY=sk-...
```

Then:

```bash
uvicorn main:app --port 8000
```

The first job downloads roughly 1 GB of model weights to the Hugging Face cache and will
appear to hang. Later jobs reuse the loaded models. API docs at `http://127.0.0.1:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Set `VITE_API_URL` if the backend is elsewhere.

### Input guidance

**Reference images.** A transparent PNG of the object alone works best; a photo on plain
white also keys correctly. An image on a coloured backdrop will be pasted wholesale,
background included — check `/debug/reference/{id}` if the result looks like a block
rather than an object.

**Clips.** Inference is a few seconds per frame on CPU, and frames are held in memory, so
keep test clips short and small. Around 2 seconds at 12 fps and 360 px wide is a good
size. Avoid clips that span a scene cut, and avoid scenes containing more than one
instance of the target.

---

## Known limitations

**No temporal tracking.** Frames are processed independently, so masks can wobble slightly
and there is no object identity. Only the highest-scoring detection is used, so a scene
with two instances of the target will edit one and may jump between them. SAM 2's video
propagation is the correct fix.

**Frames held in memory.** The whole decoded clip sits in a list. Fine at 30 frames; a
minute of 1080p would exhaust RAM. Streaming to disk is the fix.

**Job state is in-memory.** Restarting the server loses all job records and any job in
flight. Redis or a database plus a real worker queue, rather than `BackgroundTasks`, is
what this needs to survive a deploy.

**The replacement is composited, not generated.** The reference is warped into place, so
its lighting, shadow and specular highlights do not match the scene. A generative
inpainting model conditioned on the replacement text would look far better; that was the
original plan and was dropped for want of a hosted-inference budget. It is isolated in
one function (`edit_frame`), so swapping it is contained.

**Occlusion boundaries are rough.** Where the object touches something in front of it —
the subject's lip in the demo clip — SAM segments the boundary well, but the composited
replacement still meets it with a hard edge, because nothing models depth ordering.

**Orientation resolves one axis only.** The vision check and its geometric fallback both
resolve the 180° ambiguity along the long axis. Neither can recover rotation about the
other two axes — a phone face-up versus face-down is indistinguishable from a 2D mask.

**Audio is dropped.** `cv2.VideoWriter` writes video only; the output is silent.

**CPU inference is slow.** Two to five seconds per frame for detection plus segmentation.

**CORS is fully open.** `allow_origins=["*"]` is a local-development convenience and must
be narrowed before any deployment.

**Not implemented:** text replacement within the frame, and video URL input. Both optional.

---

## How it was verified

Each stage was confirmed against real output before the next was built, rather than
assembled and debugged as a whole:

- Instruction parsing returned the four expected fields for the demo prompt.
- Detection was rendered as a labelled box on frame 0 — 0.68 confidence, tight to the
  bottle. This is what exposed the Grounding DINO checkpoint problem.
- Segmentation was rendered as a mask overlay, confirming the mask followed the label
  contour and stopped at the lip rather than bleeding into the face.
- Tracking was checked as a contact sheet of all 34 frames: every frame found, mask
  locked to the bottle through the full tilt.
- Compositing was checked frame by frame, which caught both the un-keyed reference
  background and the stretched aspect ratio.
- Orientation continuity was measured directly, rotating a synthetic mask through 360°
  and recording how far the replacement moved between adjacent angles: 160 px before the
  fix, 3.3 px after.
