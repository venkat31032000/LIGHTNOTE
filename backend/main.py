from fastapi import (
    FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline, SamModel, SamProcessor
from PIL import Image
from typing import Optional
import shutil, uuid, cv2, json, subprocess, traceback, base64
import numpy as np
import torch

load_dotenv()
client = OpenAI()

app = FastAPI(title="LightNote video object editor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
FRAME_DIR = Path("frames")
REF_DIR = Path("refs")
OUTPUT_DIR = Path("outputs")
for d in (UPLOAD_DIR, FRAME_DIR, REF_DIR, OUTPUT_DIR):
    d.mkdir(exist_ok=True)

JOBS = {}


# ------------------------------------------------------------------ models

detector = None
sam_model = None
sam_processor = None


def get_detector():
    global detector
    if detector is None:
        detector = pipeline(
            "zero-shot-object-detection",
            model="google/owlv2-base-patch16-ensemble",
        )
    return detector


def get_sam():
    global sam_model, sam_processor
    if sam_model is None or sam_processor is None:
        model = SamModel.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        sam_model, sam_processor = model, processor
    return sam_model, sam_processor


def mask_for_image(image, query, threshold=0.1):
    """Detect `query` in a PIL image, segment it, return (mask, detection)."""
    dets = get_detector()(
        image,
        candidate_labels=[f"a photo of a {query}"],
        threshold=threshold,
    )
    if not dets:
        return None, None

    best = max(dets, key=lambda d: d["score"])
    b = best["box"]

    model, processor = get_sam()
    inputs = processor(
        image,
        input_boxes=[[[b["xmin"], b["ymin"], b["xmax"], b["ymax"]]]],
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    k = int(torch.argmax(outputs.iou_scores[0][0]))
    mask = masks[0][0][k].numpy().astype(np.uint8) * 255
    return mask, best


# ------------------------------------------------------------ instruction

SYSTEM = """You convert a video-editing instruction into JSON. Reply with JSON only.
{
  "operation": "replace" | "remove",
  "target": "<short noun phrase for the object to edit>",
  "detector_query": "<generic single noun for an object detector, e.g. bottle>",
  "replacement": "<what to put there, or null for remove>"
}"""


class PromptIn(BaseModel):
    prompt: str


def parse_instruction(prompt: str) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(resp.choices[0].message.content)


# -------------------------------------------------------------- reference

def load_reference(path: Path):
    """Load a reference image, derive an alpha channel, crop it tight."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError("the reference image could not be decoded")

    if raw.ndim == 3 and raw.shape[2] == 4:
        bgr, alpha = raw[..., :3], raw[..., 3]
    else:
        bgr = raw if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        background = (hsv[..., 2] > 235) & (hsv[..., 1] < 30)
        alpha = np.where(background, 0, 255).astype(np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Keep only the largest region. Drops reflections, shadows and JPEG
        # speckle that survive the colour test but are not part of the object.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            alpha = np.where(labels == biggest, 255, 0).astype(np.uint8)

        alpha = cv2.medianBlur(alpha, 5)

    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        raise RuntimeError("the reference image is entirely background")
    return (
        bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
        alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
    )


# ------------------------------------------------------------ orientation

def _data_url(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("could not encode an image for the vision call")
    return "data:image/png;base64," + base64.b64encode(buf).decode()


FLIP_SYSTEM = (
    "You compare two images of the same kind of object. The first is an object "
    "cropped from a video frame. The second is a replacement object, shown "
    "upright. Decide whether the replacement must be rotated 180 degrees so "
    "that its top points the same way as the first object's top. Reply with "
    'JSON only: {"flip": true} or {"flip": false}'
)


def decide_flip(frame_bgr, mask, ref_bgr, ref_alpha):
    """Ask a vision model which way round the replacement goes.

    Returns True/False, or None when the call fails so the caller can fall
    back to `geometric_flip`.
    """
    try:
        ys, xs = np.where(mask > 0)
        pad = 8
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad, frame_bgr.shape[0])
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad, frame_bgr.shape[1])
        target = frame_bgr[y0:y1, x0:x1]

        al = ref_alpha.astype(np.float32)[..., None] / 255.0
        white = np.full_like(ref_bgr, 255)
        ref_flat = (ref_bgr.astype(np.float32) * al +
                    white.astype(np.float32) * (1 - al)).astype(np.uint8)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": FLIP_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _data_url(target)}},
                        {"type": "image_url", "image_url": {"url": _data_url(ref_flat)}},
                    ],
                },
            ],
        )
        return bool(json.loads(resp.choices[0].message.content)["flip"])
    except Exception as e:
        print(f"orientation check failed, falling back to geometry: {e}")
        return None


def geometric_flip(mask, ref_alpha):
    """Fallback orientation guess from pixel mass.

    An asymmetric object carries less mass at its narrow end. Compares the
    mask's narrow end to the reference's. Unreliable for symmetric objects
    such as cans, where the two halves are near-equal -- which is why this is
    only the fallback for `decide_flip`.
    """
    ys = np.where(mask > 0)[0]
    mid = (int(ys.min()) + int(ys.max())) / 2
    mask_narrow_top = (ys < mid).sum() < (ys >= mid).sum()

    ref_ys = np.where(ref_alpha > 0)[0]
    ref_narrow_top = (ref_ys < ref_alpha.shape[0] / 2).sum() < (
        ref_ys >= ref_alpha.shape[0] / 2
    ).sum()

    return ref_narrow_top != mask_narrow_top


def warp_to_mask(ref_bgr, ref_alpha, mask, flip=False, pad=1.0, girth=1.0,
                 prev_long=None):
    """Rotate and scale the reference onto the mask's long axis, keeping its aspect.

    Returns (warped_bgr, warped_alpha, axis). Pass the returned axis back in as
    `prev_long` on the next frame -- see the direction note below.
    """
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(cv2.findNonZero(mask))
    a = np.deg2rad(ang)
    u = np.array([np.cos(a), np.sin(a)])
    v = np.array([-np.sin(a), np.cos(a)])

    d_long, l_long = (v, rh) if rh >= rw else (u, rw)

    # minAreaRect returns an axis, not a direction, and no fixed rule for
    # picking one is continuous over a full rotation -- any rule breaks
    # somewhere, and at that angle the replacement swaps end for end between
    # consecutive frames. So agree with the previous frame's choice where there
    # is one; only the first frame falls back to a fixed rule.
    if prev_long is not None:
        if float(np.dot(d_long, prev_long)) < 0:
            d_long = -d_long
    elif d_long[1] < 0 or (abs(d_long[1]) < 1e-9 and d_long[0] < 0):
        d_long = -d_long

    axis = d_long.copy()          # recorded before `flip`, so the chain is stable
    d_short = np.array([d_long[1], -d_long[0]])
    if flip:
        d_long, d_short = -d_long, -d_short

    l_long *= pad
    h, w = ref_alpha.shape[:2]
    scale = l_long / h
    half_long = l_long / 2
    half_short = (w * scale * girth) / 2

    c = np.array([cx, cy])
    dst = np.float32([
        c - half_short * d_short - half_long * d_long,
        c + half_short * d_short - half_long * d_long,
        c + half_short * d_short + half_long * d_long,
        c - half_short * d_short + half_long * d_long,
    ])

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    size = (mask.shape[1], mask.shape[0])
    return (
        cv2.warpPerspective(ref_bgr, M, size),
        cv2.warpPerspective(ref_alpha, M, size),
        axis,
    )


# ------------------------------------------------------------------ edits

def edit_frame(frame_bgr, mask, operation, ref, flip=False, girth=1.0,
               prev_long=None):
    """Apply the edit to one frame. Returns (frame, axis) -- thread the axis on."""
    if operation == "remove":
        dil = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        return cv2.inpaint(frame_bgr, dil, 5, cv2.INPAINT_TELEA), prev_long

    if ref is not None:
        dil = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        cleaned = cv2.inpaint(frame_bgr, dil, 5, cv2.INPAINT_TELEA)
        wref, walpha, axis = warp_to_mask(
            ref[0], ref[1], mask, flip=flip, girth=girth, prev_long=prev_long
        )
        al = cv2.GaussianBlur(walpha, (5, 5), 0).astype(np.float32)[..., None] / 255.0
        composed = (wref.astype(np.float32) * al +
                    cleaned.astype(np.float32) * (1 - al)).astype(np.uint8)
        return composed, axis

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    sel = mask > 0
    hsv[..., 0][sel] = (hsv[..., 0][sel].astype(int) + 90) % 180
    hsv[..., 1][sel] = np.clip(hsv[..., 1][sel].astype(int) + 40, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), prev_long


def encode(frames, fps, job_id):
    h, w = frames[0].shape[:2]
    raw = OUTPUT_DIR / f"{job_id}_raw.mp4"
    out = OUTPUT_DIR / f"{job_id}.mp4"

    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", str(out)],
            capture_output=True,
        )
    if not out.exists() or out.stat().st_size == 0:
        shutil.move(str(raw), str(out))
    else:
        raw.unlink(missing_ok=True)
    return out


# --------------------------------------------------------------- pipeline

def run_pipeline(job_id, prompt, video_path, ref_path, flip=False, girth=1.0):
    job = JOBS[job_id]
    try:
        job["stage"] = "parsing_instruction"
        plan = parse_instruction(prompt)
        job["plan"] = plan
        query = plan.get("detector_query") or plan.get("target") or "object"
        operation = plan.get("operation") or "replace"

        job["stage"] = "reading_video"
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        raw_frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            raw_frames.append(f)
        cap.release()
        if not raw_frames:
            raise RuntimeError("no frames could be decoded from that video")
        job["total_frames"] = len(raw_frames)

        ref = load_reference(ref_path) if ref_path else None

        job["stage"] = "tracking_object"
        edited, misses = [], 0
        resolved_flip = None
        prev_long = None

        for i, frame_bgr in enumerate(raw_frames):
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            mask, _ = mask_for_image(pil, query)

            if mask is None or not (mask > 0).any():
                misses += 1
                edited.append(frame_bgr)
            else:
                # Decide orientation once, on the first frame where the object
                # is found, then reuse it. Deciding per frame lets consecutive
                # frames disagree, which shows up as the object turning over
                # mid-clip.
                if resolved_flip is None:
                    if ref is None:
                        resolved_flip = flip
                    else:
                        decided = decide_flip(frame_bgr, mask, ref[0], ref[1])
                        if decided is None:
                            decided = geometric_flip(mask, ref[1])
                            job["orientation_source"] = "geometry"
                        else:
                            job["orientation_source"] = "vision"
                        resolved_flip = decided != flip
                        job["orientation_flipped"] = resolved_flip

                frame_out, prev_long = edit_frame(
                    frame_bgr, mask, operation, ref, resolved_flip, girth, prev_long
                )
                edited.append(frame_out)
            job["frames_done"] = i + 1

        job["misses"] = misses

        job["stage"] = "encoding_video"
        job["output"] = str(encode(edited, fps, job_id))
        job["stage"] = "done"

    except Exception as e:
        job["stage"] = "failed"
        job["error"] = str(e)
        print(traceback.format_exc())


# -------------------------------------------------------------------- API

@app.get("/health")
def health():
    return {"ok": True, "jobs": len(JOBS)}


@app.post("/jobs")
def create_job(
    background: BackgroundTasks,
    prompt: str = Form(...),
    video: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(None),
    flip: bool = Form(False),
    girth: float = Form(1.0),
):
    job_id = uuid.uuid4().hex[:8]

    video_path = UPLOAD_DIR / f"{job_id}.mp4"
    with video_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    ref_path = None
    if reference_image is not None and reference_image.filename:
        ref_path = REF_DIR / f"{job_id}.png"
        with ref_path.open("wb") as f:
            shutil.copyfileobj(reference_image.file, f)

    JOBS[job_id] = {
        "job_id": job_id,
        "stage": "queued",
        "prompt": prompt,
        "has_reference": ref_path is not None,
        "frames_done": 0,
        "total_frames": None,
    }
    background.add_task(
        run_pipeline, job_id, prompt, video_path, ref_path, flip, girth
    )
    return {"job_id": job_id, "stage": "queued"}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job id")
    return job


@app.get("/jobs/{job_id}/output")
def job_output(job_id: str):
    path = OUTPUT_DIR / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "no output for that job id")
    return FileResponse(path, media_type="video/mp4")


# ------------------------------------------------------------------ debug

@app.post("/parse")
def parse(body: PromptIn):
    return parse_instruction(body.prompt)


@app.post("/debug/upload")
def debug_upload(video: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:8]
    dest = UPLOAD_DIR / f"{job_id}.mp4"
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    return {"job_id": job_id, "size_bytes": dest.stat().st_size}


@app.get("/debug/reference/{job_id}")
def debug_reference(job_id: str):
    """Show what load_reference keeps, so a failed keying is visible at a glance."""
    path = REF_DIR / f"{job_id}.png"
    if not path.exists():
        raise HTTPException(404, "no reference image for that job id")
    bgr, alpha = load_reference(path)
    checker = np.full_like(bgr, 40)
    checker[::16, :] = 90
    checker[:, ::16] = 90
    al = alpha.astype(np.float32)[..., None] / 255.0
    composed = (bgr.astype(np.float32) * al + checker.astype(np.float32) * (1 - al))
    dest = REF_DIR / f"{job_id}_keyed.png"
    cv2.imwrite(str(dest), composed.astype(np.uint8))
    return FileResponse(dest)


@app.post("/debug/{job_id}/masks")
def debug_masks(job_id: str, query: str, threshold: float = 0.1):
    cap = cv2.VideoCapture(str(UPLOAD_DIR / f"{job_id}.mp4"))
    results, i, misses = [], 0, 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        cv2.imwrite(str(FRAME_DIR / f"{job_id}_{i}.jpg"), frame_bgr)
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mask, best = mask_for_image(pil, query, threshold)
        if mask is None:
            misses += 1
            results.append({"index": i, "found": False})
        else:
            cv2.imwrite(str(FRAME_DIR / f"{job_id}_{i}_mask.png"), mask)
            results.append({
                "index": i,
                "found": True,
                "score": round(float(best["score"]), 3),
                "pixels": int((mask > 0).sum()),
            })
        i += 1
    cap.release()
    return {"job_id": job_id, "frames": i, "misses": misses, "per_frame": results}


@app.get("/debug/{job_id}/sheet")
def debug_sheet(job_id: str, cols: int = 6, tile_w: int = 160):
    tiles, i = [], 0
    while True:
        fp = FRAME_DIR / f"{job_id}_{i}.jpg"
        if not fp.exists():
            break
        img = cv2.imread(str(fp))
        mp = FRAME_DIR / f"{job_id}_{i}_mask.png"
        if mp.exists():
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            overlay = img.copy()
            overlay[mask > 0] = (0, 0, 255)
            img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        else:
            cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 255), 3)
        h = int(img.shape[0] * tile_w / img.shape[1])
        img = cv2.resize(img, (tile_w, h))
        cv2.putText(img, str(i), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        tiles.append(img)
        i += 1

    if not tiles:
        raise HTTPException(404, "no frames for that job id")

    th, tw = tiles[0].shape[:2]
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros((th, tw, 3), np.uint8))
        rows.append(np.hstack(row))

    dest = FRAME_DIR / f"{job_id}_sheet.jpg"
    cv2.imwrite(str(dest), np.vstack(rows))
    return FileResponse(dest)