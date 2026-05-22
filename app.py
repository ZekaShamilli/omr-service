import json
import math

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

# ── Sheet layout (mm) — must match SHEET_LAYOUT in sheet-layout.ts ──────────

PPM    = 3       # pixels per mm in normalised canvas
NORM_W = 630     # 210 mm × 3
NORM_H = 891     # 297 mm × 3

NUMARA = dict(tableX=26, tableY=43, headerH=4, rowH=5.5,
              labelColW=5, digitColW=7, numCols=5, numRows=10, bubbleR=2)
VARIANT = dict(x=74, y=43, rowH=8, labelW=4, bubbleR=2.2)
GRID    = dict(startY=112, rowH=6, col1X=26, col2X=110,
               numW=7, bubbleSpacing=7, bubbleR=2.4)

MARKERS_MM = dict(
    TL=(8,   8),
    TR=(187, 8),
    ML=(8,   141),
    MR=(187, 141),
    BL=(8,   274),
    BR=(187, 274),
)
MARKER_SIZE = 15

FILL_THRESHOLD = 0.22

# ── Geometry helpers ─────────────────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    """Return [TL, TR, BR, BL] order."""
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_page_corners(gray: np.ndarray):
    """Find answer sheet corners via largest quadrilateral contour (OMRChecker style)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged   = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
    return None


def detect_marker_centers(gray: np.ndarray):
    """Fallback: detect 6 fiducial squares and return [TL,TR,ML,MR,BL,BR] centres."""
    h, w = gray.shape
    ZONES = [
        (0,    0,    0.25, 0.25),
        (0.75, 0,    1.0,  0.25),
        (0,    0.35, 0.25, 0.65),
        (0.75, 0.35, 1.0,  0.65),
        (0,    0.75, 0.25, 1.0 ),
        (0.75, 0.75, 1.0,  1.0 ),
    ]
    DARK      = 80
    MIN_RATIO = 0.025
    pts = []
    for (fx0, fy0, fx1, fy1) in ZONES:
        x0, x1 = int(fx0 * w), int(fx1 * w)
        y0, y1 = int(fy0 * h), int(fy1 * h)
        region  = gray[y0:y1, x0:x1]
        dark    = region < DARK
        count   = int(dark.sum())
        if count / ((x1 - x0) * (y1 - y0)) < MIN_RATIO:
            return None
        ys, xs = np.where(dark)
        pts.append([float(xs.mean()) + x0, float(ys.mean()) + y0])
    return pts   # [TL, TR, ML, MR, BL, BR]


# ── Perspective correction ───────────────────────────────────────────────────

def perspective_correct(gray: np.ndarray):
    """Warp image to NORM_W × NORM_H using page outline or marker fallback."""
    dst = np.array([[0, 0], [NORM_W, 0], [NORM_W, NORM_H], [0, NORM_H]], dtype="float32")

    # Primary: page outline (works even if markers are cropped)
    corners = detect_page_corners(gray)
    if corners is not None:
        src = order_points(corners)
        M   = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(gray, M, (NORM_W, NORM_H))

    # Fallback: fiducial markers (TL, TR, BL, BR)
    markers = detect_marker_centers(gray)
    if markers is None:
        return None
    half = MARKER_SIZE / 2
    tl = [markers[0][0] + half * gray.shape[1] / NORM_W,
          markers[0][1] + half * gray.shape[0] / NORM_H]  # rough centre
    # simpler: just use the averaged dark-pixel centroids as corners
    src = order_points(np.array([
        markers[0], markers[1], markers[5], markers[4]
    ], dtype="float32"))
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (NORM_W, NORM_H))


# ── Bubble fill ratio (vectorised) ───────────────────────────────────────────

def fill_ratio(img: np.ndarray, cx_mm: float, cy_mm: float, r_mm: float, thr: int) -> float:
    cx = cx_mm * PPM
    cy = cy_mm * PPM
    r  = r_mm  * PPM * 0.85

    x0 = max(0, int(cx - r));  x1 = min(img.shape[1] - 1, int(cx + r))
    y0 = max(0, int(cy - r));  y1 = min(img.shape[0] - 1, int(cy + r))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    mask   = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
    pixels = img[y0:y1 + 1, x0:x1 + 1][mask]
    return float(np.sum(pixels < thr)) / len(pixels) if len(pixels) else 0.0


# ── Sheet analysis ───────────────────────────────────────────────────────────

def analyze(norm: np.ndarray, num_questions: int, num_options: int, num_variants: int):
    # Dynamic threshold: 65 % of median sample brightness
    flat   = norm.flatten()[::max(1, len(norm.flatten()) // 500)]
    flat.sort()
    thr    = int(np.clip(flat[len(flat) // 2] * 0.65, 60, 200))

    g = GRID;  n = NUMARA;  v = VARIANT
    opts = list("ABCDE"[:num_options])
    col1 = math.ceil(num_questions / 2)

    # Answers
    answers: dict[str, str] = {}
    for q in range(num_questions):
        is2  = q >= col1
        ri   = q - col1 if is2 else q
        colX = g["col2X"] if is2 else g["col1X"]
        cy   = g["startY"] + ri * g["rowH"] + g["rowH"] / 2
        best, best_f = "", FILL_THRESHOLD
        for oi, opt in enumerate(opts):
            cx = colX + g["numW"] + oi * g["bubbleSpacing"] + g["bubbleSpacing"] / 2
            f  = fill_ratio(norm, cx, cy, g["bubbleR"], thr)
            if f > best_f:
                best_f, best = f, opt
        answers[f"q{q + 1}"] = best

    # Student PIN
    pin_digits = []
    for col in range(n["numCols"]):
        cx = n["tableX"] + n["labelColW"] + col * n["digitColW"] + n["digitColW"] / 2
        bd, bf = -1, FILL_THRESHOLD
        for d in range(n["numRows"]):
            cy = n["tableY"] + n["headerH"] + d * n["rowH"] + n["rowH"] / 2
            f  = fill_ratio(norm, cx, cy, n["bubbleR"], thr)
            if f > bf:
                bf, bd = f, d
        pin_digits.append(bd)
    pin = "".join(str(d) for d in pin_digits) if all(d >= 0 for d in pin_digits) else ""

    # Variant (Grup)
    variant, vf = -1, FILL_THRESHOLD
    for vi in range(num_variants):
        cx = v["x"] + v["labelW"] + v["bubbleR"]
        cy = v["y"] + vi * v["rowH"] + v["rowH"] / 2
        f  = fill_ratio(norm, cx, cy, v["bubbleR"], thr)
        if f > vf:
            vf, variant = f, vi

    return answers, pin, variant


# ── Endpoint ─────────────────────────────────────────────────────────────────

@app.post("/omr")
async def process_omr(
    image:         UploadFile = File(...),
    num_questions: int        = Form(...),
    num_options:   int        = Form(5),
    num_variants:  int        = Form(1),
    answer_key:    str        = Form(""),
):
    contents = await image.read()
    arr  = np.frombuffer(contents, np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        return JSONResponse({"error": "Resim okunamadı"}, status_code=400)

    norm = perspective_correct(gray)
    if norm is None:
        return JSONResponse(
            {"error": "Sayfa sınırları algılanamadı. Kağıdın tamamının görünür olduğundan emin olun."},
            status_code=422,
        )

    # CLAHE — equalises contrast under varying lighting (OMRChecker's key trick)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm  = clahe.apply(norm)

    answers, pin, variant = analyze(norm, num_questions, num_options, num_variants)

    key        = json.loads(answer_key) if answer_key else []
    ans_arr    = [answers.get(f"q{i + 1}", "") for i in range(num_questions)]
    correct    = sum(1 for i, a in enumerate(ans_arr) if key and i < len(key) and a == key[i])
    score_pct  = round(correct / num_questions * 100) if key and num_questions > 0 else None

    return {
        "answers":   answers,
        "pin":       pin,
        "variant":   variant,
        "correct":   correct if key else None,
        "score_pct": score_pct,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
