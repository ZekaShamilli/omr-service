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

NUMARA = dict(tableX=26, tableY=50, headerH=4, rowH=5.5,
              labelColW=5, digitColW=7, numCols=5, numRows=10, bubbleR=2)
VARIANT = dict(x=74, y=50, rowH=8, labelW=4, bubbleR=2.2)
GRID    = dict(startY=142, rowH=6, col1X=26, col2X=110,
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


def detect_marker_centers(gray: np.ndarray):
    """Detect 6 fiducial squares → [TL, TR, ML, MR, BL, BR] pixel centres.

    Uses Otsu threshold so it works under varying lighting. Each zone must
    contain a blob that is roughly square and large enough to be the marker.
    Returns None if any zone fails detection.
    """
    h, w = gray.shape
    # Otsu global threshold (works well for dark squares on white paper)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ZONES = [
        (0,    0,    0.25, 0.27),   # TL
        (0.75, 0,    1.0,  0.27),   # TR
        (0,    0.35, 0.25, 0.65),   # ML
        (0.75, 0.35, 1.0,  0.65),   # MR
        (0,    0.73, 0.25, 1.0 ),   # BL
        (0.75, 0.73, 1.0,  1.0 ),   # BR
    ]
    MIN_FILL = 0.015   # at least 1.5 % of zone must be dark

    pts = []
    for (fx0, fy0, fx1, fy1) in ZONES:
        x0, x1 = int(fx0 * w), int(fx1 * w)
        y0, y1 = int(fy0 * h), int(fy1 * h)
        roi = binary[y0:y1, x0:x1]
        fill = roi.mean() / 255.0
        if fill < MIN_FILL:
            return None
        ys, xs = np.where(roi > 0)
        pts.append([float(xs.mean()) + x0, float(ys.mean()) + y0])
    return pts   # [TL, TR, ML, MR, BL, BR]


def detect_page_corners(gray: np.ndarray):
    """Fallback: largest quadrilateral contour (Canny edge detection)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged   = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    img_area = gray.shape[0] * gray.shape[1]
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.1:   # ignore tiny contours
            continue
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
    return None


# ── Perspective correction ───────────────────────────────────────────────────

def perspective_correct(gray: np.ndarray):
    """Warp image to NORM_W × NORM_H.

    Primary path: detect all 6 fiducial markers → use 4 corner centres.
    Fallback: Canny page-outline detection.
    Marker path is preferred because it is immune to table edges / backgrounds.
    """
    dst = np.array([[0, 0], [NORM_W, 0], [NORM_W, NORM_H], [0, NORM_H]], dtype="float32")

    # Primary: fiducial marker centres (robust against background clutter)
    markers = detect_marker_centers(gray)
    if markers is not None:
        src = order_points(np.array([
            markers[0], markers[1], markers[5], markers[4]  # TL, TR, BR, BL
        ], dtype="float32"))
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(gray, M, (NORM_W, NORM_H))

    # Fallback: page outline via Canny edges
    corners = detect_page_corners(gray)
    if corners is not None:
        src = order_points(corners)
        M   = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(gray, M, (NORM_W, NORM_H))

    return None


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
    # Otsu threshold on the normalised sheet — works under any lighting
    thr_val, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(np.clip(thr_val * 0.85, 60, 220))

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
