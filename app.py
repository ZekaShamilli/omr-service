import json
import math
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

# ── Load template ────────────────────────────────────────────────────────────

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "template.json")
with open(_TEMPLATE_PATH) as f:
    _T = json.load(f)

PPM    = _T["ppm"]
NORM_W = _T["norm_w"]
NORM_H = _T["norm_h"]

MARKERS_MM   = _T["markers_mm"]
MARKER_SIZE  = _T["marker_size_mm"]

NUMARA  = {k.replace("_mm", ""): v for k, v in _T["numara"].items()}
VARIANT = {k.replace("_mm", ""): v for k, v in _T["variant"].items()}
GRID    = {k.replace("_mm", ""): v for k, v in _T["grid"].items()}

MIN_FILL_RATIO   = _T["analysis"]["min_fill_ratio"]
DOMINANCE_FACTOR = _T["analysis"]["dominance_factor"]

# ── Geometry helpers ─────────────────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
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

    Uses the LARGEST connected dark component in each zone so that thin bubble
    circle borders (which also appear dark) don't corrupt the centroid.
    """
    h, w = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ZONES = [
        (0,    0,    0.25, 0.27),   # TL
        (0.75, 0,    1.0,  0.27),   # TR
        (0,    0.35, 0.25, 0.65),   # ML
        (0.75, 0.35, 1.0,  0.65),   # MR
        (0,    0.73, 0.25, 1.0 ),   # BL
        (0.75, 0.73, 1.0,  1.0 ),   # BR
    ]
    MIN_AREA_RATIO = 0.005   # largest blob must be ≥ 0.5% of zone area

    pts = []
    for (fx0, fy0, fx1, fy1) in ZONES:
        x0, x1 = int(fx0 * w), int(fx1 * w)
        y0, y1 = int(fy0 * h), int(fy1 * h)
        roi = binary[y0:y1, x0:x1]

        # Find connected components; label 0 = background
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
        if n_labels < 2:
            return None

        # Pick the largest non-background component
        areas = stats[1:, cv2.CC_STAT_AREA]
        best  = int(np.argmax(areas)) + 1   # shift back to label index
        if areas[best - 1] < (x1 - x0) * (y1 - y0) * MIN_AREA_RATIO:
            return None

        cx = float(centroids[best][0]) + x0
        cy = float(centroids[best][1]) + y0
        pts.append([cx, cy])

    return pts   # [TL, TR, ML, MR, BL, BR]


def detect_page_corners(gray: np.ndarray):
    """Fallback: largest quadrilateral contour (Canny edge detection)."""
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    edged    = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    img_area = gray.shape[0] * gray.shape[1]
    for c in contours:
        if cv2.contourArea(c) < img_area * 0.1:
            continue
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
    return None


# ── Perspective correction ───────────────────────────────────────────────────

# Expected pixel positions of each corner marker's CENTER in the normalised canvas.
# Marker corner positions (mm) + half marker size → centre in mm → × PPM = px.
# This ensures that after warping, any coordinate computed as x_mm * PPM lines up
# exactly with the rendered sheet layout.
_HALF = MARKER_SIZE / 2 * PPM  # 7.5 mm × 3 px/mm = 22.5 px
_MARKER_DST = {
    k: (v[0] * PPM + _HALF, v[1] * PPM + _HALF)
    for k, v in MARKERS_MM.items()
}

def perspective_correct(gray: np.ndarray):
    # Primary: use all 6 detected marker centres with RANSAC homography.
    # More constraints → more accurate warp, robust against one bad detection.
    markers = detect_marker_centers(gray)
    if markers is not None:
        # markers order: TL TR ML MR BL BR
        key_order = ["TL", "TR", "ML", "MR", "BL", "BR"]
        src_pts = np.array(markers, dtype="float32")
        dst_pts = np.array([_MARKER_DST[k] for k in key_order], dtype="float32")
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            return cv2.warpPerspective(gray, M, (NORM_W, NORM_H))

    # Fallback: Canny finds page outline → corners map to canvas corners.
    corners = detect_page_corners(gray)
    if corners is not None:
        src = order_points(corners)
        dst = np.array([[0, 0], [NORM_W, 0], [NORM_W, NORM_H], [0, NORM_H]], dtype="float32")
        M   = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(gray, M, (NORM_W, NORM_H))

    return None


# ── Bubble fill ratio ────────────────────────────────────────────────────────

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


# ── OMRChecker-style relative comparison ─────────────────────────────────────

def pick_dominant(fills: list[float]) -> int:
    """Return index of the dominant bubble, or -1 if ambiguous/empty.

    Requires: max > MIN_FILL_RATIO  AND  max > second_max * DOMINANCE_FACTOR
    This mirrors OMRChecker's relative comparison so lightly-marked or
    double-marked bubbles don't produce false answers.
    """
    if not fills:
        return -1
    sorted_fills = sorted(fills, reverse=True)
    best_val  = sorted_fills[0]
    second    = sorted_fills[1] if len(sorted_fills) > 1 else 0.0

    if best_val < MIN_FILL_RATIO:
        return -1
    if second > 0 and best_val < second * DOMINANCE_FACTOR:
        return -1
    return fills.index(best_val)


# ── Sheet analysis ───────────────────────────────────────────────────────────

def analyze(norm: np.ndarray, num_questions: int, num_options: int, num_variants: int):
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

        fills = []
        for oi in range(num_options):
            cx = colX + g["numW"] + oi * g["bubbleSpacing"] + g["bubbleSpacing"] / 2
            fills.append(fill_ratio(norm, cx, cy, g["bubbleR"], thr))

        idx = pick_dominant(fills)
        answers[f"q{q + 1}"] = opts[idx] if idx >= 0 else ""

    # Student PIN
    # <thead> has height:4mm set explicitly → it does render; <td> rows inflate to ~6mm from bubble content
    PIN_HEADER_H = n["headerH"]
    PIN_ROW_H    = n["rowH"]
    pin_digits = []
    for col in range(n["numCols"]):
        cx = n["tableX"] + n["labelColW"] + col * n["digitColW"] + n["digitColW"] / 2
        fills = []
        for d in range(n["numRows"]):
            cy = n["tableY"] + PIN_HEADER_H + d * PIN_ROW_H + PIN_ROW_H / 2
            fills.append(fill_ratio(norm, cx, cy, n["bubbleR"], thr))
        idx = pick_dominant(fills)
        pin_digits.append(idx)   # -1 = undetected digit
    pin = "".join(str(d) for d in pin_digits) if all(d >= 0 for d in pin_digits) else ""

    # Variant (Grup)
    fills = []
    for vi in range(num_variants):
        cx = v["x"] + v["labelW"] + 2 + v["bubbleR"]
        cy = v["y"] + vi * v["rowH"] + v["rowH"] / 2
        fills.append(fill_ratio(norm, cx, cy, v["bubbleR"], thr))
    variant = pick_dominant(fills)

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


@app.post("/omr/debug")
async def debug_omr(
    image:         UploadFile = File(...),
    num_questions: int        = Form(...),
    num_options:   int        = Form(5),
    num_variants:  int        = Form(1),
):
    """Returns the warped sheet as a PNG with all expected bubble centres drawn.
    Use this to verify that perspective correction and coordinates are correct."""
    import base64

    contents = await image.read()
    arr  = np.frombuffer(contents, np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return JSONResponse({"error": "Resim okunamadı"}, status_code=400)

    # Show which detection path was used
    markers = detect_marker_centers(gray)
    detection_path = "markers" if markers is not None else "canny"

    norm = perspective_correct(gray)
    if norm is None:
        return JSONResponse({"error": "Sayfa sınırları algılanamadı", "detection": "failed"}, status_code=422)

    # Draw on a colour copy
    vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    g = GRID;  n = NUMARA;  v = VARIANT
    col1 = math.ceil(num_questions / 2)

    def px(mm_val): return int(round(mm_val * PPM))

    # Answer bubbles — green filled dot
    for q in range(num_questions):
        is2  = q >= col1
        ri   = q - col1 if is2 else q
        colX = g["col2X"] if is2 else g["col1X"]
        cy   = g["startY"] + ri * g["rowH"] + g["rowH"] / 2
        for oi in range(num_options):
            cx = colX + g["numW"] + oi * g["bubbleSpacing"] + g["bubbleSpacing"] / 2
            cv2.circle(vis, (px(cx), px(cy)), 3, (0, 180, 0), -1)

    # PIN bubbles — blue filled dot
    PIN_HEADER_H = n["headerH"]
    PIN_ROW_H    = n["rowH"]
    for col in range(n["numCols"]):
        cx = n["tableX"] + n["labelColW"] + col * n["digitColW"] + n["digitColW"] / 2
        for d in range(n["numRows"]):
            cy = n["tableY"] + PIN_HEADER_H + d * PIN_ROW_H + PIN_ROW_H / 2
            cv2.circle(vis, (px(cx), px(cy)), 3, (220, 0, 0), -1)

    # Variant bubbles — red filled dot
    for vi in range(num_variants):
        cx = v["x"] + v["labelW"] + 2 + v["bubbleR"]
        cy = v["y"] + vi * v["rowH"] + v["rowH"] / 2
        cv2.circle(vis, (px(cx), px(cy)), 3, (0, 0, 220), -1)

    _, buf = cv2.imencode(".png", vis)
    b64 = base64.b64encode(buf).decode()
    return {"detection_path": detection_path, "image_base64": b64}
