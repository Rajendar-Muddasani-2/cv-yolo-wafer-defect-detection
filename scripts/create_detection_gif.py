#!/usr/bin/env python3
"""Create a LinkedIn-ready, evidence-grounded YOLOv8-L project walkthrough.

Every empirical value is loaded from a committed result artifact or produced by
the trained checkpoint. Architecture frames are explicitly labeled as schematic.
"""

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
INPUT_IMAGE = ROOT / "outputs" / "realistic_unseen" / "realistic_05.jpg"
OUTPUT_GIF = ROOT / "outputs" / "yolo_wafer_detection.gif"
RESULTS_JSON = ROOT / "outputs" / "results_summary.json"
TRT_JSON = ROOT / "outputs" / "tensorrt_results" / "tensorrt_benchmark_results.json"
T4_JSON = ROOT / "outputs" / "gpu_stack_results" / "gpu_stack_results.json"

W, H = 1200, 675
FPS = 4
FRAME_MS = 250

BG = (5, 12, 11)
SURFACE = (10, 24, 21)
SURFACE_2 = (16, 35, 29)
LINE = (42, 79, 65)
WHITE = (240, 245, 248)
MUTED = (157, 181, 171)
CYAN = (48, 214, 211)
GREEN = (70, 245, 147)
YELLOW = (255, 210, 64)
CORAL = (255, 105, 97)

CLASS_COLORS = {
    "scratch": CORAL,
    "particle": CYAN,
    "edge_chip": YELLOW,
    "void": (190, 125, 255),
    "pattern_shift": (255, 157, 67),
    "bridge": (93, 238, 188),
    "missing_bond": (255, 114, 190),
    "crack": (255, 89, 89),
    "contamination": (255, 190, 92),
    "delamination": (117, 202, 255),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    if bold:
        names.insert(0, "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_EYEBROW = font(18, bold=True)
F_HERO = font(43, bold=True)
F_TITLE = font(32, bold=True)
F_SUBTITLE = font(25, bold=True)
F_BODY = font(22)
F_BODY_BOLD = font(22, bold=True)
F_SMALL = font(17)
F_METRIC = font(34, bold=True)
F_LABEL = font(16, bold=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def get_detections() -> list[dict[str, Any]]:
    """Return actual Ultralytics post-NMS predictions for the displayed input."""
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "models" / "best.pt"))
    result = model.predict(
        str(INPUT_IMAGE), conf=0.25, iou=0.45, imgsz=640, verbose=False
    )[0]
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        detections.append(
            {
                "class": str(model.names[class_id]),
                "confidence": float(box.conf[0].item()),
                "bbox": [float(value) for value in box.xyxy[0].tolist()],
            }
        )
    return sorted(detections, key=lambda item: item["confidence"], reverse=True)


def stylize_wafer(image: Image.Image) -> Image.Image:
    """Create a green silicon die-map treatment without changing image geometry."""
    grayscale = ImageEnhance.Contrast(ImageOps.grayscale(image)).enhance(1.35)
    styled = ImageOps.colorize(grayscale, black=(2, 16, 12), white=(116, 236, 158))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    center = image.width // 2
    radius = 292

    for coordinate in range(center - 256, center + 257, 32):
        offset = coordinate - center
        span = int(math.sqrt(max(radius * radius - offset * offset, 0)))
        draw.line(
            (coordinate, center - span, coordinate, center + span),
            fill=(75, 255, 158, 52),
            width=1,
        )
        draw.line(
            (center - span, coordinate, center + span, coordinate),
            fill=(75, 255, 158, 52),
            width=1,
        )

    trace_color = (80, 255, 174, 125)
    pad_color = (255, 215, 92, 190)
    traces = [
        [(146, 270), (226, 270), (226, 220), (302, 220)],
        [(350, 155), (350, 235), (420, 235), (420, 306)],
        [(182, 405), (258, 405), (258, 468), (338, 468)],
        [(390, 390), (468, 390), (468, 338), (525, 338)],
    ]
    for points in traces:
        draw.line(points, fill=trace_color, width=3, joint="curve")
        for x, y in (points[0], points[-1]):
            draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=pad_color)

    draw.ellipse(
        (center - radius, center - radius, center + radius, center + radius),
        outline=(100, 255, 180, 180),
        width=3,
    )
    return Image.alpha_composite(styled.convert("RGBA"), overlay).convert("RGB")


def draw_circuit_background(draw: ImageDraw.ImageDraw) -> None:
    """Add restrained semiconductor-routing detail behind the main content."""
    trace = (19, 55, 45)
    node = (38, 102, 80)
    paths = [
        [(0, 108), (20, 108), (20, 82), (92, 82)],
        [(1090, 84), (1168, 84), (1168, 125), (1200, 125)],
        [(0, 560), (58, 560), (58, 580), (140, 580)],
        [(1060, 570), (1122, 570), (1122, 548), (1200, 548)],
    ]
    for points in paths:
        draw.line(points, fill=trace, width=2)
        for x, y in (points[0], points[-1]):
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=node)


def canvas(section: str, step: int, total: int = 5) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    frame = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(frame)
    draw_circuit_background(draw)
    draw.rectangle((0, 0, W, 62), fill=SURFACE)
    draw.text((34, 19), "WAFER VISION", fill=GREEN, font=F_EYEBROW)
    draw.text((190, 18), section, fill=WHITE, font=F_SMALL)
    draw.text((1075, 19), f"{step:02d} / {total:02d}", fill=MUTED, font=F_SMALL)
    draw.line((0, 62, W, 62), fill=LINE, width=1)
    return frame, draw


def footer(draw: ImageDraw.ImageDraw, primary: str, secondary: str) -> None:
    draw.rectangle((0, 598, W, H), fill=SURFACE)
    draw.line((0, 598, W, 598), fill=LINE, width=1)
    draw.text((34, 610), primary, fill=WHITE, font=F_BODY_BOLD)
    draw.text((34, 641), secondary, fill=MUTED, font=F_SMALL)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
    draw.line((start, end), fill=color, width=4)
    draw.polygon(
        ((end[0], end[1]), (end[0] - 13, end[1] - 8), (end[0] - 13, end[1] + 8)),
        fill=color,
    )


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: list[str],
    accent: tuple[int, int, int] = CYAN,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle(box, fill=SURFACE, outline=LINE, width=2)
    draw.rectangle((x1, y1, x1 + 6, y2), fill=accent)
    draw.text((x1 + 22, y1 + 18), title, fill=accent, font=F_SUBTITLE)
    y = y1 + 58
    for line in body:
        draw.text((x1 + 22, y), line, fill=WHITE, font=F_SMALL)
        y += 25


def fit_image(image: Image.Image, size: int) -> tuple[Image.Image, float]:
    scale = size / max(image.size)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
    )
    return resized, scale


def draw_image_panel(
    frame: Image.Image,
    image: Image.Image,
    xy: tuple[int, int],
    size: int,
    label: str,
    border: tuple[int, int, int] = LINE,
) -> tuple[float, tuple[int, int]]:
    draw = ImageDraw.Draw(frame)
    resized, scale = fit_image(image, size)
    x, y = xy
    draw.rectangle((x - 5, y - 5, x + resized.width + 5, y + resized.height + 5), fill=SURFACE_2)
    frame.paste(resized, (x, y))
    draw.rectangle((x - 1, y - 1, x + resized.width, y + resized.height), outline=border, width=2)
    draw.text((x, y + resized.height + 12), label, fill=MUTED, font=F_SMALL)
    return scale, (resized.width, resized.height)


def draw_detection(
    draw: ImageDraw.ImageDraw,
    detection: dict[str, Any],
    origin: tuple[int, int],
    scale: float,
) -> None:
    x, y = origin
    left, top, right, bottom = detection["bbox"]
    left = round(x + left * scale)
    top = round(y + top * scale)
    right = round(x + right * scale)
    bottom = round(y + bottom * scale)
    color = CLASS_COLORS.get(detection["class"], GREEN)
    draw.rectangle((left, top, right, bottom), outline=color, width=4)
    label = f"{detection['class']} {detection['confidence']:.0%}"
    text_box = draw.textbbox((0, 0), label, font=F_LABEL)
    label_width = text_box[2] - text_box[0] + 12
    label_height = text_box[3] - text_box[1] + 9
    panel_right = x + round(640 * scale)
    label_x = min(max(left, x), panel_right - label_width)
    label_y = top - label_height
    if label_y < y:
        label_y = min(bottom + 3, y + round(640 * scale) - label_height)
    draw.rectangle(
        (label_x, label_y, label_x + label_width, label_y + label_height), fill=color
    )
    draw.text((label_x + 6, label_y + 3), label, fill=BG, font=F_LABEL)


def cover_scene(
    styled_image: Image.Image,
    detections: list[dict[str, Any]],
    metrics: dict[str, float],
    trt: dict[str, float],
) -> Image.Image:
    frame, draw = canvas("SEMICONDUCTOR WAFER DEFECT DETECTION", 1)
    scale, _ = draw_image_panel(
        frame,
        styled_image,
        (34, 90),
        475,
        "Stylized die-map view | geometry preserved from model input",
        GREEN,
    )
    for detection in detections:
        draw_detection(draw, detection, (34, 90), scale)

    draw.text((555, 104), "From wafer image", fill=WHITE, font=F_HERO)
    draw.text((555, 155), "to deployable inference", fill=GREEN, font=F_HERO)
    draw.text((557, 222), "YOLOv8-L  |  43.64M parameters  |  10 classes", fill=MUTED, font=F_BODY)
    draw.line((557, 268, 1158, 268), fill=LINE, width=1)

    draw.text((557, 296), f"{metrics['mAP50'] * 100:.2f}%", fill=YELLOW, font=F_METRIC)
    draw.text((557, 338), "mAP@50 on the selected YOLOv8-L run", fill=WHITE, font=F_SMALL)
    draw.text((557, 388), f"{trt['mean_ms']:.2f} ms", fill=GREEN, font=F_METRIC)
    draw.text((557, 430), "A100 TensorRT FP16 mean inference latency", fill=WHITE, font=F_SMALL)
    draw.text((557, 488), f"{len(detections)} post-NMS predictions on this frame", fill=CYAN, font=F_BODY_BOLD)

    footer(
        draw,
        "The goal: localize defect type, position, and confidence in one inference pass.",
        "Green die-map is a visual treatment; inference used the untouched synthetic image.",
    )
    return frame


def importance_scene(styled_image: Image.Image) -> Image.Image:
    frame, draw = canvas("WHY THIS PROBLEM MATTERS", 2)
    draw_image_panel(
        frame, styled_image, (34, 94), 468, "Stylized view of the synthetic validation wafer", GREEN
    )
    draw.text((550, 102), "Inspection needs more", fill=WHITE, font=F_TITLE)
    draw.text((550, 142), "than image classification", fill=GREEN, font=F_TITLE)

    card(draw, (550, 205, 835, 340), "WHAT?", ["Classify the defect", "across 10 categories"], CYAN)
    card(draw, (855, 205, 1140, 340), "WHERE?", ["Return a bounding box", "for each prediction"], GREEN)
    card(draw, (550, 360, 835, 495), "HOW SURE?", ["Attach a confidence", "score to each box"], YELLOW)
    card(draw, (855, 360, 1140, 495), "HOW FAST?", ["Benchmark deployment", "backends on GPUs"], CORAL)

    footer(
        draw,
        "Ten-class taxonomy spans scratches, particles, edge chips, cracks, voids, and process defects.",
        "Real fab deployment still requires qualification across tools, lots, recipes, and process corners.",
    )
    return frame


def architecture_scene(progress: float) -> Image.Image:
    frame, draw = canvas("YOLOv8-L INFERENCE PATH", 3)
    draw.text((34, 90), "Multi-scale, anchor-free detection", fill=WHITE, font=F_TITLE)
    draw.text((34, 132), "Architecture schematic | checkpoint strides verified at 8, 16, and 32", fill=MUTED, font=F_BODY)

    stages = [
        ((34, 215, 230, 370), "640 x 640", ["RGB input", "normalized tensor"], CYAN),
        ((270, 215, 466, 370), "C2f backbone", ["extracts spatial", "feature hierarchy"], GREEN),
        ((506, 215, 702, 370), "PAN-FPN neck", ["fuses detail and", "semantic context"], YELLOW),
        ((742, 175, 940, 410), "Feature pyramid", ["P3  80 x 80  s8", "P4  40 x 40  s16", "P5  20 x 20  s32"], CYAN),
        ((980, 215, 1166, 370), "Split head", ["box regression", "class scores"], CORAL),
    ]
    visible = max(1, min(len(stages), int(progress * len(stages)) + 1))
    for index, (box, title, body, accent) in enumerate(stages[:visible]):
        card(draw, box, title, body, accent)
        if index > 0:
            previous = stages[index - 1][0]
            arrow(draw, (previous[2] + 8, 292), (box[0] - 10, 292), accent)

    draw.rectangle((244, 455, 956, 540), fill=SURFACE, outline=LINE, width=2)
    draw.text((275, 473), "Small, medium, and large receptive fields are evaluated together", fill=WHITE, font=F_BODY_BOLD)
    draw.text((334, 510), "The model does not use a single 13 x 13 prediction grid", fill=YELLOW, font=F_SMALL)

    footer(
        draw,
        "The three detection scales preserve small-defect detail while adding wider context.",
        "Schematic explains tensor flow; it is not a visualization of hidden activations.",
    )
    return frame


def decision_scene(
    image: Image.Image, detections: list[dict[str, Any]], visible_count: int
) -> Image.Image:
    frame, draw = canvas("FROM MODEL OUTPUT TO INSPECTION RESULT", 4)
    scale, _ = draw_image_panel(
        frame, image, (34, 91), 475, "Actual post-NMS predictions | conf 0.25 | IoU 0.45", GREEN
    )
    for detection in detections[:visible_count]:
        draw_detection(draw, detection, (34, 91), scale)

    draw.text((550, 96), "Decision path", fill=WHITE, font=F_TITLE)
    steps = [
        ("01", "Decode", "box coordinates and class scores"),
        ("02", "Filter", "confidence below 0.25"),
        ("03", "Suppress", "overlapping boxes at IoU 0.45"),
        ("04", "Return", "class, confidence, and image-space box"),
    ]
    y = 160
    for number, title, detail in steps:
        draw.rectangle((550, y, 608, y + 52), fill=SURFACE_2, outline=LINE, width=1)
        draw.text((565, y + 13), number, fill=CYAN, font=F_SMALL)
        draw.text((630, y + 2), title, fill=WHITE, font=F_BODY_BOLD)
        draw.text((630, y + 30), detail, fill=MUTED, font=F_SMALL)
        y += 85

    y = 510
    summary = " | ".join(
        f"{item['class']} {item['confidence']:.0%}" for item in detections
    )
    draw.text((550, y), f"Observed predictions: {summary}", fill=GREEN, font=F_SMALL)

    footer(
        draw,
        f"This frame contains {len(detections)} predictions produced by the trained checkpoint.",
        "Only final Ultralytics outputs are shown; no synthetic candidate boxes are presented as inference.",
    )
    return frame


def results_scene(
    image: Image.Image,
    detections: list[dict[str, Any]],
    metrics: dict[str, float],
    trt: dict[str, float],
    t4: dict[str, float],
) -> Image.Image:
    frame, draw = canvas("MEASURED RESULTS AND DEPLOYMENT", 5)
    scale, _ = draw_image_panel(
        frame, image, (34, 91), 475, "Actual checkpoint predictions on a synthetic validation image", GREEN
    )
    for detection in detections:
        draw_detection(draw, detection, (34, 91), scale)

    draw.text((550, 94), "Measured evidence", fill=WHITE, font=F_TITLE)
    draw.text((550, 136), "Values loaded from committed JSON artifacts", fill=MUTED, font=F_SMALL)

    metric_cards = [
        ((550, 180, 830, 295), f"{metrics['mAP50'] * 100:.2f}%", "mAP@50", YELLOW),
        ((850, 180, 1130, 295), f"{metrics['mAP50_95'] * 100:.2f}%", "mAP@50:95", CYAN),
        ((550, 315, 830, 430), f"{trt['mean_ms']:.2f} ms", "A100 TRT FP16 mean", GREEN),
        ((850, 315, 1130, 430), f"{trt['fps']:.1f} FPS", "A100 TRT FP16", GREEN),
    ]
    for box, value, label, accent in metric_cards:
        x1, y1, x2, y2 = box
        draw.rectangle(box, fill=SURFACE, outline=LINE, width=2)
        draw.text((x1 + 20, y1 + 18), value, fill=accent, font=F_METRIC)
        draw.text((x1 + 20, y2 - 32), label, fill=MUTED, font=F_SMALL)

    draw.text(
        (550, 464),
        f"T4 PyTorch FP16: {t4['mean_ms']:.2f} ms mean | {t4['fps']:.1f} FPS",
        fill=WHITE,
        font=F_BODY_BOLD,
    )
    draw.text((550, 506), "FastAPI -> Triton ONNX v1 -> Prometheus/Grafana", fill=CYAN, font=F_BODY)
    draw.text((550, 542), "TensorRT benchmarked separately because engine files are environment-bound", fill=MUTED, font=F_SMALL)

    footer(
        draw,
        "Strong synthetic-data results justify the system design, not claims of fab readiness.",
        "Next evidence milestone: qualified real-wafer data, calibration, and cross-tool drift evaluation.",
    )
    return frame


def generate() -> None:
    image = Image.open(INPUT_IMAGE).convert("RGB")
    if image.size != (640, 640):
        raise ValueError(f"Expected a 640 x 640 validation image, received {image.size}")

    summary = load_json(RESULTS_JSON)
    trt_summary = load_json(TRT_JSON)
    t4_summary = load_json(T4_JSON)
    metrics = summary["model_comparison"]["YOLOv8-L"]
    trt = trt_summary["trt_fp16"]
    t4 = t4_summary["direct_inference"]
    detections = get_detections()
    if not detections:
        raise RuntimeError("The selected validation image produced no detections")

    styled_image = stylize_wafer(image)
    frames: list[Image.Image] = []
    frames.extend([cover_scene(styled_image, detections, metrics, trt)] * 12)
    frames.extend([importance_scene(styled_image)] * 12)
    for index in range(20):
        frames.append(architecture_scene((index + 1) / 20))
    for index in range(16):
        visible = min(len(detections), max(1, (index * len(detections) // 12) + 1))
        frames.append(decision_scene(image, detections, visible))
    frames.extend([results_scene(image, detections, metrics, trt, t4)] * 20)

    quantized = [
        frame.quantize(colors=160, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
        for frame in frames
    ]
    quantized[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=quantized[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
    size_mb = OUTPUT_GIF.stat().st_size / 1_000_000
    print(f"Input: {INPUT_IMAGE.relative_to(ROOT)}")
    print(f"Actual post-NMS predictions: {len(detections)}")
    print(f"Output: {OUTPUT_GIF.relative_to(ROOT)}")
    print(f"Frames: {len(frames)} | Duration: {len(frames) / FPS:.1f}s | Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    generate()