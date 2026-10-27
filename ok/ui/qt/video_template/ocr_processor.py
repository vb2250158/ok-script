import math
import os

import cv2

from ok import og
from ok.feature.Box import Box, sort_boxes
from ok.feature.FeatureSet import resize_image, scale_box
from ok.task.ocr_errors import OCRUnavailableError
from ok.util.image_io import cv_imwrite
from ok.ui.qt.video_template.store import (
    ANNOTATION_STATUS_CONFIRMED,
    FRAME_STATUS_CONFIRMED,
    FRAME_STATUS_OCR_DONE,
    VideoTemplatePackage,
)


class OcrCancelled(Exception):
    pass


VIDEO_TEMPLATE_OCR_TARGET_HEIGHT = 0


class VideoTemplateOcrProcessor:
    """Turns recording facts into OCR keyframe material."""

    def __init__(self, package: VideoTemplatePackage):
        self.package = package

    def process_all_operation_frames(self, progress=None):
        events = self.package.load_events()
        meta = self.package.load_meta()
        specs = build_operation_frame_specs(events, float(meta.get("duration_seconds", 0) or 0))
        return self._process_specs(specs, progress=progress)

    def process_current_frame(self, seconds, progress=None):
        spec = {
            "id": f"manual_{int(seconds * 1000)}",
            "time": seconds,
            "kind": "manual",
            "source_event_id": None,
            "source_stage": "manual",
            "operation_points": [],
        }
        return self._process_specs([spec], progress=progress)

    def _process_specs(self, specs, progress=None):
        frames = self.package.load_ocr_frames()
        by_id = {frame.get("id"): frame for frame in frames}
        processed = []
        for index, spec in enumerate(specs):
            if progress:
                progress(index, len(specs), f"图像识别加工 {index + 1}/{len(specs)} 读取画面")
            frame = self._read_frame_at(spec["time"])
            if frame is None:
                if progress:
                    progress(index + 1, len(specs), f"图像识别加工 {index + 1}/{len(specs)} 跳过：无法读取画面")
                continue
            frame_id = unique_frame_id(spec["id"], by_id)
            image_file = self._write_keyframe(frame_id, frame)
            target_height = video_template_ocr_target_height()
            if progress:
                resolution = "原始分辨率" if target_height <= 0 else f"{target_height}p"
                progress(index, len(specs), f"图像识别加工 {index + 1}/{len(specs)} 文字识别 {resolution}")
            ocr_boxes = run_ocr(frame, target_height=target_height)
            enriched_boxes = enrich_ocr_boxes(
                ocr_boxes,
                spec.get("operation_points") or [],
                frame.shape[1],
                frame.shape[0],
            )
            candidate_boxes = build_operation_candidate_boxes(
                spec.get("operation_points") or [],
                frame.shape[1],
                frame.shape[0],
                frame=frame,
            )
            item = {
                "id": frame_id,
                "time": spec["time"],
                "kind": spec["kind"],
                "source_event_id": spec.get("source_event_id"),
                "source_stage": spec.get("source_stage"),
                "image_file": image_file,
                "status": FRAME_STATUS_OCR_DONE,
                "ocr_boxes": enriched_boxes,
                "candidate_boxes": candidate_boxes,
                "ocr_text": "\n".join(box.get("text", "") for box in ocr_boxes if box.get("text")),
                "candidate_context": build_candidate_context(enriched_boxes, spec),
            }
            frames.append(item)
            by_id[item["id"]] = item
            processed.append(item)
            self.package.save_ocr_frames(frames)
            if progress:
                progress(
                    index + 1,
                    len(specs),
                    f"图像识别加工 {index + 1}/{len(specs)} 完成：文字识别框 {len(enriched_boxes)}，OpenCV ROI 候选 {len(candidate_boxes)}",
                )
        if progress:
            progress(len(specs), len(specs), "图像识别加工完成")
        self.package.save_ocr_frames(frames)
        return processed

    def _read_frame_at(self, seconds):
        cap = cv2.VideoCapture(self.package.recording_path)
        try:
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0, seconds) * 1000)
            ok, frame = cap.read()
            if ok:
                return frame
            return None
        finally:
            cap.release()

    def _write_keyframe(self, frame_id, frame):
        os.makedirs(self.package.keyframes_folder, exist_ok=True)
        filename = f"{frame_id}.png"
        path = os.path.join(self.package.keyframes_folder, filename)
        cv_imwrite(path, frame)
        return os.path.relpath(path, self.package.folder).replace("\\", "/")


def build_operation_frame_specs(events, duration):
    specs = []
    for event in events:
        event_id = event.get("id")
        event_type = event.get("type")
        if event_type == "click":
            points = [{"x": event.get("x"), "y": event.get("y"), "role": "click"}]
            specs.extend([
                frame_spec(event_id, "click_before", "click", event.get("time", 0) - 0.12, duration, points),
                frame_spec(event_id, "click_after", "click", event.get("time", 0) + 0.20, duration, points),
            ])
        elif event_type == "drag":
            start = float(event.get("time", 0) or 0)
            end = float(event.get("end_time", start) or start)
            start_points = [
                {"x": event.get("from_x"), "y": event.get("from_y"), "role": "drag_start"},
            ]
            end_points = [
                {"x": event.get("to_x"), "y": event.get("to_y"), "role": "drag_target"},
            ]
            points = start_points + end_points
            specs.extend([
                frame_spec(event_id, "drag_start_before", "drag", start - 0.12, duration, start_points),
                frame_spec(event_id, "drag_start_after", "drag", start + 0.12, duration, start_points),
                frame_spec(event_id, "drag_mid", "drag", start + (end - start) * 0.5, duration, points),
                frame_spec(event_id, "drag_end_before", "drag", end - 0.08, duration, end_points),
                frame_spec(event_id, "drag_end_after", "drag", end + 0.20, duration, end_points),
            ])
            for index, path_point in enumerate(sample_drag_path_points(event.get("path") or []), start=1):
                path_points = points + [{
                    "x": path_point.get("x"),
                    "y": path_point.get("y"),
                    "role": "drag_path",
                }]
                specs.append(frame_spec(
                    event_id,
                    f"drag_path_{index}",
                    "drag",
                    path_point.get("time", start),
                    duration,
                    path_points,
                ))
    return specs


def sample_drag_path_points(path, max_points=4):
    if len(path) <= 2:
        return []
    middle_points = path[1:-1]
    if len(middle_points) <= max_points:
        return middle_points
    result = []
    for index in range(max_points):
        source_index = round(index * (len(middle_points) - 1) / max(1, max_points - 1))
        result.append(middle_points[source_index])
    return result


def frame_spec(event_id, stage, kind, seconds, duration, points):
    return {
        "id": f"{event_id}_{stage}",
        "time": clamp(seconds, 0, duration),
        "kind": kind,
        "source_event_id": event_id,
        "source_stage": stage,
        "operation_points": points,
    }


def clamp(value, min_value, max_value):
    if max_value <= min_value:
        return min_value
    return max(min_value, min(max_value, float(value or 0)))


def unique_frame_id(base, existing):
    frame_id = safe_id(base)
    if frame_id not in existing:
        return frame_id
    index = 1
    while f"{frame_id}_{index}" in existing:
        index += 1
    return f"{frame_id}_{index}"


def safe_id(text):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text or "frame")).strip("_") or "frame"


def run_ocr(frame, target_height=None):
    executor = getattr(og, "executor", None)
    if executor is None:
        raise OCRUnavailableError("OCR is not available. Start the app first and make sure OCR is configured.")
    if target_height is None:
        target_height = video_template_ocr_target_height()
    boxes = run_executor_ocr(executor, frame, threshold=0.1, target_height=target_height)
    result = []
    for box in boxes or []:
        result.append({
            "text": str(getattr(box, "name", "") or ""),
            "bbox": [
                int(getattr(box, "x", 0)),
                int(getattr(box, "y", 0)),
                int(getattr(box, "width", 0)),
                int(getattr(box, "height", 0)),
            ],
            "confidence": float(getattr(box, "confidence", 0) or 0),
        })
    return result


def run_executor_ocr(executor, frame, threshold=0.1, target_height=VIDEO_TEMPLATE_OCR_TARGET_HEIGHT, lib="default"):
    config = ocr_lib_config(executor, lib)
    image, scale_factor = resize_image(frame, frame.shape[0], target_height)
    lib_name = config.get("lib")
    if lib_name == "rapidocr":
        boxes = run_rapid_ocr(executor.ocr_lib(lib), image, threshold, scale_factor)
    elif lib_name == "paddleocr":
        boxes = run_paddle_ocr(executor.ocr_lib(lib), image, threshold, scale_factor)
    elif lib_name == "dgocr":
        boxes = run_duguang_ocr(executor.ocr_lib(lib), image, threshold, scale_factor)
    elif lib_name == "onnxocr":
        boxes = run_onnx_ocr(executor.ocr_lib(lib), image, threshold, scale_factor)
    else:
        raise OCRUnavailableError(f"OCR lib not supported: {lib_name}")
    fix_texts(executor, boxes)
    return sort_boxes(boxes)


def ocr_lib_config(executor, lib):
    ocr_config = getattr(executor, "config", {}).get("ocr")
    if not isinstance(ocr_config, dict):
        raise OCRUnavailableError("OCR is not configured.")
    lib_config = ocr_config.get(lib)
    if not isinstance(lib_config, dict) and lib == "default" and ocr_config.get("lib"):
        lib_config = ocr_config
    if not isinstance(lib_config, dict) or not lib_config.get("lib"):
        raise OCRUnavailableError(f"OCR config for '{lib}' is missing a 'lib' value.")
    return lib_config


def run_rapid_ocr(ocr_lib, image, threshold, scale_factor):
    result = ocr_lib(image, use_det=True, use_cls=False, use_rec=True)
    boxes = []
    if getattr(result, "boxes", None) is None:
        return boxes
    for pos, text, confidence in zip(result.boxes, result.txts, result.scores):
        box = box_from_quad(pos, text, confidence, threshold, scale_factor)
        if box is not None:
            boxes.append(box)
    return boxes


def run_onnx_ocr(ocr_lib, image, threshold, scale_factor):
    result = ocr_lib.ocr(image)
    boxes = []
    if not result or result[0] is None:
        return boxes
    for item in result[0]:
        pos = item[0]
        text = item[1][0]
        confidence = item[1][1]
        box = box_from_quad(pos, text, confidence, threshold, scale_factor)
        if box is not None:
            boxes.append(box)
    return boxes


def run_duguang_ocr(ocr_lib, image, threshold, scale_factor):
    results = ocr_lib.run(image)
    boxes = []
    for image_results in results or []:
        for item in image_results or []:
            pos = item[0]
            text, confidence = item[1]
            box = box_from_quad(pos, text, confidence, threshold, scale_factor)
            if box is not None:
                boxes.append(box)
    return boxes


def run_paddle_ocr(ocr_lib, image, threshold, scale_factor):
    results = ocr_lib.predict(image)
    boxes = []
    if not results:
        return boxes
    result = results[0]
    for pos, text, confidence in zip(result.get("rec_boxes", []), result.get("rec_texts", []), result.get("rec_scores", [])):
        width = round(pos[2] - pos[0])
        height = round(pos[3] - pos[1])
        if width <= 0 or height <= 0 or confidence < threshold:
            continue
        box = Box(pos[0], pos[1], width, height, confidence, text)
        scale_box(box, scale_factor)
        boxes.append(box)
    return boxes


def box_from_quad(pos, text, confidence, threshold, scale_factor):
    width = round(pos[2][0] - pos[0][0])
    height = round(pos[2][1] - pos[0][1])
    if width <= 0 or height <= 0 or confidence < threshold:
        return None
    box = Box(pos[0][0], pos[0][1], width, height, confidence, text)
    scale_box(box, scale_factor)
    return box


def fix_texts(executor, boxes):
    text_fix = getattr(executor, "text_fix", {}) or {}
    translation = getattr(executor, "ocr_po_translation", None)
    for box in boxes:
        box.name = str(box.name or "").strip()
        if translation is not None:
            fixed = translation.gettext(box.name)
            if fixed != box.name:
                box.name = fixed
            else:
                no_space = box.name.replace(" ", "")
                fixed = translation.gettext(no_space)
                if fixed != no_space:
                    box.name = fixed
        if box.name in text_fix:
            box.name = text_fix[box.name]


def video_template_ocr_target_height():
    executor = getattr(og, "executor", None)
    configured = int(getattr(executor, "ocr_target_height", 0) or 0)
    return configured if configured > 0 else VIDEO_TEMPLATE_OCR_TARGET_HEIGHT


def task_for_ocr():
    return None


def first_task_with_ocr():
    return None


def build_operation_candidate_boxes(operation_points, width, height, box_size=96, frame=None):
    boxes = []
    seen = set()
    for point in operation_points:
        role = point.get("role") or "operation"
        x_value = point.get("x")
        y_value = point.get("y")
        if x_value is None or y_value is None:
            continue
        cx = int(round(float(x_value) * width))
        cy = int(round(float(y_value) * height))
        box = detect_operation_object_box(frame, cx, cy, width, height, box_size=box_size)
        if box is None:
            box = fallback_operation_box(cx, cy, width, height, box_size=box_size)
        x, y, box_width, box_height, source = box
        key = (x, y, box_width, box_height, role, source)
        if key in seen:
            continue
        seen.add(key)
        boxes.append({
            "label": role,
            "bbox": [x, y, int(box_width), int(box_height)],
            "source": source,
        })
    return boxes


def detect_operation_object_box(frame, cx, cy, width, height, box_size=96):
    if frame is None or not hasattr(frame, "shape"):
        return None
    if not (0 <= cx < width and 0 <= cy < height):
        return None
    search_size = max(140, int(box_size * 2.2))
    half = search_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(width, cx + half)
    y2 = min(height, cy + half)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    local_x = cx - x1
    local_y = cy - y1
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 35, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate = best_operation_contour_box(contours, local_x, local_y, roi.shape[1], roi.shape[0])
    if candidate is None:
        return None
    x, y, w, h = candidate
    padding = 6
    x = max(0, x1 + x - padding)
    y = max(0, y1 + y - padding)
    x_to = min(width, x1 + candidate[0] + candidate[2] + padding)
    y_to = min(height, y1 + candidate[1] + candidate[3] + padding)
    w = max(1, x_to - x)
    h = max(1, y_to - y)
    return int(x), int(y), int(w), int(h), "operation_shape"


def best_operation_contour_box(contours, local_x, local_y, roi_width, roi_height):
    best = None
    best_score = None
    roi_area = max(1, roi_width * roi_height)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if w < 10 or h < 10:
            continue
        if area < 160 or area > roi_area * 0.82:
            continue
        contains = (x - 8) <= local_x <= (x + w + 8) and (y - 8) <= local_y <= (y + h + 8)
        center_x = x + w / 2
        center_y = y + h / 2
        distance = math.hypot(center_x - local_x, center_y - local_y)
        if not contains and distance > min(roi_width, roi_height) * 0.35:
            continue
        aspect = max(w / max(1, h), h / max(1, w))
        aspect_penalty = max(0, aspect - 4) * 600
        size_penalty = abs(math.sqrt(area) - 72) * 4
        score = (0 if contains else 4000) + distance * 18 + size_penalty + aspect_penalty
        if best_score is None or score < best_score:
            best_score = score
            best = (x, y, w, h)
    return best


def fallback_operation_box(cx, cy, width, height, box_size=96):
    box_width = min(box_size, max(1, width))
    box_height = min(box_size, max(1, height))
    x = int(max(0, min(width - box_width, round(cx - box_width / 2))))
    y = int(max(0, min(height - box_height, round(cy - box_height / 2))))
    return x, y, int(box_width), int(box_height), "operation_point"


def enrich_ocr_boxes(boxes, operation_points, width, height):
    enriched = []
    for box in boxes:
        item = dict(box)
        item["near_operation"] = is_near_operation(box.get("bbox") or [], operation_points, width, height)
        enriched.append(item)
    return enriched


def is_near_operation(bbox, operation_points, width, height):
    if len(bbox) < 4 or not operation_points:
        return False
    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    for point in operation_points:
        px = float(point.get("x") or 0) * width
        py = float(point.get("y") or 0) * height
        if math.hypot(cx - px, cy - py) <= 120:
            return True
    return False


def build_candidate_context(ocr_boxes, spec):
    texts = [box.get("text") for box in ocr_boxes if box.get("text")]
    near_texts = [box.get("text") for box in ocr_boxes if box.get("near_operation") and box.get("text")]
    return {
        "source_stage": spec.get("source_stage"),
        "text_count": len(texts),
        "texts": texts[:30],
        "near_operation_texts": near_texts[:12],
    }


def confirm_frame(package, frame_id, template_name="", notes=""):
    frames = package.load_ocr_frames()
    annotations = package.load_annotations()
    for frame in frames:
        if frame.get("id") == frame_id:
            frame["status"] = FRAME_STATUS_CONFIRMED
            break
    existing = None
    for annotation in annotations:
        if annotation.get("frame_id") == frame_id:
            existing = annotation
            break
    if existing is None:
        annotations.append({
            "frame_id": frame_id,
            "status": ANNOTATION_STATUS_CONFIRMED,
            "template_name": template_name,
            "notes": notes,
            "boxes": [],
        })
    else:
        existing.update({
            "status": ANNOTATION_STATUS_CONFIRMED,
            "template_name": template_name,
            "notes": notes,
        })
    package.save_ocr_frames(frames)
    package.save_annotations(annotations)
