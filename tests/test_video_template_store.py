import os

import numpy as np

from ok.ui.qt.video_template.ocr_processor import (
    build_candidate_context,
    build_operation_candidate_boxes,
    build_operation_frame_specs,
    enrich_ocr_boxes,
    run_executor_ocr,
)
from ok.ui.qt.video_template.store import (
    ANNOTATION_STATUS_CONFIRMED,
    FRAME_STATUS_CONFIRMED,
    FRAME_STATUS_OCR_DONE,
    PACKAGE_STATUS_EXPORTED,
    PACKAGE_STATUS_OCR,
    VideoTemplateStore,
)


def test_build_operation_frame_specs_captures_click_and_drag_targets():
    events = [
        {"id": "event_0001", "type": "click", "time": 1.0, "x": 0.2, "y": 0.3},
        {
            "id": "event_0002",
            "type": "drag",
            "time": 2.0,
            "end_time": 3.0,
            "from_x": 0.1,
            "from_y": 0.1,
            "to_x": 0.8,
            "to_y": 0.7,
            "path": [
                {"time": 2.0, "x": 0.1, "y": 0.1},
                {"time": 2.3, "x": 0.3, "y": 0.4},
                {"time": 2.6, "x": 0.5, "y": 0.6},
                {"time": 3.0, "x": 0.8, "y": 0.7},
            ],
        },
    ]

    specs = build_operation_frame_specs(events, duration=5)

    stages = [spec["source_stage"] for spec in specs]
    assert stages == [
        "click_before",
        "click_after",
        "drag_start_before",
        "drag_start_after",
        "drag_mid",
        "drag_end_before",
        "drag_end_after",
        "drag_path_1",
        "drag_path_2",
    ]
    assert specs[2]["operation_points"] == [{"x": 0.1, "y": 0.1, "role": "drag_start"}]
    assert specs[3]["operation_points"] == [{"x": 0.1, "y": 0.1, "role": "drag_start"}]
    drag_end_before = specs[5]
    assert drag_end_before["operation_points"] == [{"x": 0.8, "y": 0.7, "role": "drag_target"}]
    assert specs[7]["operation_points"][2]["role"] == "drag_path"
    assert specs[8]["time"] == 2.6


def test_package_status_and_export_only_confirmed_material(tmp_path):
    store = VideoTemplateStore(root=tmp_path)
    package = store.create_package()
    os.makedirs(package.keyframes_folder, exist_ok=True)
    image_path = os.path.join(package.keyframes_folder, "confirmed.png")
    with open(image_path, "wb") as file:
        file.write(b"fake image")

    package.save_ocr_frames([
        {
            "id": "confirmed",
            "status": FRAME_STATUS_OCR_DONE,
            "image_file": "keyframes/confirmed.png",
        },
        {
            "id": "draft",
            "status": FRAME_STATUS_OCR_DONE,
            "image_file": "keyframes/draft.png",
        },
    ])
    assert package.summary()["status"] == PACKAGE_STATUS_OCR

    frames = package.load_ocr_frames()
    frames[0]["status"] = FRAME_STATUS_CONFIRMED
    package.save_ocr_frames(frames)

    package.save_annotations([
        {
            "frame_id": "confirmed",
            "status": ANNOTATION_STATUS_CONFIRMED,
            "template_name": "确认模板",
            "boxes": [],
        }
    ])

    manifest = package.export_confirmed()

    assert len(manifest["frames"]) == 1
    assert manifest["frames"][0]["id"] == "confirmed"
    assert package.summary()["status"] == PACKAGE_STATUS_EXPORTED
    assert os.path.exists(os.path.join(package.exports_folder, "confirmed.png"))


def test_ocr_context_keeps_near_operation_texts():
    boxes = [{
        "text": "删除存档",
        "bbox": [90, 90, 80, 30],
        "confidence": 0.9,
    }]
    points = [{"x": 0.1, "y": 0.1, "role": "click"}]

    enriched = enrich_ocr_boxes(boxes, points, width=1000, height=1000)
    context = build_candidate_context(enriched, {"source_stage": "click_before"})

    assert enriched[0]["near_operation"] is True
    assert context["near_operation_texts"] == ["删除存档"]


def test_operation_points_create_template_candidate_boxes():
    candidates = build_operation_candidate_boxes(
        [
            {"x": 0.05, "y": 0.05, "role": "click"},
            {"x": 0.05, "y": 0.05, "role": "click"},
            {"x": 0.95, "y": 0.95, "role": "drag_target"},
        ],
        width=200,
        height=100,
        box_size=80,
    )

    assert candidates == [
        {"label": "click", "bbox": [0, 0, 80, 80], "source": "operation_point"},
        {"label": "drag_target", "bbox": [120, 20, 80, 80], "source": "operation_point"},
    ]


def test_operation_points_fit_clicked_object_shape():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    frame[40:78, 60:142] = (240, 240, 240)

    candidates = build_operation_candidate_boxes(
        [{"x": 0.5, "y": 0.5, "role": "click"}],
        width=200,
        height=120,
        frame=frame,
    )

    assert candidates[0]["source"] == "operation_shape"
    x, y, width, height = candidates[0]["bbox"]
    assert 50 <= x <= 64
    assert 30 <= y <= 44
    assert 80 <= width <= 104
    assert 38 <= height <= 60


def test_video_template_ocr_calls_executor_lib_directly():
    class Executor:
        paused = True
        current_task = object()
        config = {"ocr": {"default": {"lib": "rapidocr"}}}
        ocr_po_translation = None
        text_fix = {}

        def ocr_lib(self, name="default"):
            class RapidLike:
                boxes = [[[1, 2], [4, 2], [4, 6], [1, 6]]]
                txts = ["文字"]
                scores = [0.9]

            return lambda image, use_det=True, use_cls=False, use_rec=True: RapidLike()

    boxes = run_executor_ocr(Executor(), np.zeros((20, 30, 3), dtype=np.uint8), threshold=0.1)

    assert boxes[0].name == "文字"
    assert boxes[0].x == 1
