import json
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox

from ok import og
from ok.feature.Box import Box
from ok.feature.Feature import Feature
from ok.feature.FeatureSet import FeatureSet
from ok.feature.VisualRuleTemplate import template_names_with_visual_rules, visual_rule_name_for_template
from ok.ui.qt.tasks.VideoTemplateTab import (
    DEFAULT_TEMPLATE_GROUP_NAME,
    TEMPLATE_REF_TYPE_RULE,
    TEMPLATE_REF_TYPE_TEMPLATE,
    PreviewMatchWorker,
    RecordVideoTemplatePage,
    ensure_feature_set_uses_template_coco,
)


def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def test_visual_rule_name_uses_last_underscore():
    assert visual_rule_name_for_template("任务图标_A") == "任务图标"
    assert visual_rule_name_for_template("1_2_3_4") == "1_2_3"
    assert visual_rule_name_for_template("NoSuffix") is None
    assert template_names_with_visual_rules(["任务图标_A", "任务图标_B", "OnlyOne_A"]) == [
        "OnlyOne_A",
        "任务图标",
        "任务图标_A",
        "任务图标_B",
    ]


def test_runtime_feature_set_prefers_workspace_template_coco(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    workspace = project_root / "panghu"
    root_coco = project_root / "ok_templates" / "coco_annotations.json"
    workspace_coco = workspace / "ok_templates" / "coco_annotations.json"
    root_coco.parent.mkdir(parents=True)
    workspace_coco.parent.mkdir(parents=True)
    root_coco.write_text("{}", encoding="utf-8")
    workspace_coco.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(workspace)
    og.config = {
        "project_root": str(project_root),
        "resolved_working_directory": str(workspace),
        "template_matching": {"coco_feature_json": "ok_templates/coco_annotations.json"},
    }

    changed_paths = []

    class FakeFeatureSet:
        coco_json = str(root_coco)

        def set_coco_json(self, path):
            changed_paths.append(path)
            self.coco_json = path
            return True

    feature_set = FakeFeatureSet()

    assert ensure_feature_set_uses_template_coco(feature_set) is feature_set
    assert changed_paths == [str(workspace_coco)]
    assert feature_set.coco_json == str(workspace_coco)


def test_feature_set_finds_visual_rule_template_from_sample_group(tmp_path):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    _draw_rule_sample(image, 10, (255, 80, 80))
    _draw_rule_sample(image, 65, (80, 120, 255))
    image_path = tmp_path / "screen.png"
    cv2.imwrite(str(image_path), image)

    coco_path = tmp_path / "coco_annotations.json"
    coco_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": image_path.name, "width": 120, "height": 80}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 36, 36]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [65, 20, 36, 36]},
        ],
        "categories": [
            {"id": 1, "name": "TaskIcon_A", "supercategory": ""},
            {"id": 2, "name": "TaskIcon_B", "supercategory": ""},
        ],
    }, ensure_ascii=False), encoding="utf-8")

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    _draw_rule_sample(frame, 38, (80, 255, 120))
    feature_set = FeatureSet(False, str(coco_path), 1, 1, default_threshold=0.75)

    boxes = feature_set.find_feature(frame, "TaskIcon", threshold=0.65, limit=1)

    assert "TaskIcon" in feature_set.template_names()
    assert boxes
    assert boxes[0].name == "TaskIcon"
    assert boxes[0].confidence >= 0.65


def test_preview_match_worker_rebuilds_visual_rule_after_frame_size_change(tmp_path):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    _draw_rule_sample(image, 10, (255, 80, 80))
    _draw_rule_sample(image, 65, (80, 120, 255))
    image_path = tmp_path / "screen.png"
    cv2.imwrite(str(image_path), image)

    coco_path = tmp_path / "coco_annotations.json"
    coco_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": image_path.name, "width": 120, "height": 80}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 36, 36]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [65, 20, 36, 36]},
        ],
        "categories": [
            {"id": 1, "name": "TaskIcon_A", "supercategory": ""},
            {"id": 2, "name": "TaskIcon_B", "supercategory": ""},
        ],
    }, ensure_ascii=False), encoding="utf-8")

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    _draw_rule_sample(frame, 38, (80, 255, 120))
    feature_set = FeatureSet(False, str(coco_path), 1, 1, default_threshold=0.65)
    feature_set.check_size(np.zeros((160, 240, 3), dtype=np.uint8))
    feature_set.process_data()
    assert feature_set.width == 240
    assert "TaskIcon" in feature_set.visual_rule_dict

    worker = PreviewMatchWorker(
        1,
        feature_set,
        frame,
        selected_refs=[{"name": "TaskIcon", "type": TEMPLATE_REF_TYPE_RULE}],
        confidence_threshold=0.65,
        min_roi_similarity=0.9,
    )
    finished = []
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))

    worker.run()

    assert feature_set.width == 120
    assert "TaskIcon" in feature_set.visual_rule_dict
    assert [(box.x, box.y, box.name) for box in finished[0][1]] == [(38, 20, "TaskIcon")]


def test_feature_set_rebuilds_visual_rules_when_coco_source_changes(tmp_path):
    image_path = tmp_path / "screen.png"
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    _draw_rule_sample(image, 5, (255, 80, 80))
    _draw_rule_sample(image, 65, (80, 120, 255))
    cv2.imwrite(str(image_path), image)
    coco_path = tmp_path / "coco_annotations.json"

    _write_rule_coco(coco_path, image_path.name, "TaskIcon_B")
    feature_set = FeatureSet(False, str(coco_path), 1, 1)

    assert "TaskIcon" in feature_set.template_names()

    _write_rule_coco(coco_path, image_path.name, "OtherIcon_B")

    assert "TaskIcon" not in feature_set.template_names()
    assert "OtherIcon" not in feature_set.template_names()


def test_feature_set_template_names_survive_duplicate_annotation_names(tmp_path):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[20:40, 70:90] = (90, 180, 240)
    image_path = tmp_path / "screen.png"
    cv2.imwrite(str(image_path), image)
    coco_path = tmp_path / "coco_annotations.json"
    coco_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": image_path.name, "width": 120, "height": 80}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 20, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [30, 20, 20, 20]},
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [70, 20, 20, 20]},
        ],
        "categories": [
            {"id": 1, "name": "DuplicateTemplate", "supercategory": ""},
            {"id": 2, "name": "TaskIcon", "supercategory": ""},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    feature_set = FeatureSet(False, str(coco_path), 1, 1)

    assert "DuplicateTemplate" in feature_set.template_names()
    assert "TaskIcon" in feature_set.template_names()
    assert feature_set.get_feature_by_name(image, "TaskIcon") is not None


def test_feature_set_missing_feature_does_not_rebuild_after_full_load():
    class CountingFeatureSet(FeatureSet):
        def __init__(self):
            self.feature_dict = {}
            self.visual_rule_dict = {}
            self.lock = threading.Lock()
            self._visual_rule_source_fingerprint = None
            self._missing = {}
            self.calls = []

        def _refresh_if_visual_rule_sources_changed(self):
            pass

        def _source_signature(self):
            return ()

        def process_data(self, feature_name=None):
            self.calls.append(feature_name)
            if feature_name is None:
                self._visual_rule_source_fingerprint = ()
            return True

    feature_set = CountingFeatureSet()

    feature_set.ensure_feature("Missing")
    feature_set.ensure_feature("Missing")

    assert feature_set.calls == ["Missing", None]


def test_feature_set_duplicate_template_name_searches_alternative_local_samples(tmp_path):
    source = np.zeros((80, 140, 3), dtype=np.uint8)
    sample = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.circle(sample, (10, 10), 8, (40, 180, 240), thickness=-1)
    cv2.line(sample, (4, 10), (16, 10), (240, 240, 80), thickness=2)
    source[20:40, 5:25] = sample
    source[20:40, 95:115] = sample
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    coco_path = tmp_path / "coco_annotations.json"
    coco_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": source_path.name, "width": 140, "height": 80}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 20, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [95, 20, 20, 20]},
        ],
        "categories": [
            {"id": 1, "name": "RepeatedButton", "supercategory": ""},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    frame = np.zeros((80, 140, 3), dtype=np.uint8)
    frame[20:40, 95:115] = sample
    feature_set = FeatureSet(False, str(coco_path), 0.02, 0.02, default_threshold=0.9)

    boxes = feature_set.find_feature(frame, "RepeatedButton", threshold=0.9, limit=1)

    assert boxes
    assert boxes[0].x >= 90


def test_feature_set_duplicate_template_name_keeps_local_search(tmp_path):
    source = np.zeros((80, 140, 3), dtype=np.uint8)
    sample = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.circle(sample, (10, 10), 8, (40, 180, 240), thickness=-1)
    cv2.line(sample, (4, 10), (16, 10), (240, 240, 80), thickness=2)
    source[20:40, 5:25] = sample
    source[20:40, 95:115] = sample
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    coco_path = tmp_path / "coco_annotations.json"
    coco_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": source_path.name, "width": 140, "height": 80}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 20, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [95, 20, 20, 20]},
        ],
        "categories": [
            {"id": 1, "name": "RepeatedButton", "supercategory": ""},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    frame = np.zeros((80, 140, 3), dtype=np.uint8)
    frame[50:70, 58:78] = sample
    feature_set = FeatureSet(False, str(coco_path), 0.02, 0.02, default_threshold=0.9)

    boxes = feature_set.find_feature(frame, "RepeatedButton", threshold=0.9, limit=1)

    assert not boxes


def test_feature_set_full_load_reads_each_coco_image(tmp_path):
    first = np.zeros((80, 140, 3), dtype=np.uint8)
    first[20:40, 5:25] = (20, 20, 220)
    second = np.zeros((80, 140, 3), dtype=np.uint8)
    sample = np.zeros((20, 24, 3), dtype=np.uint8)
    sample[:, :] = (30, 180, 230)
    cv2.line(sample, (4, 4), (20, 16), (240, 240, 240), thickness=2)
    second[50:70, 90:114] = sample
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    cv2.imwrite(str(first_path), first)
    cv2.imwrite(str(second_path), second)
    coco_path = tmp_path / "coco_annotations.json"
    coco_path.write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": first_path.name, "width": 140, "height": 80},
            {"id": 2, "file_name": second_path.name, "width": 140, "height": 80},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 20, 20, 20]},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [90, 50, 24, 20]},
        ],
        "categories": [
            {"id": 1, "name": "FirstButton", "supercategory": ""},
            {"id": 2, "name": "SecondButton", "supercategory": ""},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    frame = np.zeros((80, 140, 3), dtype=np.uint8)
    frame[50:70, 90:114] = sample
    feature_set = FeatureSet(False, str(coco_path), 0.02, 0.02, default_threshold=0.9)
    feature_set.template_names()

    boxes = feature_set.find_feature(frame, "SecondButton", threshold=0.9, limit=1)

    assert boxes
    assert boxes[0].x == 90
    assert feature_set.get_feature_by_name(frame, "SecondButton").mat.std() == sample.std()


def test_feature_set_target_loads_multiple_names_from_same_image(tmp_path):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[20:40, 10:30] = (90, 180, 240)
    image[20:40, 70:90] = (240, 180, 90)
    image_path = tmp_path / "screen.png"
    cv2.imwrite(str(image_path), image)
    coco_path = tmp_path / "coco_annotations.json"
    coco_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": image_path.name, "width": 120, "height": 80}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [70, 20, 20, 20]},
        ],
        "categories": [
            {"id": 1, "name": "StartButton", "supercategory": ""},
            {"id": 2, "name": "SettingsButton", "supercategory": ""},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    feature_set = FeatureSet(False, str(coco_path), 1, 1)

    assert feature_set.get_feature_by_name(image, "SettingsButton") is not None
    assert feature_set.get_feature_by_name(image, "StartButton") is not None


def test_preview_match_worker_uses_feature_set_template_names_for_all_templates():
    class FakeFeatureSet:
        def __init__(self):
            self.seen_names = []

        def template_names(self):
            return ["Start", "TaskIcon"]

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0):
            self.seen_names.append(names)
            return []

    feature_set = FakeFeatureSet()
    worker = PreviewMatchWorker(1, feature_set, np.zeros((12, 12, 3), dtype=np.uint8))

    worker.run()

    assert feature_set.seen_names == ["Start", "TaskIcon"]


def test_preview_match_worker_does_not_drop_late_template_names():
    names = [f"Template_{index:03d}" for index in range(240)] + ["StartSettings"]

    class FakeFeatureSet:
        def __init__(self):
            self.seen_names = None
            self.seen_threshold = None

        def template_names(self):
            return names

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0):
            self.seen_names = names
            self.seen_threshold = threshold
            return []

    feature_set = FakeFeatureSet()
    worker = PreviewMatchWorker(1, feature_set, np.zeros((12, 12, 3), dtype=np.uint8), confidence_threshold=0.5)

    worker.run()

    assert "StartSettings" in feature_set.seen_names
    assert feature_set.seen_threshold == 0.5


def test_preview_match_worker_falls_back_when_one_template_breaks_batch():
    class FakeFeatureSet:
        def __init__(self):
            self.calls = []

        def template_names(self):
            return ["DuplicateTemplate", "TaskIcon"]

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0):
            self.calls.append(names)
            if isinstance(names, list):
                raise ValueError("Multiple boxes found for category DuplicateTemplate")
            if names == "DuplicateTemplate":
                raise ValueError("Multiple boxes found for category DuplicateTemplate")
            return [SimpleNamespace(name=names)]

    finished = []
    failed = []
    worker = PreviewMatchWorker(2, FakeFeatureSet(), np.zeros((12, 12, 3), dtype=np.uint8))
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))
    worker.signals.failed.connect(lambda seq, message: failed.append((seq, message)))

    worker.run()

    assert not failed
    assert finished
    assert finished[0][0] == 2
    assert [box.name for box in finished[0][1]] == ["TaskIcon"]


def test_preview_match_worker_tries_next_group_when_current_group_misses():
    class FakeFeatureSet:
        def __init__(self):
            self.calls = []

        def template_names(self):
            return ["Missing", "Hit", "SameGroup", "Later"]

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0):
            self.calls.append(names)
            if names in {"Hit", "SameGroup"}:
                return [Box(len(self.calls) * 10, 1, 5, 5, 1.0, names)]
            return []

    feature_set = FakeFeatureSet()
    finished = []
    worker = PreviewMatchWorker(
        8,
        feature_set,
        np.zeros((20, 20, 3), dtype=np.uint8),
        match_groups=[
            {"name": "第一组", "templates": ["Missing"]},
            {"name": "第二组", "templates": ["Hit", "SameGroup"]},
            {"name": "后续组", "templates": ["Later"]},
        ],
        min_roi_similarity=0.0,
    )
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))

    worker.run()

    assert feature_set.calls == ["Missing", "Hit", "SameGroup"]
    assert [box.name for box in finished[0][1]] == ["Hit", "SameGroup"]


def test_preview_match_worker_stops_after_first_matching_group():
    class FakeFeatureSet:
        def __init__(self):
            self.calls = []

        def template_names(self):
            return ["Hit", "AlsoHit", "Later"]

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0):
            self.calls.append(names)
            return [Box(len(self.calls) * 10, 1, 5, 5, 1.0, names)]

    feature_set = FakeFeatureSet()
    finished = []
    worker = PreviewMatchWorker(
        9,
        feature_set,
        np.zeros((20, 20, 3), dtype=np.uint8),
        match_groups=[
            {"name": "命中组", "templates": ["Hit", "AlsoHit"]},
            {"name": "后续组", "templates": ["Later"]},
        ],
        min_roi_similarity=0.0,
    )
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))

    worker.run()

    assert feature_set.calls == ["Hit", "AlsoHit"]
    assert [box.name for box in finished[0][1]] == ["Hit", "AlsoHit"]


def test_preview_match_worker_reports_group_detail_hits():
    class FakeFeatureSet:
        def template_names(self):
            return ["Missing", "Hit"]

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0):
            if names == "Hit":
                return [Box(10, 1, 5, 5, 1.0, "Hit")]
            return []

    details = []
    worker = PreviewMatchWorker(
        10,
        FakeFeatureSet(),
        np.zeros((20, 20, 3), dtype=np.uint8),
        match_groups=[{"name": "测试分组", "templates": ["Missing", "Hit"]}],
        min_roi_similarity=0.0,
    )
    worker.signals.detail.connect(lambda seq, detail: details.append((seq, detail)))

    worker.run()

    assert details[-1][0] == 10
    assert details[-1][1]["group"] == "测试分组"
    assert details[-1][1]["template"] == "Hit"
    assert details[-1][1]["hit_counts"] == {"Hit": 1}


def test_record_template_groups_are_flat_and_keep_unassigned_templates_in_default_group():
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    page.template_group_config = {"groups": [{"name": "农田", "templates": ["Farm"]}]}
    page.available_record_template_names = lambda: ["Farm", "RuleFarm", "TaskIcon"]

    assert page.complete_template_groups() == [
        {"name": "农田", "templates": ["Farm"]},
        {"name": DEFAULT_TEMPLATE_GROUP_NAME, "templates": ["RuleFarm", "TaskIcon"], "default": True},
    ]


def test_record_template_preview_match_groups_no_longer_needs_toggle():
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    page.show_all_templates_check = SimpleNamespace(isChecked=lambda: True)
    page.template_group_tabs = SimpleNamespace(currentIndex=lambda: 0)
    page.specific_preview_tab_index = 2
    groups = [{"name": DEFAULT_TEMPLATE_GROUP_NAME, "templates": ["Farm"], "default": True}]
    page.complete_template_groups = lambda: groups

    assert page.preview_match_groups() is groups


def test_record_template_specific_preview_tab_uses_specified_templates():
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    page.template_group_config = {
        "groups": [],
        "default_order": [],
        "specified_templates": [
            {"name": "Farm", "type": "rule"},
            {"name": "TaskIcon", "type": "template"},
        ],
    }
    page.template_group_tabs = SimpleNamespace(currentIndex=lambda: 2)
    page.specific_preview_tab_index = 2
    page.show_all_templates_check = SimpleNamespace(isChecked=lambda: True)

    assert page.selected_preview_template_names() == ["Farm", "TaskIcon"]
    assert page.selected_preview_template_refs() == [
        {"name": "Farm", "type": "rule"},
        {"name": "TaskIcon", "type": "template"},
    ]
    assert page.preview_match_groups() is None


def test_record_template_specific_preview_empty_list_does_not_start_worker():
    class Label:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    page.tr = lambda text: text
    page.template_group_config = {"groups": [], "default_order": [], "specified_templates": []}
    page.template_group_tabs = SimpleNamespace(currentIndex=lambda: 2)
    page.specific_preview_tab_index = 2
    page.show_all_templates_check = SimpleNamespace(isChecked=lambda: True)
    page.match_status_label = Label()
    page.preview_match_boxes = [object()]
    page.preview_match_worker = None

    page.schedule_template_preview_match(np.zeros((8, 8, 3), dtype=np.uint8))

    assert page.preview_match_worker is None
    assert page.preview_match_boxes == []
    assert page.match_status_label.text == "模板匹配：未指定模板"
    assert page.preview_match_current_text == "未指定模板"


def test_record_template_agent_assist_prompt_describes_frame_writeback_contract():
    from ok.ui.qt.tasks.VideoTemplateTab import EditVideoTemplatePage

    prompt = EditVideoTemplatePage.build_agent_assist_prompt(
        {
            "package_folder": r"C:\recordings\pkg",
            "recording_path": r"C:\recordings\pkg\recording.mp4",
            "frames_path": r"C:\recordings\pkg\ocr_frames.json",
            "annotations_path": r"C:\recordings\pkg\annotations_draft.json",
            "frame_id": "event_0002_drag_start_before",
            "frame_index": 2,
            "frame_total": 8,
            "time_seconds": 1.25,
            "kind": "event",
            "source_stage": "drag_start_before",
            "image_path": r"C:\recordings\pkg\keyframes\event_0002.png",
            "existing_boxes": [{"label": "旧框", "bbox": [1, 2, 3, 4]}],
        },
        "空农田",
    )

    assert "识别目标：空农田" in prompt
    assert "当前 frame_id：event_0002_drag_start_before" in prompt
    assert "第 2/8 个" in prompt
    assert "当前视频时间：1.250 秒" in prompt
    assert "当前帧截图" in prompt
    assert "annotations_draft.json" in prompt
    assert "只修改标注草稿数据" in prompt
    assert "{\"label\":\"标注名\",\"bbox\":[x,y,width,height],\"source\":\"agent\"}" in prompt
    assert '"旧框"' in prompt


def test_export_template_list_starts_background_worker(tmp_path, monkeypatch):
    from ok.ui.qt.tasks import VideoTemplateTab as video_tab
    from ok.ui.qt.tasks.VideoTemplateTab import EditVideoTemplatePage

    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), np.zeros((12, 16, 3), dtype=np.uint8))

    captured = {}

    class FakeSignal:
        def connect(self, slot):
            self.slot = slot

    class FakeWorker:
        def __init__(self, source_path, boxes, target_name, overwrite_labels=None):
            captured["source_path"] = source_path
            captured["boxes"] = boxes
            captured["target_name"] = target_name
            captured["overwrite_labels"] = overwrite_labels
            self.signals = SimpleNamespace(
                progress=FakeSignal(),
                finished=FakeSignal(),
                failed=FakeSignal(),
            )

    started = []
    monkeypatch.setattr(video_tab, "ExportTemplateWorker", FakeWorker)
    monkeypatch.setattr(video_tab, "QThreadPool", SimpleNamespace(globalInstance=lambda: SimpleNamespace(start=started.append)))

    page = EditVideoTemplatePage.__new__(EditVideoTemplatePage)
    page.tr = lambda text: text
    page.current_package = SimpleNamespace(folder=str(tmp_path))
    page.current_frame = {"image_file": image_path.name}
    page.export_template_worker = None
    page.flush_pending_auto_confirm = lambda: None
    page.parse_annotation_boxes = lambda show_error=True: [{"label": "Farm", "bbox": [1, 2, 3, 4]}]
    page.valid_template_annotation_boxes = lambda boxes: boxes
    page.current_template_export_image_name = lambda: "target.png"
    page.find_export_template_conflicts = lambda boxes: []
    busy = []
    page.set_busy = lambda busy_state, message, cancellable=True: busy.append((busy_state, message, cancellable))
    progress = []
    page.on_worker_progress = lambda done, total, message: progress.append((done, total, message))

    page.export_current_annotation_templates()

    assert captured == {
        "source_path": str(image_path),
        "boxes": [{"label": "Farm", "bbox": [1, 2, 3, 4]}],
        "target_name": "target.png",
        "overwrite_labels": [],
    }
    assert busy == [(True, "导出模板列表：检查同名模板...", False)]
    assert progress == [(0, 4, "导出模板列表：准备中...")]
    assert started and started[0] is page.export_template_worker


def test_export_template_conflict_scan_reports_existing_and_new_boxes():
    from ok.ui.qt.tasks.VideoTemplateTab import EditVideoTemplatePage

    coco = {
        "images": [
            {"id": 1, "file_name": "old.png", "width": 100, "height": 80},
            {"id": 2, "file_name": "other.png", "width": 100, "height": 80},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 11]},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [3, 4, 12, 13]},
        ],
        "categories": [
            {"id": 1, "name": "Farm", "supercategory": ""},
            {"id": 2, "name": "Task", "supercategory": ""},
        ],
    }

    conflicts = EditVideoTemplatePage.find_export_template_conflicts_in_coco(
        coco,
        [
            {"label": "Farm", "bbox": [20, 21, 30, 31]},
            {"label": "NewOnly", "bbox": [5, 6, 7, 8]},
        ],
    )

    assert conflicts == [
        {
            "label": "Farm",
            "existing": [{"annotation_id": 1, "image_file": "old.png", "bbox": [1, 2, 10, 11]}],
            "new": [{"label": "Farm", "bbox": [20, 21, 30, 31]}],
        }
    ]
    detail = EditVideoTemplatePage.export_conflict_detail_text(conflicts, r"C:\tmp\frame.png", "target.png")
    assert "同名模板：Farm" in detail
    assert "图片：old.png，bbox：[1, 2, 10, 11]" in detail
    assert "bbox：[20, 21, 30, 31]" in detail


def test_replace_coco_annotations_can_overwrite_same_named_labels_across_images():
    from ok.ui.qt.tasks.VideoTemplateTab import EditVideoTemplatePage

    coco = {
        "images": [
            {"id": 1, "file_name": "old.png", "width": 100, "height": 80},
            {"id": 2, "file_name": "target.png", "width": 100, "height": 80},
            {"id": 3, "file_name": "keep.png", "width": 100, "height": 80},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 11], "area": 110, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [3, 4, 12, 13], "area": 156, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 2, "bbox": [5, 6, 14, 15], "area": 210, "iscrowd": 0},
        ],
        "categories": [
            {"id": 1, "name": "Farm", "supercategory": ""},
            {"id": 2, "name": "Task", "supercategory": ""},
        ],
    }

    EditVideoTemplatePage.replace_coco_annotations_for_image(
        coco,
        2,
        [{"label": "Farm", "bbox": [20, 21, 30, 31]}],
        overwrite_labels={"Farm"},
    )

    remaining = [
        (annotation["image_id"], annotation["category_id"], annotation["bbox"])
        for annotation in coco["annotations"]
    ]
    assert remaining == [
        (3, 2, [5, 6, 14, 15]),
        (2, 1, [20, 21, 30, 31]),
    ]


def test_export_template_list_passes_conflict_labels_when_prompt_is_skipped(tmp_path, monkeypatch):
    from ok.ui.qt.tasks import VideoTemplateTab as video_tab
    from ok.ui.qt.tasks.VideoTemplateTab import EditVideoTemplatePage

    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), np.zeros((12, 16, 3), dtype=np.uint8))

    captured = {}

    class FakeSignal:
        def connect(self, slot):
            self.slot = slot

    class FakeWorker:
        def __init__(self, source_path, boxes, target_name, overwrite_labels=None):
            captured["source_path"] = source_path
            captured["boxes"] = boxes
            captured["target_name"] = target_name
            captured["overwrite_labels"] = overwrite_labels
            self.signals = SimpleNamespace(
                progress=FakeSignal(),
                finished=FakeSignal(),
                failed=FakeSignal(),
            )

    started = []
    monkeypatch.setattr(video_tab, "ExportTemplateWorker", FakeWorker)
    monkeypatch.setattr(video_tab, "QThreadPool", SimpleNamespace(globalInstance=lambda: SimpleNamespace(start=started.append)))

    page = EditVideoTemplatePage.__new__(EditVideoTemplatePage)
    page.tr = lambda text: text
    page.current_package = SimpleNamespace(folder=str(tmp_path))
    page.current_frame = {"image_file": image_path.name}
    page.export_template_worker = None
    page.skip_export_template_conflict_prompt = True
    page.flush_pending_auto_confirm = lambda: None
    page.parse_annotation_boxes = lambda show_error=True: [{"label": "Farm", "bbox": [1, 2, 3, 4]}]
    page.valid_template_annotation_boxes = lambda boxes: boxes
    page.current_template_export_image_name = lambda: "target.png"
    page.find_export_template_conflicts = lambda boxes: [{"label": "Farm", "existing": [], "new": []}]
    page.set_busy = lambda *_args, **_kwargs: None
    page.on_worker_progress = lambda *_args, **_kwargs: None

    page.export_current_annotation_templates()

    assert captured["overwrite_labels"] == ["Farm"]
    assert started and started[0] is page.export_template_worker


def test_record_template_group_config_preserves_specified_templates():
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    page.template_group_config = {
        "groups": [{"name": "农田", "templates": ["Farm"]}],
        "default_order": ["TaskIcon"],
        "specified_templates": [
            {"name": "Farm", "type": "rule"},
            {"name": "TaskIcon", "type": "template"},
        ],
    }

    assert page.template_group_config_from(groups=[]) == {
        "groups": [],
        "default_order": ["TaskIcon"],
        "specified_templates": [
            {"name": "Farm", "type": "rule"},
            {"name": "TaskIcon", "type": "template"},
        ],
    }


def test_record_template_group_config_migrates_legacy_specified_template_strings(monkeypatch):
    monkeypatch.setattr("ok.ui.qt.tasks.VideoTemplateTab.visual_rule_template_names", lambda choices=None: {"Farm"})

    assert RecordVideoTemplatePage.sanitize_template_group_config({
        "groups": [],
        "default_order": [],
        "specified_templates": ["Farm", "TaskIcon"],
    })["specified_templates"] == [
        {"name": "Farm", "type": "rule"},
        {"name": "TaskIcon", "type": "template"},
    ]


def test_record_template_name_from_combo_prefers_item_data():
    app()
    combo = QComboBox()
    combo.setEditable(True)
    combo.addItem("规则模板  Farm", "Farm")
    combo.setCurrentIndex(0)

    assert RecordVideoTemplatePage.template_name_from_combo(combo) == "Farm"
    combo.setEditText("TaskIcon")
    assert RecordVideoTemplatePage.template_name_from_combo(combo) == "TaskIcon"


def test_record_template_rule_choice_combo_uses_validation_display_and_data():
    app()
    from ok.ui.qt.tasks.VideoTemplateTab import add_template_choice_to_combo

    combo = QComboBox()
    combo.setEditable(True)
    add_template_choice_to_combo(
        combo,
        "空农田",
        True,
        "规则模板",
    )
    combo.setCurrentIndex(0)

    assert combo.itemText(0) == "规则模板  空农田"
    assert combo.itemData(0) == {"name": "空农田", "type": "rule"}
    assert "规则模板" in combo.itemData(0, Qt.ToolTipRole)
    assert RecordVideoTemplatePage.template_ref_from_combo(combo) == {"name": "空农田", "type": "rule"}


def test_record_template_combo_keeps_same_named_rule_and_template_distinct():
    app()
    from ok.ui.qt.tasks.VideoTemplateTab import add_template_choice_to_combo

    combo = QComboBox()
    combo.setEditable(True)
    add_template_choice_to_combo(combo, "空农田", True, "规则模板")
    add_template_choice_to_combo(combo, "空农田", False, "")

    combo.setCurrentIndex(0)
    assert RecordVideoTemplatePage.template_ref_from_combo(combo) == {"name": "空农田", "type": "rule"}

    combo.setCurrentIndex(1)
    assert RecordVideoTemplatePage.template_ref_from_combo(combo) == {"name": "空农田", "type": "template"}

    combo.setCurrentIndex(-1)
    combo.setEditText("空农田")
    assert RecordVideoTemplatePage.template_ref_from_combo(combo) == {"name": "空农田", "type": "template"}


def test_record_template_preview_match_template_state_prefers_current_over_hit():
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    page.preview_match_current_group = "农田"
    page.preview_match_current_template = "Farm"
    page.preview_match_hit_counts = {"Farm": 2, "Soil": 1}
    page.preview_match_scanned_templates = {"Water"}

    assert page.preview_match_template_state("农田", "Farm") == "current"
    assert page.preview_match_template_state("农田", "Soil") == "hit"
    assert page.preview_match_template_state("农田", "Water") == "miss"
    assert page.preview_match_template_state("农田", "Seed") == ""
    page.preview_match_current_template = ""
    assert page.preview_match_template_state("农田", "Farm") == "hit"


def test_record_template_preview_hit_counts_refresh_visible_item_states_only():
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    reload_calls = []
    refresh_calls = []
    page.preview_match_tab_index = 1
    page.template_group_tabs = SimpleNamespace(currentIndex=lambda: 0)
    page.reload_preview_match_groups = lambda: reload_calls.append("reload")
    page.refresh_preview_match_item_states = lambda template_names=None: refresh_calls.append(template_names)

    page.update_preview_match_hit_counts({"Farm": 1})

    assert reload_calls == []
    assert refresh_calls == []
    assert page.preview_match_hit_counts == {"Farm": 1}

    page.template_group_tabs = SimpleNamespace(currentIndex=lambda: 1)
    page.update_preview_match_hit_counts({"Farm": 2})

    assert reload_calls == []
    assert refresh_calls == [None]
    assert page.preview_match_hit_counts == {"Farm": 2}


def test_record_template_preview_detail_marks_scanned_template_as_miss():
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    page.tr = lambda text: text
    page.preview_match_seq = 3
    page.preview_match_current_group = ""
    page.preview_match_current_template = ""
    page.preview_match_scanned_templates = set()
    page.preview_match_hit_counts = {}
    page.last_preview_detail_ui_at = 0.0
    page.preview_match_tab_index = 1
    page.template_group_tabs = SimpleNamespace(currentIndex=lambda: 0)

    page.on_preview_match_detail(3, {
        "group": "默认分组",
        "template": "Farm",
        "current": 2,
        "total": 10,
        "hit_counts": {},
        "matched_count": 0,
        "elapsed_ms": 12,
    })

    assert page.preview_match_scanned_templates == {"Farm"}
    assert page.preview_match_template_state("默认分组", "Farm") == "miss"


def test_record_template_preview_detail_searching_state_is_current_until_scanned():
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    page.tr = lambda text: text
    page.preview_match_seq = 4
    page.preview_match_current_group = ""
    page.preview_match_current_template = ""
    page.preview_match_scanned_templates = set()
    page.preview_match_hit_counts = {}
    page.last_preview_detail_ui_at = 0.0
    page.preview_match_tab_index = 1
    page.template_group_tabs = SimpleNamespace(currentIndex=lambda: 0)

    page.on_preview_match_detail(4, {
        "group": "默认分组",
        "template": "Farm",
        "current": 1,
        "total": 10,
        "hit_counts": {},
        "matched_count": 0,
        "elapsed_ms": 1,
        "searching": True,
    })

    assert page.preview_match_template_state("默认分组", "Farm") == "current"
    assert page.preview_match_scanned_templates == set()

    page.on_preview_match_detail(4, {
        "group": "默认分组",
        "template": "Farm",
        "current": 1,
        "total": 10,
        "hit_counts": {},
        "matched_count": 0,
        "elapsed_ms": 2,
    })

    assert page.preview_match_template_state("默认分组", "Farm") == "miss"


def test_record_template_preview_match_ui_updates_are_throttled():
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    page.last_preview_progress_ui_at = time.monotonic()

    assert not page.should_update_preview_match_progress_ui(2, 10)
    assert page.should_update_preview_match_progress_ui(1, 10)
    page.last_preview_detail_ui_at = time.monotonic()
    assert page.should_update_preview_match_detail_ui(5, 10, skipped=True)


def test_record_template_groups_apply_default_template_order():
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    page.template_group_config = {
        "groups": [{"name": "农田", "templates": ["Farm"]}],
        "default_order": ["TaskIcon", "RuleFarm"],
    }
    page.available_record_template_names = lambda: ["Farm", "RuleFarm", "TaskIcon", "Other"]

    assert page.complete_template_groups()[-1] == {
        "name": DEFAULT_TEMPLATE_GROUP_NAME,
        "templates": ["TaskIcon", "RuleFarm", "Other"],
        "default": True,
    }


def test_record_template_groups_move_template_between_custom_groups():
    groups = [
        {"name": "农田", "templates": ["Farm"]},
        {"name": "任务", "templates": ["TaskIcon"]},
    ]

    moved = RecordVideoTemplatePage.add_template_to_group(groups, "任务", "Farm")

    assert moved == [
        {"name": "农田", "templates": []},
        {"name": "任务", "templates": ["TaskIcon", "Farm"]},
    ]
    assert RecordVideoTemplatePage.remove_template_from_group(moved, "任务", "Farm") == [
        {"name": "农田", "templates": []},
        {"name": "任务", "templates": ["TaskIcon"]},
    ]


def test_record_template_groups_move_template_to_target_or_default_group():
    groups = [
        {"name": "农田", "templates": ["Farm", "Soil"]},
        {"name": "任务", "templates": ["TaskIcon"]},
    ]

    moved = RecordVideoTemplatePage.move_template_to_group(groups, "任务", "Farm")

    assert moved == [
        {"name": "农田", "templates": ["Soil"]},
        {"name": "任务", "templates": ["TaskIcon", "Farm"]},
    ]
    assert RecordVideoTemplatePage.move_template_to_group(moved, DEFAULT_TEMPLATE_GROUP_NAME, "Farm") == [
        {"name": "农田", "templates": ["Soil"]},
        {"name": "任务", "templates": ["TaskIcon"]},
    ]


def test_record_template_groups_move_multiple_templates_to_target_or_default_group():
    groups = [
        {"name": "农田", "templates": ["Farm", "Soil", "Water"]},
        {"name": "任务", "templates": ["TaskIcon"]},
    ]

    moved = RecordVideoTemplatePage.move_templates_to_group(groups, "任务", ["Farm", "Soil"])

    assert moved == [
        {"name": "农田", "templates": ["Water"]},
        {"name": "任务", "templates": ["TaskIcon", "Farm", "Soil"]},
    ]
    assert RecordVideoTemplatePage.move_templates_to_group(moved, DEFAULT_TEMPLATE_GROUP_NAME, ["Farm", "Soil"]) == [
        {"name": "农田", "templates": ["Water"]},
        {"name": "任务", "templates": ["TaskIcon"]},
    ]


def test_record_template_groups_update_default_order_when_moving_templates():
    assert RecordVideoTemplatePage.next_default_template_order(["Farm", "Soil"], "任务", ["Farm"]) == ["Soil"]
    assert RecordVideoTemplatePage.next_default_template_order(["Farm"], DEFAULT_TEMPLATE_GROUP_NAME, ["Soil"]) == [
        "Farm",
        "Soil",
    ]


def test_record_template_groups_reorder_template_names():
    assert RecordVideoTemplatePage.reorder_template_names(["A", "B", "C", "D"], ["B", "C"], -1) == [
        "B",
        "C",
        "A",
        "D",
    ]
    assert RecordVideoTemplatePage.reorder_template_names(["A", "B", "C", "D"], ["B", "C"], 1) == [
        "A",
        "D",
        "B",
        "C",
    ]


def test_record_template_group_search_matches_template_name_or_display_name():
    assert RecordVideoTemplatePage.template_group_template_matches_search("仓库_修复完成", "修复")
    assert RecordVideoTemplatePage.template_group_template_matches_search("Farm", "规则", "规则模板  Farm")
    assert not RecordVideoTemplatePage.template_group_template_matches_search("仓库_修复完成", "农田")


def test_record_template_group_search_display_shows_filtered_count():
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    page.collapsed_template_groups = {DEFAULT_TEMPLATE_GROUP_NAME}

    assert page.template_group_display_name({
        "name": DEFAULT_TEMPLATE_GROUP_NAME,
        "templates": ["Farm"],
        "default": True,
        "searching": True,
        "total_count": 223,
    }) == "- 默认分组 (1/223)"


def test_record_template_groups_migrate_legacy_tree_config_to_flat_group():
    legacy = [
        {"name": "Parent", "children": [{"name": "Child", "children": []}]},
    ]

    assert RecordVideoTemplatePage.sanitize_template_groups(legacy) == [
        {"name": "Parent", "templates": ["Parent", "Child"]},
    ]


def test_record_template_group_names_include_visual_rules():
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    page.available_record_visual_rule_template_names = lambda: {"Farm"}

    class FeatureSet:
        feature_dict = {"Farm": object(), "Farm_1": object(), "Farm_2": object()}

        def template_names(self):
            return ["Farm", "Farm_1", "Farm_2"]

    og.executor = SimpleNamespace(feature_set=FeatureSet())

    assert page.available_record_template_names() == ["Farm", "Farm_1", "Farm_2"]
    assert page.record_template_display_name("Farm") == "规则模板  Farm"


def test_record_template_names_include_coco_categories_when_feature_set_is_stale(monkeypatch):
    monkeypatch.setattr(
        "ok.ui.qt.tasks.TemplateTab.load_coco",
        lambda: {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "空农田_1", "supercategory": ""},
                {"id": 2, "name": "空农田_2", "supercategory": ""},
            ],
        },
    )
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    page.available_record_visual_rule_template_names = lambda: set()
    og.executor = SimpleNamespace(feature_set=SimpleNamespace(template_names=lambda: ["OldTemplate"]))

    assert page.available_record_template_names() == ["OldTemplate", "空农田", "空农田_1", "空农田_2"]


def test_record_template_rule_names_include_coco_rule_samples(monkeypatch):
    monkeypatch.setattr(
        "ok.ui.qt.tasks.TemplateTab.load_coco",
        lambda: {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "空农田_1", "supercategory": ""},
                {"id": 2, "name": "空农田_2", "supercategory": ""},
                {"id": 3, "name": "空农田_3", "supercategory": ""},
            ],
        },
    )
    page = RecordVideoTemplatePage.__new__(RecordVideoTemplatePage)
    og.executor = SimpleNamespace(feature_set=None)

    assert "空农田" in page.available_record_visual_rule_template_names()


def test_preview_match_worker_rejects_high_confidence_roi_mismatch():
    template = np.zeros((20, 20, 3), dtype=np.uint8)
    template[:, :] = (20, 160, 220)
    cv2.line(template, (4, 4), (16, 16), (240, 240, 240), thickness=2)
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[10:30, 10:30] = template
    frame[10:30, 45:65] = (20, 160, 220)

    class FakeFeatureSet:
        def template_names(self):
            return ["RealButton", "WrongButton"]

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0):
            if names == "RealButton":
                return [Box(10, 10, 20, 20, 1.0, "RealButton")]
            return [Box(45, 10, 20, 20, 1.0, "WrongButton")]

        def get_feature_by_name(self, frame, name):
            return Feature(template, 0, 0)

    finished = []
    worker = PreviewMatchWorker(3, FakeFeatureSet(), frame, min_roi_similarity=0.95, max_boxes=0)
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))

    worker.run()

    assert [box.name for box in finished[0][1]] == ["RealButton"]


def test_preview_match_worker_keeps_multiple_matches_for_same_template():
    template = np.full((10, 10, 3), (40, 180, 230), dtype=np.uint8)
    frame = np.zeros((30, 50, 3), dtype=np.uint8)
    frame[5:15, 5:15] = template
    frame[5:15, 30:40] = template

    class FakeFeatureSet:
        def template_names(self):
            return ["TaskCircle"]

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0):
            assert limit == 0
            return [
                Box(5, 5, 10, 10, 1.0, "TaskCircle"),
                Box(30, 5, 10, 10, 0.99, "TaskCircle"),
            ]

        def get_feature_by_name(self, frame, name):
            return Feature(template, 0, 0)

    finished = []
    worker = PreviewMatchWorker(4, FakeFeatureSet(), frame, min_roi_similarity=0.95)
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))

    worker.run()

    assert [(box.x, box.y, box.name) for box in finished[0][1]] == [
        (5, 5, "TaskCircle"),
        (30, 5, "TaskCircle"),
    ]


def test_preview_match_worker_deduplicates_overlapping_matches():
    template = np.full((10, 10, 3), (40, 180, 230), dtype=np.uint8)
    frame = np.zeros((30, 60, 3), dtype=np.uint8)
    frame[5:15, 5:15] = template
    frame[5:15, 25:35] = template

    class FakeFeatureSet:
        def template_names(self):
            return ["TaskCircle"]

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0):
            return [
                Box(5, 5, 10, 10, 1.0, "TaskCircle"),
                Box(6, 5, 10, 10, 0.98, "TaskCircle"),
                Box(25, 5, 10, 10, 0.96, "TaskCircle"),
            ]

        def get_feature_by_name(self, frame, name):
            return Feature(template, 0, 0)

    finished = []
    worker = PreviewMatchWorker(6, FakeFeatureSet(), frame, min_roi_similarity=0.95)
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))

    worker.run()

    assert [(box.x, box.y, box.name) for box in finished[0][1]] == [
        (5, 5, "TaskCircle"),
        (25, 5, "TaskCircle"),
    ]


def test_preview_match_worker_can_prefer_visual_rule_over_same_named_template():
    template = np.full((10, 10, 3), (40, 180, 230), dtype=np.uint8)
    frame = np.zeros((30, 70, 3), dtype=np.uint8)
    frame[5:15, 30:40] = template
    visual_feature = Feature(template, 30, 5)
    rule = SimpleNamespace(feature=visual_feature)

    class FakeFeatureSet:
        def __init__(self):
            self.feature_dict = {"Farm": Feature(np.full((10, 10, 3), 10, dtype=np.uint8), 5, 5)}
            self.visual_rule_dict = {"Farm": rule}
            self.calls = []

        def template_names(self):
            return ["Farm", "Farm_1", "Farm_2"]

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0, template=None, mask_function=None):
            self.calls.append((names, template is visual_feature.mat, callable(mask_function)))
            if template is visual_feature.mat:
                return [Box(30, 5, 10, 10, 1.0, "Farm")]
            return [Box(5, 5, 10, 10, 1.0, "Farm")]

        def get_feature_by_name(self, frame, name):
            return visual_feature

    feature_set = FakeFeatureSet()
    finished = []
    worker = PreviewMatchWorker(
        5,
        feature_set,
        frame,
        selected_names=["Farm"],
        min_roi_similarity=0.95,
        prefer_visual_rules=True,
    )
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))

    worker.run()

    assert [(box.x, box.y, box.name) for box in finished[0][1]] == [(30, 5, "Farm")]
    assert feature_set.calls == [("Farm", True, True)]


def test_preview_match_worker_selected_ref_rule_forces_visual_rule_over_same_named_template():
    template = np.full((10, 10, 3), (40, 180, 230), dtype=np.uint8)
    frame = np.zeros((30, 70, 3), dtype=np.uint8)
    frame[5:15, 30:40] = template
    visual_feature = Feature(template, 30, 5)
    rule = SimpleNamespace(feature=visual_feature)

    class FakeFeatureSet:
        def __init__(self):
            self.feature_dict = {"Farm": Feature(np.full((10, 10, 3), 10, dtype=np.uint8), 5, 5)}
            self.visual_rule_dict = {"Farm": rule}
            self.calls = []

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0, template=None, mask_function=None):
            self.calls.append((names, template is visual_feature.mat))
            if template is visual_feature.mat:
                return [Box(30, 5, 10, 10, 1.0, "Farm")]
            return [Box(5, 5, 10, 10, 1.0, "Farm")]

        def get_feature_by_name(self, frame, name):
            return visual_feature

    feature_set = FakeFeatureSet()
    finished = []
    worker = PreviewMatchWorker(
        11,
        feature_set,
        frame,
        selected_refs=[{"name": "Farm", "type": TEMPLATE_REF_TYPE_RULE}],
        min_roi_similarity=0.95,
    )
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))

    worker.run()

    assert [(box.x, box.y, box.name) for box in finished[0][1]] == [(30, 5, "Farm")]
    assert feature_set.calls == [("Farm", True)]


def test_preview_match_worker_deduplicates_visual_rule_candidate_boxes_in_progress():
    rule_template = np.full((10, 10, 3), (40, 180, 230), dtype=np.uint8)
    sample_template = np.full((10, 10, 3), (42, 182, 232), dtype=np.uint8)
    frame = np.zeros((30, 70, 3), dtype=np.uint8)
    frame[5:15, 5:15] = sample_template
    rule_feature = Feature(rule_template, 5, 5)
    sample_feature = Feature(sample_template, 6, 6)
    rule = SimpleNamespace(feature=rule_feature, sample_names=["Farm_1"])

    class FakeFeatureSet:
        def __init__(self):
            self.feature_dict = {"Farm_1": sample_feature}
            self.visual_rule_dict = {"Farm": rule}
            self.calls = []

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0, template=None, mask_function=None):
            self.calls.append((names, template))
            if template is rule_template:
                return [Box(5, 5, 10, 10, 0.8, "Farm")]
            if template is sample_template:
                return [Box(6, 6, 10, 10, 0.9, "Farm_1")]
            return []

    feature_set = FakeFeatureSet()
    finished = []
    progress = []
    worker = PreviewMatchWorker(
        15,
        feature_set,
        frame,
        selected_refs=[{"name": "Farm", "type": TEMPLATE_REF_TYPE_RULE}],
        min_roi_similarity=0.0,
    )
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))
    worker.signals.progress.connect(
        lambda seq, current, total, matched, name, elapsed: progress.append((current, total, matched, name))
    )

    worker.run()

    assert [(box.x, box.y, box.confidence, box.name) for box in finished[0][1]] == [(6, 6, 0.9, "Farm")]
    assert progress[-1][2] == 1
    assert feature_set.calls == [("Farm", rule_template), ("Farm", sample_template)]


def test_preview_match_worker_rule_ref_rebuilds_visual_rules_after_lazy_normal_load():
    template = np.full((10, 10, 3), (40, 180, 230), dtype=np.uint8)
    frame = np.zeros((30, 70, 3), dtype=np.uint8)
    frame[5:15, 30:40] = template
    visual_feature = Feature(template, 30, 5)

    class FakeFeatureSet:
        def __init__(self):
            self.feature_dict = {}
            self.visual_rule_dict = {}
            self.calls = []
            self.processed_full = 0

        def ensure_feature(self, name):
            self.feature_dict[name] = Feature(np.full((10, 10, 3), 10, dtype=np.uint8), 5, 5)

        def process_data(self):
            self.processed_full += 1
            self.visual_rule_dict = {"Farm": SimpleNamespace(feature=visual_feature)}

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0, template=None, mask_function=None):
            self.calls.append((names, template is visual_feature.mat))
            if template is visual_feature.mat:
                return [Box(30, 5, 10, 10, 1.0, "Farm")]
            return [Box(5, 5, 10, 10, 1.0, "Farm")]

        def get_feature_by_name(self, frame, name):
            return visual_feature

    feature_set = FakeFeatureSet()
    finished = []
    worker = PreviewMatchWorker(
        13,
        feature_set,
        frame,
        selected_refs=[{"name": "Farm", "type": TEMPLATE_REF_TYPE_RULE}],
        min_roi_similarity=0.95,
    )
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))

    worker.run()

    assert feature_set.processed_full == 1
    assert [(box.x, box.y, box.name) for box in finished[0][1]] == [(30, 5, "Farm")]
    assert feature_set.calls == [("Farm", True)]


def test_preview_match_worker_prefer_visual_rule_rebuilds_after_lazy_normal_load():
    template = np.full((10, 10, 3), (40, 180, 230), dtype=np.uint8)
    frame = np.zeros((30, 70, 3), dtype=np.uint8)
    frame[5:15, 30:40] = template
    visual_feature = Feature(template, 30, 5)

    class FakeFeatureSet:
        def __init__(self):
            self.feature_dict = {}
            self.visual_rule_dict = {}
            self.calls = []
            self.processed_full = 0

        def ensure_feature(self, name):
            self.feature_dict[name] = Feature(np.full((10, 10, 3), 10, dtype=np.uint8), 5, 5)

        def process_data(self):
            self.processed_full += 1
            self.visual_rule_dict = {"Farm": SimpleNamespace(feature=visual_feature)}

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0, template=None, mask_function=None):
            self.calls.append((names, template is visual_feature.mat))
            if template is visual_feature.mat:
                return [Box(30, 5, 10, 10, 1.0, "Farm")]
            return [Box(5, 5, 10, 10, 1.0, "Farm")]

        def get_feature_by_name(self, frame, name):
            return visual_feature

    feature_set = FakeFeatureSet()
    finished = []
    worker = PreviewMatchWorker(
        14,
        feature_set,
        frame,
        selected_names=["Farm"],
        min_roi_similarity=0.95,
        prefer_visual_rules=True,
    )
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))

    worker.run()

    assert feature_set.processed_full == 1
    assert [(box.x, box.y, box.name) for box in finished[0][1]] == [(30, 5, "Farm")]
    assert feature_set.calls == [("Farm", True)]


def test_preview_match_worker_only_loads_visual_rules_for_possible_rule_names():
    template = np.full((10, 10, 3), (40, 180, 230), dtype=np.uint8)
    frame = np.zeros((30, 70, 3), dtype=np.uint8)
    frame[5:15, 30:40] = template
    visual_feature = Feature(template, 30, 5)

    class FakeFeatureSet:
        def __init__(self):
            self.feature_dict = {}
            self.visual_rule_dict = {}
            self.calls = []
            self.processed_full = 0

        def possible_visual_rule_names(self):
            return {"Farm"}

        def process_data(self):
            self.processed_full += 1
            self.visual_rule_dict = {"Farm": SimpleNamespace(feature=visual_feature)}

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0, template=None, mask_function=None):
            self.calls.append((names, template is not None))
            if template is visual_feature.mat:
                return [Box(30, 5, 10, 10, 1.0, "Farm")]
            return []

        def get_feature_by_name(self, frame, name):
            return visual_feature

    feature_set = FakeFeatureSet()
    finished = []
    worker = PreviewMatchWorker(
        16,
        feature_set,
        frame,
        selected_names=["Plain", "Farm", "Other"],
        min_roi_similarity=0.95,
        prefer_visual_rules=True,
    )
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))

    worker.run()

    assert feature_set.processed_full == 1
    assert [(box.x, box.y, box.name) for box in finished[0][1]] == [(30, 5, "Farm")]
    assert feature_set.calls == [("Plain", False), ("Farm", True), ("Other", False)]


def test_preview_match_worker_selected_ref_template_avoids_visual_rule_same_named_template():
    template = np.full((10, 10, 3), (40, 180, 230), dtype=np.uint8)
    frame = np.zeros((30, 70, 3), dtype=np.uint8)
    frame[5:15, 5:15] = template
    visual_feature = Feature(template, 30, 5)
    rule = SimpleNamespace(feature=visual_feature)

    class FakeFeatureSet:
        def __init__(self):
            self.feature_dict = {"Farm": Feature(template, 5, 5)}
            self.visual_rule_dict = {"Farm": rule}
            self.calls = []

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0, template=None, mask_function=None):
            self.calls.append((names, template is None))
            if template is None:
                return [Box(5, 5, 10, 10, 1.0, "Farm")]
            return [Box(30, 5, 10, 10, 1.0, "Farm")]

        def get_feature_by_name(self, frame, name):
            return self.feature_dict[name]

    feature_set = FakeFeatureSet()
    finished = []
    worker = PreviewMatchWorker(
        12,
        feature_set,
        frame,
        selected_refs=[{"name": "Farm", "type": TEMPLATE_REF_TYPE_TEMPLATE}],
        min_roi_similarity=0.95,
        prefer_visual_rules=True,
    )
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))

    worker.run()

    assert [(box.x, box.y, box.name) for box in finished[0][1]] == [(5, 5, "Farm")]
    assert feature_set.calls == [("Farm", True)]


def test_preview_match_worker_matches_visual_rule_samples_as_candidates():
    rule_template = np.full((10, 10, 3), (10, 10, 10), dtype=np.uint8)
    sample_template = np.full((10, 10, 3), (40, 180, 230), dtype=np.uint8)
    frame = np.zeros((30, 60, 3), dtype=np.uint8)
    frame[5:15, 5:15] = sample_template
    rule_feature = Feature(rule_template, 0, 0)
    sample_feature = Feature(sample_template, 5, 5)
    rule = SimpleNamespace(feature=rule_feature, sample_names=["Farm_1"])

    class FakeFeatureSet:
        def __init__(self):
            self.feature_dict = {"Farm_1": sample_feature}
            self.visual_rule_dict = {"Farm": rule}

        def template_names(self):
            return ["Farm"]

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0, template=None, mask_function=None):
            if template is rule_template:
                return []
            if template is sample_template:
                return [Box(5, 5, 10, 10, 1.0, "Farm_1")]
            return []

    finished = []
    worker = PreviewMatchWorker(
        7,
        FakeFeatureSet(),
        frame,
        selected_names=["Farm"],
        min_roi_similarity=0.95,
        prefer_visual_rules=True,
    )
    worker.signals.finished.connect(lambda seq, boxes: finished.append((seq, boxes)))

    worker.run()

    assert [(box.x, box.y, box.name) for box in finished[0][1]] == [(5, 5, "Farm")]


def test_validation_template_name_uses_empty_text_over_stale_combo_data():
    app()
    from PySide6.QtWidgets import QComboBox

    combo = QComboBox()
    combo.setEditable(True)
    combo.addItem("规则模板  Farm", "Farm")
    combo.setCurrentIndex(0)

    assert PreviewMatchWorker is not None
    from ok.ui.qt.tasks.VideoTemplateTab import EditVideoTemplatePage

    assert EditVideoTemplatePage.validation_combo_template_name(combo) == "Farm"
    combo.setEditText("")

    assert EditVideoTemplatePage.validation_combo_template_name(combo) == ""


def test_validation_template_name_supports_structured_combo_data():
    app()
    from PySide6.QtWidgets import QComboBox
    from ok.ui.qt.tasks.VideoTemplateTab import EditVideoTemplatePage, add_template_choice_to_combo

    combo = QComboBox()
    combo.setEditable(True)
    add_template_choice_to_combo(combo, "Farm", True, "规则模板")
    combo.setCurrentIndex(0)

    assert EditVideoTemplatePage.validation_combo_template_name(combo) == "Farm"
    assert EditVideoTemplatePage.validation_combo_template_ref(combo) == {"name": "Farm", "type": "rule"}
    assert EditVideoTemplatePage.find_validation_template_combo_index(combo, "Farm") == 0


def test_validation_template_ref_strips_rule_display_text_when_combo_index_is_stale():
    app()
    from PySide6.QtWidgets import QComboBox
    from ok.ui.qt.tasks.VideoTemplateTab import EditVideoTemplatePage, add_template_choice_to_combo

    combo = QComboBox()
    combo.setEditable(True)
    add_template_choice_to_combo(combo, "Farm", True, "规则模板")
    combo.setCurrentIndex(-1)
    combo.setEditText("规则模板  Farm")

    assert EditVideoTemplatePage.validation_combo_template_ref(combo) == {"name": "Farm", "type": "rule"}
    assert EditVideoTemplatePage.validation_combo_template_name(combo) == "Farm"


def test_validation_match_worker_receives_structured_rule_ref(monkeypatch):
    app()
    from ok.ui.qt.tasks import VideoTemplateTab as video_tab
    from ok.ui.qt.tasks.VideoTemplateTab import EditVideoTemplatePage, add_template_choice_to_combo

    combo = QComboBox()
    combo.setEditable(True)
    add_template_choice_to_combo(combo, "Farm", True, "规则模板")
    combo.setCurrentIndex(-1)
    combo.setEditText("规则模板  Farm")

    captured = {}

    class FakeSignal:
        def connect(self, _slot):
            pass

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            self.signals = SimpleNamespace(
                finished=FakeSignal(),
                failed=FakeSignal(),
                progress=FakeSignal(),
            )

    started = []
    pool = SimpleNamespace(start=lambda worker: started.append(worker))
    monkeypatch.setattr(video_tab, "PreviewMatchWorker", FakeWorker)
    monkeypatch.setattr(video_tab, "QThreadPool", SimpleNamespace(globalInstance=lambda: pool))
    monkeypatch.setattr(og, "executor", SimpleNamespace(feature_set=SimpleNamespace()), raising=False)

    page = EditVideoTemplatePage.__new__(EditVideoTemplatePage)
    page.validation_template_combo = combo
    page.current_preview_frame = np.zeros((20, 20, 3), dtype=np.uint8)
    page.validation_match_worker = None
    page.validation_match_seq = 0
    page.validation_match_progress = None
    page.validation_match_timer = SimpleNamespace(start=lambda _ms: None)
    page.validation_status_label = SimpleNamespace(setText=lambda _text: None)
    page.validation_target_height = lambda _frame: 0
    page.validation_confidence_threshold = lambda: 0.72

    page.run_template_validation_match()

    assert captured["kwargs"]["selected_refs"] == [{"name": "Farm", "type": "rule"}]
    assert captured["kwargs"]["selected_names"] is None
    assert captured["kwargs"]["prefer_visual_rules"] is False
    assert len(started) == 1


def test_export_prefers_generated_rule_name_for_validation_combo():
    from ok.ui.qt.tasks.VideoTemplateTab import EditVideoTemplatePage

    assert EditVideoTemplatePage.template_name_to_validate_from_rules(["Farm_1", "Farm_2"], {"Farm"}) == "Farm"
    assert EditVideoTemplatePage.template_name_to_validate_from_rules(["SingleTemplate"], {"Farm"}) == "SingleTemplate"


def test_rule_template_names_include_same_named_normal_template():
    from ok.ui.qt.tasks.VideoTemplateTab import EditVideoTemplatePage

    names = ["Farm", "Farm_1", "Farm_2", "OnlyOne_1"]

    assert EditVideoTemplatePage.rule_template_names_from_samples(names) == {"Farm"}


def test_preview_match_worker_reports_progress():
    template = np.full((10, 10, 3), (40, 180, 230), dtype=np.uint8)
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[5:15, 5:15] = template

    class FakeFeatureSet:
        def template_names(self):
            return ["Missing", "Hit"]

        def find_feature(self, frame, names, threshold=0, limit=0, target_height=0):
            if names == "Hit":
                return [Box(5, 5, 10, 10, 1.0, "Hit")]
            return []

        def get_feature_by_name(self, frame, name):
            return Feature(template, 0, 0)

    progress = []
    worker = PreviewMatchWorker(4, FakeFeatureSet(), frame, min_roi_similarity=0.95)
    worker.signals.progress.connect(
        lambda seq, current, total, matched, name, elapsed: progress.append((seq, current, total, matched, name))
    )

    worker.run()

    assert progress[0] == (4, 0, 2, 0, "")
    assert progress[-1] == (4, 2, 2, 1, "Hit")


def _draw_rule_sample(image, x, inner_color):
    center = (x + 18, 38)
    cv2.circle(image, center, 16, (40, 180, 90), thickness=-1)
    cv2.circle(image, center, 16, (230, 240, 220), thickness=3)
    cv2.rectangle(image, (x + 12, 32), (x + 24, 44), inner_color, thickness=-1)


def _write_rule_coco(coco_path, image_name, second_category_name):
    coco_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": image_name, "width": 120, "height": 80}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 20, 36, 36]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [65, 20, 36, 36]},
        ],
        "categories": [
            {"id": 1, "name": "TaskIcon_A", "supercategory": ""},
            {"id": 2, "name": second_category_name, "supercategory": ""},
        ],
    }, ensure_ascii=False), encoding="utf-8")


def test_visual_rules_load_from_upstream_split_template_samples(tmp_path):
    from ok.feature.FeatureSet import generate_templates

    source = tmp_path / "source"
    source.mkdir()
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    _draw_rule_sample(image, 5, (255, 80, 80))
    _draw_rule_sample(image, 65, (80, 120, 255))
    image_path = source / "screen.png"
    assert cv2.imwrite(str(image_path), image)
    coco_path = source / "coco_annotations.json"
    _write_rule_coco(coco_path, image_path.name, "TaskIcon_B")
    crops = generate_templates(str(coco_path), str(tmp_path / "templates"))
    feature_set = FeatureSet(False, None, 1, 1, template_folder=str(crops))
    feature_set.check_size(image)

    assert "TaskIcon" in feature_set.template_names()
    assert feature_set.visual_rule_dict["TaskIcon"].sample_names == ["TaskIcon_A", "TaskIcon_B"]
    assert feature_set.get_feature_by_name(image, "TaskIcon") is not None
