import time

import cv2
from PySide6.QtCore import QObject, QRunnable, Signal

from ok.feature.Box import box_overlap_ratio, deduplicate_boxes
from ok.feature.TemplateRef import TEMPLATE_REF_TYPE_RULE, TEMPLATE_REF_TYPE_TEMPLATE
from ok.ui.qt.video_template.template_sources import sanitize_template_refs
from ok.util.logger import Logger


logger = Logger.get_logger(__name__)


class PreviewMatchSignals(QObject):
    finished = Signal(int, list)
    failed = Signal(int, str)
    progress = Signal(int, int, int, int, str, float)
    detail = Signal(int, dict)


class PreviewMatchWorker(QRunnable):
    def __init__(self, seq, feature_set, frame, selected_names=None, target_height=0,
                 confidence_threshold=0.72, max_names=0, max_boxes=0, min_roi_similarity=0.88,
                 matches_per_template=0, horizontal_variance=0, vertical_variance=0,
                 prefer_visual_rules=False, match_groups=None, selected_refs=None):
        super().__init__()
        self.seq = seq
        self.feature_set = feature_set
        self.frame = frame
        self.selected_names = selected_names
        self.selected_refs = sanitize_template_refs(selected_refs) if selected_refs is not None else None
        self.match_groups = match_groups
        self.target_height = target_height
        self.confidence_threshold = confidence_threshold
        self.max_names = max_names
        self.max_boxes = max_boxes
        self.min_roi_similarity = min_roi_similarity
        self.matches_per_template = matches_per_template
        self.horizontal_variance = horizontal_variance
        self.vertical_variance = vertical_variance
        self.prefer_visual_rules = prefer_visual_rules
        self._visual_rules_load_attempted = False
        self._visual_rules_known_loaded = False
        self.signals = PreviewMatchSignals()

    def run(self):
        try:
            use_refs = self.selected_refs is not None
            use_groups = self.match_groups is not None and self.selected_names is None and not use_refs
            names = self.selected_names
            if use_refs:
                names = [ref["name"] for ref in self.selected_refs]
            if names is None:
                if hasattr(self.feature_set, "template_names"):
                    names = list(self.feature_set.template_names())
                else:
                    names = list(getattr(self.feature_set, "feature_dict", {}).keys())
                if not names:
                    self.feature_set.process_data()
                    if hasattr(self.feature_set, "template_names"):
                        names = list(self.feature_set.template_names())
                    else:
                        names = list(getattr(self.feature_set, "feature_dict", {}).keys())
            names = [name for name in names if name]
            group_names = self.flatten_match_group_names(self.match_groups) if use_groups else []
            if self.max_names > 0 and not use_groups:
                names = names[:self.max_names]
            if not names and not group_names:
                self.signals.finished.emit(self.seq, [])
                return
            if use_refs:
                boxes = self.collect_preview_ref_boxes(self.selected_refs)
            elif use_groups:
                boxes = self.collect_preview_group_boxes(self.match_groups)
            else:
                boxes = self.collect_preview_boxes(names)
            boxes = self.limit_preview_boxes(boxes)
            self.signals.finished.emit(self.seq, boxes or [])
        except Exception as e:
            logger.debug(f"template preview matching failed: {e}")
            self.signals.failed.emit(self.seq, str(e))

    def collect_preview_boxes(self, names):
        boxes = []
        total = len(names)
        start_time = time.time()
        self.signals.progress.emit(self.seq, 0, total, 0, "", 0.0)
        for index, name in enumerate(names, 1):
            elapsed_ms = (time.time() - start_time) * 1000
            self.emit_preview_detail("", name, boxes, index, total, elapsed_ms, searching=True)
            matched = self.find_preview_boxes([name])
            boxes.extend(self.filter_preview_boxes(matched))
            boxes = self.sorted_unique_preview_boxes(boxes)
            elapsed_ms = (time.time() - start_time) * 1000
            self.signals.progress.emit(self.seq, index, total, len(boxes), name, elapsed_ms)
            self.emit_preview_detail("", name, boxes, index, total, elapsed_ms)
        return boxes

    def collect_preview_ref_boxes(self, refs):
        boxes = []
        total = len(refs)
        start_time = time.time()
        self.signals.progress.emit(self.seq, 0, total, 0, "", 0.0)
        for index, ref in enumerate(refs, 1):
            name = str(ref.get("name") or "").strip()
            elapsed_ms = (time.time() - start_time) * 1000
            self.emit_preview_detail("", name, boxes, index, total, elapsed_ms, searching=True)
            matched = self.find_preview_boxes_for_ref(ref)
            boxes.extend(self.filter_preview_boxes(matched))
            boxes = self.sorted_unique_preview_boxes(boxes)
            elapsed_ms = (time.time() - start_time) * 1000
            self.signals.progress.emit(self.seq, index, total, len(boxes), name, elapsed_ms)
            self.emit_preview_detail("", name, boxes, index, total, elapsed_ms)
        return boxes

    def collect_preview_group_boxes(self, groups):
        boxes = []
        flat_names = self.flatten_match_group_names(groups)
        total = len(flat_names)
        visited = 0
        start_time = time.time()
        self.signals.progress.emit(self.seq, 0, total, 0, "", 0.0)

        for group in groups or []:
            group_name = str((group or {}).get("name") or "").strip()
            group_boxes = []
            names = []
            seen = set()
            for name in (group or {}).get("templates") or []:
                name = str(name or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                names.append(name)
            if not names:
                continue
            for name in names:
                visited += 1
                elapsed_ms = (time.time() - start_time) * 1000
                self.emit_preview_detail(group_name, name, boxes, visited, total, elapsed_ms, searching=True)
                matched = self.filter_preview_boxes(self.find_preview_boxes([name]))
                group_boxes.extend(matched)
                boxes.extend(matched)
                group_boxes = self.sorted_unique_preview_boxes(group_boxes)
                boxes = self.sorted_unique_preview_boxes(boxes)
                elapsed_ms = (time.time() - start_time) * 1000
                display_name = f"{group_name} / {name}" if group_name else name
                self.signals.progress.emit(self.seq, visited, total, len(boxes), display_name, elapsed_ms)
                self.emit_preview_detail(group_name, name, boxes, visited, total, elapsed_ms)
            if group_boxes:
                skipped = total - visited
                if skipped > 0:
                    elapsed_ms = (time.time() - start_time) * 1000
                    name = self.truncated_group_progress_name(group_name)
                    self.signals.progress.emit(self.seq, total, total, len(boxes), name, elapsed_ms)
                    self.emit_preview_detail(group_name, "", boxes, total, total, elapsed_ms, skipped=True)
                return boxes
        return boxes

    def emit_preview_detail(self, group_name, template_name, boxes, current, total, elapsed_ms, skipped=False, searching=False):
        hit_counts = {}
        for box in boxes or []:
            name = str(getattr(box, "name", "") or "").strip()
            if not name:
                continue
            hit_counts[name] = hit_counts.get(name, 0) + 1
        self.signals.detail.emit(self.seq, {
            "group": str(group_name or "").strip(),
            "template": str(template_name or "").strip(),
            "current": int(current or 0),
            "total": int(total or 0),
            "hit_counts": hit_counts,
            "matched_count": len(boxes or []),
            "elapsed_ms": float(elapsed_ms or 0),
            "skipped": bool(skipped),
            "searching": bool(searching),
        })

    @staticmethod
    def truncated_group_progress_name(group_name):
        group_name = str(group_name or "").strip()
        return f"{group_name} 已命中，后续分组已跳过" if group_name else "当前分组已命中，后续分组已跳过"

    def flatten_match_group_names(self, groups):
        names = []
        for group in groups or []:
            seen = set()
            for name in (group or {}).get("templates") or []:
                name = str(name or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                names.append(name)
        return names

    def find_preview_boxes(self, names):
        boxes = []
        for name in names:
            try:
                if self.prefer_visual_rules:
                    visual_rule_boxes = self.find_visual_rule_boxes(name)
                    if visual_rule_boxes is not None:
                        boxes.extend(visual_rule_boxes)
                        continue
                boxes.extend(self.find_normal_template_boxes(name))
            except Exception as e:
                logger.debug(f"template preview matching skipped {name}: {e}")
        return boxes

    def find_preview_boxes_for_ref(self, ref):
        name = str((ref or {}).get("name") or "").strip()
        ref_type = str((ref or {}).get("type") or TEMPLATE_REF_TYPE_TEMPLATE).strip()
        if not name:
            return []
        try:
            if ref_type == TEMPLATE_REF_TYPE_RULE:
                visual_rule_boxes = self.find_visual_rule_boxes(name)
                return visual_rule_boxes or []
            return self.find_normal_template_boxes(name)
        except Exception as e:
            logger.debug(f"template preview matching skipped {name} ({ref_type}): {e}")
            return []

    def find_normal_template_boxes(self, name):
        variance_kwargs = {}
        if self.horizontal_variance:
            variance_kwargs["horizontal_variance"] = self.horizontal_variance
        if self.vertical_variance:
            variance_kwargs["vertical_variance"] = self.vertical_variance
        return self.feature_set.find_feature(
            self.frame,
            name,
            threshold=self.confidence_threshold,
            limit=self.matches_per_template,
            target_height=self.target_height,
            **variance_kwargs,
        ) or []

    def find_visual_rule_boxes(self, name):
        check_size = getattr(self.feature_set, "check_size", None)
        if callable(check_size):
            check_size(self.frame)
        visual_rules = getattr(self.feature_set, "visual_rule_dict", None)
        if not isinstance(visual_rules, dict):
            visual_rules = {}
        if name not in visual_rules:
            if not self.should_load_visual_rules_for_name(name):
                return None
            visual_rules = self.ensure_visual_rules_loaded()
        if not isinstance(visual_rules, dict) or name not in visual_rules:
            return None
        rule = visual_rules[name]
        feature = getattr(rule, "feature", None)
        template = getattr(feature, "mat", None)
        if feature is None or template is None:
            return []
        matched = []
        for match_feature in self.visual_rule_match_features(rule):
            match_template = getattr(match_feature, "mat", None)
            if match_template is None:
                continue

            def mask_function(_template, current_feature=match_feature):
                return getattr(current_feature, "mask", None)

            feature_boxes = self.feature_set.find_feature(
                self.frame,
                name,
                threshold=self.confidence_threshold,
                limit=self.matches_per_template,
                target_height=self.target_height,
                template=match_template,
                mask_function=mask_function,
            )
            for box in feature_boxes or []:
                setattr(box, "matched_feature", match_feature)
                box.name = name
            matched.extend(feature_boxes or [])
        return matched

    def should_load_visual_rules_for_name(self, name):
        name = str(name or "").strip()
        if not name or self.visual_rules_ready():
            return False
        possible_names = getattr(self.feature_set, "possible_visual_rule_names", None)
        if callable(possible_names):
            try:
                return name in possible_names()
            except Exception as e:
                logger.debug(f"template preview cannot inspect visual rule names: {e}")
        return not self._visual_rules_load_attempted

    def visual_rules_ready(self):
        if self._visual_rules_known_loaded or self._visual_rules_load_attempted:
            return True
        loaded = getattr(self.feature_set, "visual_rules_loaded", None)
        if callable(loaded):
            try:
                self._visual_rules_known_loaded = bool(loaded())
            except Exception as e:
                logger.debug(f"template preview cannot inspect visual rule load state: {e}")
        elif getattr(self.feature_set, "_visual_rule_source_fingerprint", None) is not None:
            self._visual_rules_known_loaded = True
        return self._visual_rules_known_loaded

    def ensure_visual_rules_loaded(self):
        self._visual_rules_load_attempted = True
        ensure_visual_rules = getattr(self.feature_set, "ensure_visual_rules", None)
        if callable(ensure_visual_rules):
            try:
                visual_rules = ensure_visual_rules()
                self._visual_rules_known_loaded = True
                return visual_rules
            except Exception as e:
                logger.debug(f"template preview cannot load visual rules: {e}")
                return getattr(self.feature_set, "visual_rule_dict", None)
        process_data = getattr(self.feature_set, "process_data", None)
        if callable(process_data):
            process_data()
            self._visual_rules_known_loaded = True
        return getattr(self.feature_set, "visual_rule_dict", None)

    def visual_rule_match_features(self, rule):
        features = []
        rule_feature = getattr(rule, "feature", None)
        if rule_feature is not None:
            features.append(rule_feature)
        feature_dict = getattr(self.feature_set, "feature_dict", {}) or {}
        for sample_name in getattr(rule, "sample_names", []) or []:
            sample_feature = feature_dict.get(sample_name)
            if sample_feature is not None and sample_feature not in features:
                features.append(sample_feature)
        return features

    def limit_preview_boxes(self, boxes):
        boxes = self.sorted_unique_preview_boxes(boxes)
        if self.max_boxes > 0:
            boxes = boxes[:self.max_boxes]
        return boxes

    def sorted_unique_preview_boxes(self, boxes):
        boxes = sorted(boxes or [], key=lambda box: float(getattr(box, "confidence", 0) or 0), reverse=True)
        return self.deduplicate_preview_boxes(boxes)

    def deduplicate_preview_boxes(self, boxes, overlap_threshold=0.5):
        return deduplicate_boxes(boxes, overlap_threshold)

    @staticmethod
    def box_overlap_ratio(a, b):
        return box_overlap_ratio(a, b)

    def filter_preview_boxes(self, boxes):
        filtered = []
        for box in boxes or []:
            similarity = self.preview_roi_similarity(box)
            if similarity >= self.min_roi_similarity:
                filtered.append(box)
            else:
                logger.debug(
                    f"template preview rejected {getattr(box, 'name', '')}: "
                    f"confidence={float(getattr(box, 'confidence', 0) or 0):.3f}, roi_similarity={similarity:.3f}"
                )
        return filtered

    def preview_roi_similarity(self, box):
        if not hasattr(self.feature_set, "get_feature_by_name"):
            return 1.0
        feature = getattr(box, "matched_feature", None)
        if feature is None:
            try:
                feature = self.feature_set.get_feature_by_name(self.frame, box.name)
            except Exception as e:
                logger.debug(f"template preview cannot load feature {getattr(box, 'name', '')}: {e}")
                return 0.0
        template = getattr(feature, "mat", None)
        if template is None:
            return 1.0
        x1 = max(0, int(box.x))
        y1 = max(0, int(box.y))
        x2 = min(self.frame.shape[1], x1 + max(1, int(box.width)))
        y2 = min(self.frame.shape[0], y1 + max(1, int(box.height)))
        roi = self.frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        template = template[:, :, :3] if template.ndim == 3 else template
        roi = roi[:, :, :3] if roi.ndim == 3 else roi
        if roi.shape[:2] != template.shape[:2]:
            roi = cv2.resize(roi, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
        if roi.ndim != template.ndim:
            if roi.ndim == 2 and template.ndim == 3:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            elif roi.ndim == 3 and template.ndim == 2:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(roi, template)
        mask = getattr(feature, "mask", None)
        if mask is not None and mask.shape[:2] == diff.shape[:2] and cv2.countNonZero(mask) > 0:
            values = diff[mask > 0]
        else:
            values = diff
        return 1.0 - float(values.mean()) / 255.0
