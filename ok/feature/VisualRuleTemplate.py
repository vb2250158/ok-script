from dataclasses import dataclass

import cv2
import numpy as np

from ok.feature.Box import Box
from ok.feature.Feature import Feature


MIN_VISUAL_RULE_SAMPLES = 2
MIN_STABLE_PIXEL_RATIO = 0.08
MAX_STABLE_COLOR_STD = 35.0


@dataclass
class VisualRuleTemplate:
    name: str
    feature: Feature
    box: Box
    sample_names: list[str]
    stable_pixel_ratio: float


def visual_rule_name_for_template(template_name):
    if not template_name or "_" not in str(template_name):
        return None
    prefix, _suffix = str(template_name).rsplit("_", 1)
    return prefix or None


def template_names_with_visual_rules(template_names, min_samples=MIN_VISUAL_RULE_SAMPLES):
    names = {str(name) for name in template_names if name}
    group_counts = {}
    for name in names:
        rule_name = visual_rule_name_for_template(name)
        if rule_name:
            group_counts[rule_name] = group_counts.get(rule_name, 0) + 1
    for rule_name, count in group_counts.items():
        if count >= min_samples:
            names.add(rule_name)
    return sorted(names)


def build_visual_rule_templates(feature_dict, box_dict, min_samples=MIN_VISUAL_RULE_SAMPLES):
    groups = {}
    for sample_name, feature in feature_dict.items():
        rule_name = visual_rule_name_for_template(sample_name)
        if not rule_name:
            continue
        groups.setdefault(rule_name, []).append((sample_name, feature, box_dict.get(sample_name)))

    rules = {}
    for rule_name, samples in groups.items():
        if len(samples) < min_samples:
            continue
        rule = _build_visual_rule_template(rule_name, samples)
        if rule is not None:
            rules[rule_name] = rule
    return rules


def _build_visual_rule_template(rule_name, samples):
    valid_samples = [
        (name, feature, box)
        for name, feature, box in samples
        if feature is not None and getattr(feature, "mat", None) is not None and box is not None
    ]
    if len(valid_samples) < MIN_VISUAL_RULE_SAMPLES:
        return None

    widths = [max(1, feature.width) for _name, feature, _box in valid_samples]
    heights = [max(1, feature.height) for _name, feature, _box in valid_samples]
    target_width = int(round(float(np.median(widths))))
    target_height = int(round(float(np.median(heights))))
    if target_width <= 0 or target_height <= 0:
        return None

    mats = []
    edge_mats = []
    for _name, feature, _box in valid_samples:
        mat = feature.mat[:, :, :3]
        if mat.shape[1] != target_width or mat.shape[0] != target_height:
            mat = cv2.resize(mat, (target_width, target_height), interpolation=cv2.INTER_AREA)
        mats.append(mat)
        gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        edge_mats.append(cv2.Canny(gray, 60, 140))

    stack = np.stack(mats).astype(np.float32)
    median_template = np.median(stack, axis=0).astype(np.uint8)
    color_std = np.mean(np.std(stack, axis=0), axis=2)
    stable_color_mask = color_std <= MAX_STABLE_COLOR_STD

    edge_stack = np.stack(edge_mats).astype(np.float32) / 255.0
    stable_edge_mask = np.mean(edge_stack, axis=0) >= 0.5
    mask = np.logical_or(stable_color_mask, stable_edge_mask).astype(np.uint8) * 255
    mask = _clean_stable_mask(mask)

    stable_pixel_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if stable_pixel_ratio < MIN_STABLE_PIXEL_RATIO:
        return None

    avg_x = int(round(sum(box.x for _name, _feature, box in valid_samples) / len(valid_samples)))
    avg_y = int(round(sum(box.y for _name, _feature, box in valid_samples) / len(valid_samples)))
    feature = Feature(median_template, avg_x, avg_y, 1)
    feature.mask = mask
    box = Box(avg_x, avg_y, target_width, target_height, name=rule_name)
    return VisualRuleTemplate(
        name=rule_name,
        feature=feature,
        box=box,
        sample_names=[name for name, _feature, _box in valid_samples],
        stable_pixel_ratio=stable_pixel_ratio,
    )


def _clean_stable_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
