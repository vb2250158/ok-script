import os

from ok import og
from ok.feature.TemplateRef import (
    sanitize_template_ref as sanitize_template_ref_value,
    sanitize_template_refs as sanitize_template_ref_values,
    template_choice_refs_from_names as base_template_choice_refs_from_names,
    template_ref_key as template_ref_value_key,
    visual_rule_template_names_from_samples,
)
from ok.feature.VisualRuleTemplate import template_names_with_visual_rules
from ok.util.logger import Logger


logger = Logger.get_logger(__name__)


def ensure_feature_set_uses_template_coco(feature_set):
    if feature_set is None:
        return None
    try:
        from ok.ui.qt.tasks.TemplateTab import get_coco_path

        target_coco = os.path.abspath(get_coco_path())
        current_coco = os.path.abspath(str(getattr(feature_set, "coco_json", "") or ""))
        if not target_coco or current_coco == target_coco:
            return feature_set
        set_coco_json = getattr(feature_set, "set_coco_json", None)
        if callable(set_coco_json):
            changed = set_coco_json(target_coco)
            if changed:
                logger.info(f"Switched runtime template COCO to workspace path: {target_coco}")
        elif hasattr(feature_set, "coco_json"):
            feature_set.coco_json = target_coco
            for attr, value in [
                ("feature_dict", {}),
                ("box_dict", {}),
                ("visual_rule_dict", {}),
                ("_processed_images", set()),
                ("_visual_rule_source_fingerprint", None),
            ]:
                if hasattr(feature_set, attr):
                    setattr(feature_set, attr, value)
            logger.info(f"Updated runtime template COCO to workspace path: {target_coco}")
    except Exception as e:
        logger.warning(f"Could not sync runtime template COCO path: {e}")
    return feature_set


def template_name_groups(names):
    grouped = set()
    for name in names or []:
        base, sep, suffix = str(name).rpartition("_")
        if sep and suffix.isdigit() and base:
            grouped.add(base)
    return grouped


def runtime_template_names():
    names = set()
    try:
        feature_set = getattr(getattr(og, "executor", None), "feature_set", None)
        feature_set = ensure_feature_set_uses_template_coco(feature_set)
        if feature_set is not None:
            if hasattr(feature_set, "template_names"):
                names.update(feature_set.template_names())
            else:
                names.update(getattr(feature_set, "feature_dict", {}).keys())
    except Exception as e:
        logger.debug(f"failed to load runtime template names: {e}")
    return {str(name) for name in names if name}


def coco_template_names():
    try:
        from ok.ui.qt.tasks.TemplateTab import load_coco
        return {
            str(category.get("name") or "").strip()
            for category in load_coco().get("categories", [])
            if str(category.get("name") or "").strip()
        }
    except Exception as e:
        logger.debug(f"failed to load coco template names: {e}")
    return set()


def visual_rule_template_names(choices=None):
    rule_names = set()
    try:
        feature_set = getattr(getattr(og, "executor", None), "feature_set", None)
        feature_set = ensure_feature_set_uses_template_coco(feature_set)
        if feature_set is not None:
            if hasattr(feature_set, "template_names"):
                feature_set.template_names()
            rule_names.update(getattr(feature_set, "visual_rule_dict", {}).keys())
            rule_names.update(visual_rule_template_names_from_samples(getattr(feature_set, "feature_dict", {}).keys()))
    except Exception as e:
        logger.debug(f"failed to load runtime visual rule names: {e}")
    rule_names.update(visual_rule_template_names_from_samples(coco_template_names()))
    rule_names = {str(name) for name in rule_names if name}
    if choices is not None:
        choice_set = {str(choice) for choice in choices if choice}
        rule_names = {name for name in rule_names if name in choice_set}
    return rule_names


def template_choice_names():
    names = set()
    runtime_names = runtime_template_names()
    names.update(runtime_names)
    names.update(coco_template_names())
    names.update(template_names_with_visual_rules(coco_template_names()))
    names.update(template_name_groups(names))
    return sorted(str(name) for name in names if name)


def template_choice_refs():
    raw_names = set(runtime_template_names())
    raw_names.update(coco_template_names())
    rule_names = visual_rule_template_names()
    return base_template_choice_refs_from_names(raw_names, rule_names)


def sanitize_template_ref(value, rule_names=None):
    return sanitize_template_ref_value(
        value,
        rule_names if rule_names is not None else visual_rule_template_names(),
    )


def sanitize_template_refs(values, rule_names=None):
    return sanitize_template_ref_values(
        values,
        rule_names if rule_names is not None else visual_rule_template_names(),
    )


def template_ref_key(ref):
    ref = sanitize_template_ref(ref)
    return template_ref_value_key(ref)
