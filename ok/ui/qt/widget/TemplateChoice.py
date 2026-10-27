from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont

from ok.feature.TemplateRef import (
    TEMPLATE_REF_TYPE_RULE,
    TEMPLATE_REF_TYPE_TEMPLATE,
    sanitize_template_ref,
    template_param_value,
    template_ref,
    template_ref_key,
)


def template_choice_display_name(name, is_rule_template):
    return f"规则模板  {name}" if is_rule_template else str(name)


def split_template_choice_display_text(text):
    text = str(text or "").strip()
    prefix = "规则模板"
    if text.startswith(prefix):
        suffix = text[len(prefix):]
        if suffix and suffix[0].isspace():
            return suffix.strip(), True
    return text, False


def template_name_from_choice_display_text(text):
    return split_template_choice_display_text(text)[0]


def is_rule_template_choice_display_text(text):
    return split_template_choice_display_text(text)[1]


def template_ref_from_choice_display_text(text):
    name, is_rule = split_template_choice_display_text(text)
    if not name:
        return None
    return template_ref(name, TEMPLATE_REF_TYPE_RULE if is_rule else TEMPLATE_REF_TYPE_TEMPLATE)


def add_template_choice_to_combo(combo, name, is_rule_template, tooltip_text="", data=None):
    name = str(name or "").strip()
    if data is None:
        data = template_ref(name, TEMPLATE_REF_TYPE_RULE if is_rule_template else TEMPLATE_REF_TYPE_TEMPLATE)
    combo.addItem(template_choice_display_name(name, is_rule_template), data)
    index = combo.count() - 1
    if not is_rule_template:
        return
    combo.setItemData(index, QColor(0, 120, 215), Qt.ForegroundRole)
    combo.setItemData(index, QColor(225, 241, 255), Qt.BackgroundRole)
    font = QFont(combo.font())
    font.setBold(True)
    combo.setItemData(index, font, Qt.FontRole)
    if tooltip_text:
        combo.setItemData(index, tooltip_text, Qt.ToolTipRole)


def add_template_ref_choice_to_combo(combo, ref, tooltip_text=""):
    ref = sanitize_template_ref(ref)
    if not ref:
        return
    add_template_choice_to_combo(
        combo,
        ref["name"],
        ref["type"] == TEMPLATE_REF_TYPE_RULE,
        tooltip_text,
        data=dict(ref),
    )


def populate_template_combo(combo, choices, tooltip_text=""):
    combo.clear()
    for choice in choices or []:
        if isinstance(choice, dict):
            add_template_ref_choice_to_combo(combo, choice, tooltip_text)
        else:
            combo.addItem(str(choice), str(choice))


def template_name_from_combo_data(data):
    if isinstance(data, dict):
        return str(data.get("name") or "").strip()
    return str(data or "").strip()


def template_ref_from_combo_item(combo, index):
    if combo is None or index < 0 or index >= combo.count():
        return None
    item_text = combo.itemText(index).strip()
    data = combo.itemData(index)
    if isinstance(data, dict):
        return sanitize_template_ref(data)
    display_ref = template_ref_from_choice_display_text(item_text)
    data_name = template_name_from_combo_data(data)
    if data_name:
        if display_ref and display_ref["name"] == data_name:
            return template_ref(data_name, display_ref["type"])
        return template_ref(data_name, TEMPLATE_REF_TYPE_TEMPLATE)
    return display_ref


def template_ref_from_combo_choice(combo):
    if combo is None:
        return None
    text = combo.lineEdit().text().strip() if combo.isEditable() and combo.lineEdit() else combo.currentText().strip()
    if not text:
        return None

    current_index = combo.currentIndex()
    if current_index >= 0 and text == combo.itemText(current_index).strip():
        current_ref = template_ref_from_combo_item(combo, current_index)
        if current_ref:
            return current_ref

    exact_text_refs = []
    plain_name_refs = []
    for index in range(combo.count()):
        item_text = combo.itemText(index).strip()
        ref = template_ref_from_combo_item(combo, index)
        if not ref:
            continue
        if text == item_text:
            exact_text_refs.append(ref)
        elif text == ref["name"] and not is_rule_template_choice_display_text(item_text):
            plain_name_refs.append(ref)

    if len(exact_text_refs) == 1:
        return exact_text_refs[0]
    if exact_text_refs:
        normal_ref = next((ref for ref in exact_text_refs if ref["type"] == TEMPLATE_REF_TYPE_TEMPLATE), None)
        return normal_ref or exact_text_refs[0]

    display_ref = template_ref_from_choice_display_text(text)
    if display_ref and display_ref["type"] == TEMPLATE_REF_TYPE_RULE:
        return display_ref

    if plain_name_refs:
        normal_ref = next((ref for ref in plain_name_refs if ref["type"] == TEMPLATE_REF_TYPE_TEMPLATE), None)
        return normal_ref or plain_name_refs[0]

    return template_ref(template_name_from_choice_display_text(text), TEMPLATE_REF_TYPE_TEMPLATE)


def template_value_from_combo_choice(combo):
    return template_param_value(template_ref_from_combo_choice(combo))


def find_template_combo_index(combo, value):
    ref = sanitize_template_ref(value)
    if not ref:
        return -1
    target_key = template_ref_key(ref)
    for index in range(combo.count()):
        item_ref = template_ref_from_combo_item(combo, index)
        if item_ref and template_ref_key(item_ref) == target_key:
            return index
    return -1


def set_template_combo_current_value(combo, value):
    if value in (None, ""):
        combo.setCurrentIndex(-1)
        combo.setCurrentText("")
        return
    index = find_template_combo_index(combo, value)
    if index >= 0:
        combo.setCurrentIndex(index)
        return
    name = str(value.get("name") or "").strip() if isinstance(value, dict) else str(value or "").strip()
    combo.addItem(name, name)
    combo.setCurrentIndex(combo.count() - 1)
