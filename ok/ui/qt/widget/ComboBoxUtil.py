from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QCompleter


def make_combo_searchable(combo: QComboBox):
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.NoInsert)
    combo.completer().setCompletionMode(QCompleter.PopupCompletion)
    combo.completer().setFilterMode(Qt.MatchContains)
    combo.completer().setCaseSensitivity(Qt.CaseInsensitive)


def combo_current_data(combo: QComboBox, default=""):
    text = combo.currentText()
    for index in range(combo.count()):
        if combo.itemData(index) == text:
            return combo.itemData(index)
        if combo.itemText(index) == text:
            data = combo.itemData(index)
            return data if data not in (None, "") else text
    data = combo.currentData()
    if data not in (None, ""):
        return data
    return default or text
