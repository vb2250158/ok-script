import copy
import json
import os
import shutil
import time

import cv2
from ok.util.image_io import cv_imread, cv_imwrite
from PySide6.QtCore import QObject, QPoint, QRunnable, QRect, QSize, QThreadPool, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QBrush, QDoubleValidator, QFont, QIcon, QImage, QIntValidator, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QLineEdit as QtLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)
from qfluentwidgets import (
    BodyLabel,
    CheckBox,
    FluentIcon,
    LineEdit,
    MessageBox,
    MessageBoxBase,
    PrimaryPushButton,
    PushButton,
    Action,
    RoundMenu,
    SearchLineEdit,
    SubtitleLabel,
)

from ok import og
from ok.feature.TemplateRef import (
    TEMPLATE_REF_TYPE_RULE,
    TEMPLATE_REF_TYPE_TEMPLATE,
    template_ref,
)
from ok.feature.VisualRuleTemplate import MIN_VISUAL_RULE_SAMPLES
from ok.ui.qt.util.Alert import alert_error, alert_info
from ok.ui.qt.widget.TemplateChoice import (
    add_template_choice_to_combo,
    template_choice_display_name,
    template_name_from_choice_display_text,
    template_name_from_combo_data,
    template_ref_from_combo_choice,
)
from ok.ui.qt.widget.ComboBoxUtil import make_combo_searchable
from ok.ui.qt.video_template.dialogs import (
    AgentAssistPromptDialog,
    ExportTemplateConflictDialog,
)
from ok.ui.qt.video_template.ocr_processor import OcrCancelled, VideoTemplateOcrProcessor
from ok.ui.qt.video_template.preview_match import PreviewMatchWorker
from ok.ui.qt.video_template.recorder import VideoTemplateRecorder
from ok.ui.qt.video_template.store import (
    ANNOTATION_STATUS_CONFIRMED,
    ANNOTATION_STATUS_DRAFT,
    FRAME_STATUS_DISCARDED,
    FRAME_STATUS_CONFIRMED,
    FRAME_STATUS_SELECTED,
    VideoTemplatePackage,
    VideoTemplateStore,
)
from ok.ui.qt.video_template import template_sources as template_source_utils
from ok.ui.qt.video_template.template_sources import visual_rule_template_names_from_samples
from ok.util.logger import Logger


logger = Logger.get_logger(__name__)


FRAME_ROLE = Qt.UserRole + 1
PACKAGE_ROLE = Qt.UserRole + 2
DEFAULT_TEMPLATE_GROUP_NAME = "默认分组"
TEMPLATE_GROUP_ITEM_KIND_ROLE = Qt.UserRole + 3
TEMPLATE_GROUP_NAME_ROLE = Qt.UserRole + 4
TEMPLATE_GROUP_RULE_ROLE = Qt.UserRole + 5
TEMPLATE_REF_TYPE_ROLE = Qt.UserRole + 6
TEMPLATE_GROUP_KIND_GROUP = "group"
TEMPLATE_GROUP_KIND_TEMPLATE = "template"
PREVIEW_MATCH_INTERVAL_SECONDS = 0.0
PREVIEW_MATCH_UI_UPDATE_INTERVAL_SECONDS = 0.15
PREVIEW_MATCH_ICON_BATCH_SIZE = 12


def frame_to_pixmap(frame):
    if frame is None:
        return QPixmap()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def format_seconds(seconds):
    seconds = int(seconds or 0)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def draw_outlined_text(painter, x, y, text):
    painter.save()
    font = painter.font()
    if font.pointSize() > 0:
        font.setPointSize(font.pointSize() * 2)
    elif font.pixelSize() > 0:
        font.setPixelSize(font.pixelSize() * 2)
    painter.setFont(font)
    outline = QColor(0, 0, 0, 220)
    fill = QColor(0, 220, 255)
    painter.setPen(QPen(outline, 4))
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)]:
        painter.drawText(x + dx, y + dy, text)
    painter.setPen(QPen(fill, 1))
    painter.drawText(x, y, text)
    painter.restore()


def draw_outlined_rect(painter, x, y, width, height, color, inner_width=2):
    painter.save()
    painter.setPen(QPen(QColor(0, 0, 0, 230), inner_width + 4))
    painter.drawRect(int(x), int(y), int(width), int(height))
    painter.setPen(QPen(color, inner_width))
    painter.drawRect(int(x), int(y), int(width), int(height))
    painter.restore()


def ensure_feature_set_uses_template_coco(feature_set):
    return template_source_utils.ensure_feature_set_uses_template_coco(feature_set)


def template_name_groups(names):
    return template_source_utils.template_name_groups(names)


def runtime_template_names():
    return template_source_utils.runtime_template_names()


def coco_template_names():
    return template_source_utils.coco_template_names()


def visual_rule_template_names(choices=None):
    return template_source_utils.visual_rule_template_names(choices)


def template_choice_names():
    return template_source_utils.template_choice_names()


def template_choice_refs():
    return template_source_utils.template_choice_refs()


def sanitize_template_ref(value, rule_names=None):
    return template_source_utils.sanitize_template_ref(
        value,
        rule_names if rule_names is not None else visual_rule_template_names(),
    )


def sanitize_template_refs(values, rule_names=None):
    return template_source_utils.sanitize_template_refs(
        values,
        rule_names if rule_names is not None else visual_rule_template_names(),
    )


def template_ref_key(ref):
    return template_source_utils.template_ref_key(sanitize_template_ref(ref))


class TimelineWidget(QWidget):
    seek_requested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.duration = 0
        self.current_time = 0
        self.events = []
        self.setMinimumHeight(54)
        self.setMouseTracking(True)

    def set_data(self, duration, events, current_time=0):
        self.duration = max(0, float(duration or 0))
        self.events = list(events or [])
        self.current_time = max(0, float(current_time or 0))
        self.update()

    def set_current_time(self, current_time):
        self.current_time = max(0, float(current_time or 0))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        center_y = self.height() // 2
        painter.setPen(QPen(QColor(150, 150, 150), 2))
        painter.drawLine(8, center_y, width - 8, center_y)

        for item in self.events:
            x = self._x_for_time(item_time(item), width)
            color = QColor(0, 120, 212) if item.get("type") == "click" else QColor(230, 126, 34)
            painter.setBrush(color)
            painter.setPen(QPen(color.darker(120), 1))
            if item.get("type") == "drag":
                painter.drawRect(x - 4, center_y - 10, 8, 20)
                end_x = self._x_for_time(float(item.get("end_time", item.get("time", 0)) or 0), width)
                painter.setPen(QPen(color, 2, Qt.DashLine))
                painter.drawLine(x, center_y - 15, end_x, center_y - 15)
            else:
                painter.drawEllipse(x - 5, center_y - 5, 10, 10)

        current_x = self._x_for_time(self.current_time, width)
        painter.setPen(QPen(QColor(220, 20, 60), 2))
        painter.drawLine(current_x, 8, current_x, self.height() - 8)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.duration > 0:
            self.seek_requested.emit(self._time_for_x(event.position().x()))

    def mouseMoveEvent(self, event):
        nearest = self._nearest_event(event.position().x())
        if nearest:
            self.setToolTip(event_tooltip(nearest))
        else:
            self.setToolTip("")

    def _x_for_time(self, seconds, width):
        if self.duration <= 0:
            return 8
        return int(8 + max(0, min(1, seconds / self.duration)) * max(1, width - 16))

    def _time_for_x(self, x):
        if self.duration <= 0:
            return 0
        ratio = max(0, min(1, (x - 8) / max(1, self.width() - 16)))
        return ratio * self.duration

    def _nearest_event(self, x):
        if not self.events or self.duration <= 0:
            return None
        best = None
        best_distance = 9999
        for event in self.events:
            event_x = self._x_for_time(item_time(event), self.width())
            distance = abs(event_x - x)
            if distance < best_distance:
                best = event
                best_distance = distance
        return best if best_distance <= 12 else None


def item_time(event):
    return float(event.get("time", 0) or 0)


def event_tooltip(event):
    if event.get("type") == "drag":
        return (
            f"drag {format_seconds(event.get('time'))}-{format_seconds(event.get('end_time'))}\n"
            f"from ({event.get('from_x', 0):.3f}, {event.get('from_y', 0):.3f}) "
            f"to ({event.get('to_x', 0):.3f}, {event.get('to_y', 0):.3f})"
        )
    return f"click {format_seconds(event.get('time'))}\n({event.get('x', 0):.3f}, {event.get('y', 0):.3f})"


class OcrWorkerSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(list)
    failed = Signal(str)


class OcrWorker(QRunnable):
    def __init__(self, package, mode, seconds=0):
        super().__init__()
        self.package = package
        self.mode = mode
        self.seconds = seconds
        self.cancel_requested = False
        self.signals = OcrWorkerSignals()

    def run(self):
        processor = VideoTemplateOcrProcessor(self.package)

        def progress(done, total, message):
            if self.cancel_requested:
                raise OcrCancelled()
            self.signals.progress.emit(done, total, message)

        try:
            if self.mode == "all":
                result = processor.process_all_operation_frames(progress=progress)
            else:
                result = processor.process_current_frame(self.seconds, progress=progress)
            self.signals.finished.emit(result)
        except OcrCancelled:
            self.signals.failed.emit("图像识别加工已取消")
        except Exception as e:
            logger.error(f"video template OCR failed: {e}")
            self.signals.failed.emit(str(e))


class ExportTemplateWorkerSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(dict)
    failed = Signal(str)


class ExportTemplateWorker(QRunnable):
    def __init__(self, source_path, boxes, target_name, overwrite_labels=None):
        super().__init__()
        self.source_path = source_path
        self.boxes = copy.deepcopy(boxes or [])
        self.target_name = target_name
        self.overwrite_labels = {
            str(label or "").strip()
            for label in overwrite_labels or []
            if str(label or "").strip()
        }
        self.signals = ExportTemplateWorkerSignals()

    def run(self):
        try:
            self.signals.progress.emit(0, 4, "导出模板列表：准备中...")
            self.signals.progress.emit(1, 4, "导出模板列表：写入模板图片与 COCO...")
            exported_path, labels = EditVideoTemplatePage.write_annotations_to_template_coco(
                self.source_path,
                self.boxes,
                self.target_name,
                overwrite_labels=self.overwrite_labels,
            )
            self.signals.progress.emit(3, 4, "导出模板列表：刷新模板匹配缓存...")
            EditVideoTemplatePage.reload_feature_set_after_template_export()
            self.signals.progress.emit(4, 4, "导出模板列表：完成")
            self.signals.finished.emit({
                "exported_path": exported_path,
                "labels": labels,
                "count": len(self.boxes),
            })
        except Exception as e:
            logger.error(f"export video template annotations failed: {e}")
            self.signals.failed.emit(str(e))


VIDEO_HANDLE_MARGIN = 8
VIDEO_HANDLE_NONE = 0
VIDEO_HANDLE_TOP = 1
VIDEO_HANDLE_BOTTOM = 2
VIDEO_HANDLE_LEFT = 3
VIDEO_HANDLE_RIGHT = 4
VIDEO_HANDLE_TL = 5
VIDEO_HANDLE_TR = 6
VIDEO_HANDLE_BL = 7
VIDEO_HANDLE_BR = 8


def video_detect_handle(pos, rect, margin=VIDEO_HANDLE_MARGIN):
    x, y = pos.x(), pos.y()
    rx, ry, rw, rh = rect.x(), rect.y(), rect.width(), rect.height()
    right = rx + rw
    bottom = ry + rh
    near_left = abs(x - rx) <= margin and ry - margin <= y <= bottom + margin
    near_right = abs(x - right) <= margin and ry - margin <= y <= bottom + margin
    near_top = abs(y - ry) <= margin and rx - margin <= x <= right + margin
    near_bottom = abs(y - bottom) <= margin and rx - margin <= x <= right + margin
    if near_top and near_left:
        return VIDEO_HANDLE_TL
    if near_top and near_right:
        return VIDEO_HANDLE_TR
    if near_bottom and near_left:
        return VIDEO_HANDLE_BL
    if near_bottom and near_right:
        return VIDEO_HANDLE_BR
    if near_top:
        return VIDEO_HANDLE_TOP
    if near_bottom:
        return VIDEO_HANDLE_BOTTOM
    if near_left:
        return VIDEO_HANDLE_LEFT
    if near_right:
        return VIDEO_HANDLE_RIGHT
    return VIDEO_HANDLE_NONE


def video_cursor_for_handle(handle):
    if handle in (VIDEO_HANDLE_TL, VIDEO_HANDLE_BR):
        return Qt.SizeFDiagCursor
    if handle in (VIDEO_HANDLE_TR, VIDEO_HANDLE_BL):
        return Qt.SizeBDiagCursor
    if handle in (VIDEO_HANDLE_TOP, VIDEO_HANDLE_BOTTOM):
        return Qt.SizeVerCursor
    if handle in (VIDEO_HANDLE_LEFT, VIDEO_HANDLE_RIGHT):
        return Qt.SizeHorCursor
    return Qt.ArrowCursor


class AnnotationVideoLabel(QLabel):
    annotation_created = Signal(dict)
    annotation_boxes_changed = Signal(list, str)
    annotation_selection_changed = Signal(int)
    annotation_edit_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source_pixmap = QPixmap()
        self.ocr_boxes = []
        self.candidate_boxes = []
        self.annotation_boxes = []
        self.template_match_boxes = []
        self.template_validation_mode = False
        self.draw_mode = False
        self.selected_index = -1
        self.hover_index = -1
        self.hover_handle = VIDEO_HANDLE_NONE
        self.drag_start = None
        self.drag_current = None
        self.dragging_index = -1
        self.drag_origin_point = None
        self.drag_origin_bbox = None
        self.resizing_index = -1
        self.resize_handle = VIDEO_HANDLE_NONE
        self.resize_origin_point = None
        self.resize_origin_bbox = None
        self.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_frame_pixmap(self, pixmap, ocr_boxes=None, annotation_boxes=None, candidate_boxes=None):
        self.source_pixmap = QPixmap(pixmap)
        self.ocr_boxes = list(ocr_boxes or [])
        self.candidate_boxes = list(candidate_boxes or [])
        self.annotation_boxes = list(annotation_boxes or [])
        self.selected_index = -1
        self.hover_index = -1
        self.hover_handle = VIDEO_HANDLE_NONE
        self._clear_drag_state()
        self.refresh_display()

    def set_annotation_boxes(self, annotation_boxes):
        self.annotation_boxes = list(annotation_boxes or [])
        if self.selected_index >= len(self.annotation_boxes):
            self.selected_index = -1
            self.annotation_selection_changed.emit(-1)
        self.refresh_display()

    def set_draw_mode(self, enabled):
        self.draw_mode = bool(enabled)
        self._clear_drag_state()
        self.setCursor(Qt.CrossCursor if self.draw_mode else Qt.ArrowCursor)
        self.refresh_display()

    def set_template_validation_boxes(self, boxes, enabled):
        self.template_match_boxes = list(boxes or [])
        self.template_validation_mode = bool(enabled)
        self.refresh_display()

    def refresh_display(self):
        if self.source_pixmap.isNull():
            self.clear()
            return
        pixmap = QPixmap(self.source_pixmap)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.template_validation_mode:
            for box in self.template_match_boxes:
                self._draw_match_box(painter, box)
            painter.end()
            self.setPixmap(pixmap.scaled(self.contentsRect().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            return
        if not self.template_validation_mode:
            painter.setPen(QPen(QColor(220, 45, 45), 2))
            for box in self.ocr_boxes:
                self._draw_bbox(painter, box.get("bbox") or [], QColor(255, 80, 80), 2)
            for box in self.candidate_boxes:
                self._draw_bbox(painter, box.get("bbox") or [], QColor(255, 176, 40), 2)
                self._draw_label(painter, box.get("label"), box.get("bbox") or [])
        for index, box in enumerate(self.annotation_boxes):
            if index == self.selected_index:
                color = QColor(0, 170, 255)
                width = 4
            elif index == self.hover_index:
                color = QColor(255, 176, 40)
                width = 3
            else:
                color = QColor(50, 220, 120)
                width = 3
            self._draw_bbox(painter, box.get("bbox") or [], color, width)
            self._draw_label(painter, box.get("label"), box.get("bbox") or [])
            if index == self.hover_index and self.hover_handle != VIDEO_HANDLE_NONE:
                self._draw_hover_handle(painter, box.get("bbox") or [], self.hover_handle)
        if self.drag_start is not None and self.drag_current is not None:
            painter.setPen(QPen(QColor(30, 160, 90), 2, Qt.DashLine))
            painter.drawRect(QRect(self.drag_start, self.drag_current).normalized())
        painter.end()
        self.setPixmap(pixmap.scaled(self.contentsRect().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_display()

    def mousePressEvent(self, event):
        self.setFocus()
        if self.template_validation_mode:
            super().mousePressEvent(event)
            return
        point = self.widget_to_image_point(event.position().toPoint())
        if event.button() == Qt.RightButton and point is not None:
            index = self.annotation_index_at(point)
            if index >= 0:
                self.select_annotation(index)
                menu = RoundMenu(parent=self)
                menu.addAction(Action(self.tr("复制标注"), triggered=self.copy_selected_annotation))
                menu.addAction(Action(self.tr("修改标注名"), triggered=lambda: self.annotation_edit_requested.emit(index)))
                menu.addAction(Action(self.tr("删除标注"), triggered=self.delete_selected_annotation))
                menu.exec_(self.mapToGlobal(event.position().toPoint()))
                event.accept()
                return
        if event.button() == Qt.LeftButton and not self.source_pixmap.isNull():
            point = self.widget_to_image_point(event.position().toPoint())
            if point is not None:
                if self.draw_mode:
                    self.drag_start = point
                    self.drag_current = point
                else:
                    index, handle = self.annotation_handle_at(point)
                    if index >= 0 and handle != VIDEO_HANDLE_NONE:
                        self.select_annotation(index)
                        self.resizing_index = index
                        self.resize_handle = handle
                        self.resize_origin_point = point
                        self.resize_origin_bbox = list(self.annotation_boxes[index].get("bbox") or [0, 0, 0, 0])
                    else:
                        index = self.annotation_index_at(point)
                        self.select_annotation(index)
                        if index >= 0:
                            self.dragging_index = index
                            self.drag_origin_point = point
                            self.drag_origin_bbox = list(self.annotation_boxes[index].get("bbox") or [0, 0, 0, 0])
                self.refresh_display()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        point = self.widget_to_image_point(event.position().toPoint())
        if point is not None:
            if self.drag_start is not None:
                self.drag_current = point
                self.refresh_display()
            elif self.dragging_index >= 0:
                self.move_dragged_annotation(point)
            elif self.resizing_index >= 0:
                self.resize_dragged_annotation(point)
            elif not self.draw_mode and not self.template_validation_mode:
                index, handle = self.annotation_handle_at(point)
                if index >= 0 and handle != VIDEO_HANDLE_NONE:
                    self.hover_index = index
                    self.hover_handle = handle
                    self.setCursor(video_cursor_for_handle(handle))
                else:
                    index = self.annotation_index_at(point)
                    self.hover_index = index
                    self.hover_handle = VIDEO_HANDLE_NONE
                    self.setCursor(Qt.SizeAllCursor if index >= 0 else Qt.ArrowCursor)
                self.refresh_display()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.template_validation_mode:
            super().mouseReleaseEvent(event)
            return
        changed_label = None
        if event.button() == Qt.LeftButton and self.drag_start is not None:
            end = self.widget_to_image_point(event.position().toPoint())
            start = self.drag_start
            self.drag_start = None
            self.drag_current = None
            if end is not None:
                rect = QRect(start, end).normalized()
                if rect.width() >= 4 and rect.height() >= 4:
                    self.annotation_created.emit({
                        "bbox": [rect.x(), rect.y(), rect.width(), rect.height()],
                    })
            self.refresh_display()
        elif event.button() == Qt.LeftButton and self.dragging_index >= 0:
            current = self.annotation_boxes[self.dragging_index].get("bbox") or []
            if list(current[:4]) != list((self.drag_origin_bbox or [])[:4]):
                changed_label = "move_annotation_box"
        elif event.button() == Qt.LeftButton and self.resizing_index >= 0:
            current = self.annotation_boxes[self.resizing_index].get("bbox") or []
            if list(current[:4]) != list((self.resize_origin_bbox or [])[:4]):
                changed_label = "resize_annotation_box"
        if changed_label:
            self.annotation_boxes_changed.emit(copy.deepcopy(self.annotation_boxes), changed_label)
        if event.button() == Qt.LeftButton:
            self.dragging_index = -1
            self.drag_origin_point = None
            self.drag_origin_bbox = None
            self.resizing_index = -1
            self.resize_handle = VIDEO_HANDLE_NONE
            self.resize_origin_point = None
            self.resize_origin_bbox = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.template_validation_mode:
            super().mouseDoubleClickEvent(event)
            return
        point = self.widget_to_image_point(event.position().toPoint())
        if point is not None:
            index = self.annotation_index_at(point)
            if index >= 0:
                self.select_annotation(index)
                self.annotation_edit_requested.emit(index)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def widget_to_image_point(self, point):
        if self.source_pixmap.isNull():
            return None
        image_size = self.source_pixmap.size()
        display_rect = self.image_display_rect()
        if display_rect.isNull():
            return None
        if not (display_rect.x() <= point.x() <= display_rect.x() + display_rect.width()):
            return None
        if not (display_rect.y() <= point.y() <= display_rect.y() + display_rect.height()):
            return None
        x = round((point.x() - display_rect.x()) * image_size.width() / max(1, display_rect.width()))
        y = round((point.y() - display_rect.y()) * image_size.height() / max(1, display_rect.height()))
        return QPoint(
            max(0, min(image_size.width() - 1, x)),
            max(0, min(image_size.height() - 1, y)),
        )

    def image_display_rect(self):
        if self.source_pixmap.isNull():
            return QRect()
        contents = self.contentsRect()
        scaled = self.source_pixmap.size().scaled(contents.size(), Qt.KeepAspectRatio)
        alignment = self.alignment()
        if alignment & Qt.AlignRight:
            x = contents.right() - scaled.width() + 1
        elif alignment & Qt.AlignHCenter:
            x = contents.x() + (contents.width() - scaled.width()) / 2
        else:
            x = contents.x()
        if alignment & Qt.AlignBottom:
            y = contents.bottom() - scaled.height() + 1
        elif alignment & Qt.AlignVCenter:
            y = contents.y() + (contents.height() - scaled.height()) / 2
        else:
            y = contents.y()
        return QRect(int(round(x)), int(round(y)), scaled.width(), scaled.height())

    def annotation_rect(self, index):
        if not (0 <= index < len(self.annotation_boxes)):
            return QRect()
        bbox = self.annotation_boxes[index].get("bbox") or []
        if len(bbox) != 4:
            return QRect()
        return QRect(int(round(bbox[0])), int(round(bbox[1])), int(round(bbox[2])), int(round(bbox[3])))

    def annotation_index_at(self, point):
        for index in range(len(self.annotation_boxes) - 1, -1, -1):
            if self.annotation_rect(index).contains(point):
                return index
        return -1

    def annotation_handle_at(self, point):
        for index in range(len(self.annotation_boxes) - 1, -1, -1):
            rect = self.annotation_rect(index)
            handle = video_detect_handle(point, rect)
            if handle != VIDEO_HANDLE_NONE:
                return index, handle
        return -1, VIDEO_HANDLE_NONE

    def select_annotation(self, index):
        self.selected_index = index if 0 <= index < len(self.annotation_boxes) else -1
        self.annotation_selection_changed.emit(self.selected_index)
        self.refresh_display()

    def move_dragged_annotation(self, point):
        if self.drag_origin_point is None or not self.drag_origin_bbox:
            return
        dx = point.x() - self.drag_origin_point.x()
        dy = point.y() - self.drag_origin_point.y()
        x, y, width, height = self.drag_origin_bbox[:4]
        image_size = self.source_pixmap.size()
        new_x = max(0, min(int(round(x + dx)), image_size.width() - int(width)))
        new_y = max(0, min(int(round(y + dy)), image_size.height() - int(height)))
        self.annotation_boxes[self.dragging_index]["bbox"] = [new_x, new_y, int(width), int(height)]
        self.refresh_display()

    def resize_dragged_annotation(self, point):
        if self.resize_origin_point is None or not self.resize_origin_bbox:
            return
        dx = point.x() - self.resize_origin_point.x()
        dy = point.y() - self.resize_origin_point.y()
        x, y, width, height = [float(value) for value in self.resize_origin_bbox[:4]]
        nx, ny, nw, nh = x, y, width, height
        handle = self.resize_handle
        if handle in (VIDEO_HANDLE_LEFT, VIDEO_HANDLE_TL, VIDEO_HANDLE_BL):
            nx = x + dx
            nw = width - dx
        if handle in (VIDEO_HANDLE_RIGHT, VIDEO_HANDLE_TR, VIDEO_HANDLE_BR):
            nw = width + dx
        if handle in (VIDEO_HANDLE_TOP, VIDEO_HANDLE_TL, VIDEO_HANDLE_TR):
            ny = y + dy
            nh = height - dy
        if handle in (VIDEO_HANDLE_BOTTOM, VIDEO_HANDLE_BL, VIDEO_HANDLE_BR):
            nh = height + dy
        min_size = 5
        if nw < min_size:
            if handle in (VIDEO_HANDLE_LEFT, VIDEO_HANDLE_TL, VIDEO_HANDLE_BL):
                nx = x + width - min_size
            nw = min_size
        if nh < min_size:
            if handle in (VIDEO_HANDLE_TOP, VIDEO_HANDLE_TL, VIDEO_HANDLE_TR):
                ny = y + height - min_size
            nh = min_size
        image_size = self.source_pixmap.size()
        nx = max(0, min(nx, image_size.width() - min_size))
        ny = max(0, min(ny, image_size.height() - min_size))
        nw = min(nw, image_size.width() - nx)
        nh = min(nh, image_size.height() - ny)
        self.annotation_boxes[self.resizing_index]["bbox"] = [
            int(round(nx)), int(round(ny)), int(round(nw)), int(round(nh))
        ]
        self.refresh_display()

    def copy_selected_annotation(self):
        if not (0 <= self.selected_index < len(self.annotation_boxes)):
            return
        source = copy.deepcopy(self.annotation_boxes[self.selected_index])
        bbox = list(source.get("bbox") or [])
        if len(bbox) == 4:
            image_size = self.source_pixmap.size()
            bbox[0] = max(0, min(int(round(bbox[0])) + 20, image_size.width() - int(round(bbox[2]))))
            bbox[1] = max(0, min(int(round(bbox[1])) + 20, image_size.height() - int(round(bbox[3]))))
            source["bbox"] = bbox
        self.annotation_boxes.append(source)
        self.selected_index = len(self.annotation_boxes) - 1
        self.annotation_selection_changed.emit(self.selected_index)
        self.annotation_boxes_changed.emit(copy.deepcopy(self.annotation_boxes), "copy_annotation_box")
        self.refresh_display()

    def delete_selected_annotation(self):
        if not (0 <= self.selected_index < len(self.annotation_boxes)):
            return
        self.annotation_boxes.pop(self.selected_index)
        self.selected_index = -1
        self.annotation_selection_changed.emit(-1)
        self.annotation_boxes_changed.emit(copy.deepcopy(self.annotation_boxes), "delete_annotation_box")
        self.refresh_display()

    def copy_selected_annotation_to_clipboard(self):
        if not (0 <= self.selected_index < len(self.annotation_boxes)):
            return
        QApplication.clipboard().setText(
            json.dumps(self.annotation_boxes[self.selected_index], ensure_ascii=False)
        )

    def _clear_drag_state(self):
        self.drag_start = None
        self.drag_current = None
        self.dragging_index = -1
        self.drag_origin_point = None
        self.drag_origin_bbox = None
        self.resizing_index = -1
        self.resize_handle = VIDEO_HANDLE_NONE
        self.resize_origin_point = None
        self.resize_origin_bbox = None

    @staticmethod
    def _draw_bbox(painter, bbox, color, width):
        if len(bbox) == 4:
            draw_outlined_rect(painter, bbox[0], bbox[1], bbox[2], bbox[3], color, width)

    @staticmethod
    def _draw_label(painter, text, bbox):
        if not text or len(bbox) != 4:
            return
        draw_outlined_text(painter, int(bbox[0]), max(14, int(bbox[1]) - 5), str(text))

    @staticmethod
    def _draw_match_box(painter, box):
        draw_outlined_rect(painter, box.x, box.y, box.width, box.height, QColor(0, 170, 255), 2)
        name = str(getattr(box, "name", "") or "")
        confidence = max(0, min(100, round(float(getattr(box, "confidence", 0) or 0) * 100)))
        label = f"{name} {confidence}%" if name else f"{confidence}%"
        if label:
            draw_outlined_text(painter, int(box.x), max(14, int(box.y) - 5), label)

    @staticmethod
    def _draw_hover_handle(painter, bbox, handle):
        if len(bbox) != 4:
            return
        x, y, width, height = [int(round(value)) for value in bbox[:4]]
        right = x + width
        bottom = y + height
        painter.save()
        painter.setPen(QPen(QColor(0, 255, 120), 3))
        if handle in (VIDEO_HANDLE_TOP, VIDEO_HANDLE_TL, VIDEO_HANDLE_TR):
            painter.drawLine(x, y, right, y)
        if handle in (VIDEO_HANDLE_BOTTOM, VIDEO_HANDLE_BL, VIDEO_HANDLE_BR):
            painter.drawLine(x, bottom, right, bottom)
        if handle in (VIDEO_HANDLE_LEFT, VIDEO_HANDLE_TL, VIDEO_HANDLE_BL):
            painter.drawLine(x, y, x, bottom)
        if handle in (VIDEO_HANDLE_RIGHT, VIDEO_HANDLE_TR, VIDEO_HANDLE_BR):
            painter.drawLine(right, y, right, bottom)
        painter.restore()


class RecordVideoTemplatePage(QWidget):
    package_created = Signal()

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self.recorder = VideoTemplateRecorder(store=store)
        self.preview_match_boxes = []
        self.preview_match_worker = None
        self.preview_match_seq = 0
        self.last_preview_match_at = 0
        self.preview_match_progress = None
        self.template_group_config = {"groups": [], "default_order": [], "specified_templates": []}
        self.collapsed_template_groups = set()
        self.preview_match_current_group = ""
        self.preview_match_current_template = ""
        self.preview_match_hit_counts = {}
        self.preview_match_scanned_templates = set()
        self.preview_match_render_groups = []
        self.preview_match_render_rule_names = set()
        self.preview_match_group_items = {}
        self.template_icon_cache = {}
        self.preview_match_icon_queue = []
        self.last_preview_progress_ui_at = 0.0
        self.last_preview_detail_ui_at = 0.0
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_match_icon_timer = QTimer(self)
        self.preview_match_icon_timer.setInterval(0)
        self.preview_match_icon_timer.timeout.connect(self.load_preview_match_icons_batch)
        self.record_timer = QTimer(self)
        self.record_timer.timeout.connect(self.update_record_status)
        self.init_ui()
        self.update_preview_timer()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        self.start_btn = PrimaryPushButton(FluentIcon.PLAY, self.tr("开始录制"))
        self.start_btn.clicked.connect(self.start_recording)
        toolbar.addWidget(self.start_btn)
        self.stop_btn = PushButton(FluentIcon.PAUSE, self.tr("停止录制"))
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)
        toolbar.addWidget(self.stop_btn)
        self.duration_label = BodyLabel(self.tr("时长 00:00"))
        toolbar.addWidget(self.duration_label)
        self.event_count_label = BodyLabel(self.tr("事件 0"))
        toolbar.addWidget(self.event_count_label)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        settings = QHBoxLayout()
        self.show_all_templates_check = CheckBox(self.tr("显示全部已标注模板"))
        self.confidence_edit = LineEdit()
        confidence_validator = QDoubleValidator(0.1, 1.0, 2, self.confidence_edit)
        confidence_validator.setNotation(QDoubleValidator.StandardNotation)
        self.confidence_edit.setValidator(confidence_validator)
        self.confidence_edit.setText("0.72")
        self.confidence_edit.setFixedWidth(76)
        self.preview_fps_edit = LineEdit()
        self.preview_fps_edit.setValidator(QIntValidator(1, 30, self.preview_fps_edit))
        self.preview_fps_edit.setText("8")
        self.preview_fps_edit.setFixedWidth(72)
        self.preview_fps_edit.textChanged.connect(self.update_preview_timer)
        self.preview_resolution_edit = LineEdit()
        self.preview_resolution_edit.setValidator(QIntValidator(25, 100, self.preview_resolution_edit))
        self.preview_resolution_edit.setText("50")
        self.preview_resolution_edit.setFixedWidth(72)
        settings.addWidget(self.show_all_templates_check)
        settings.addWidget(BodyLabel(self.tr("置信度阈值")))
        settings.addWidget(self.confidence_edit)
        settings.addWidget(BodyLabel(self.tr("预览帧率")))
        settings.addWidget(self.preview_fps_edit)
        settings.addWidget(BodyLabel("FPS"))
        settings.addWidget(BodyLabel(self.tr("识别分辨率")))
        settings.addWidget(self.preview_resolution_edit)
        settings.addWidget(BodyLabel("%"))
        settings.addStretch(1)
        layout.addLayout(settings)

        preview_splitter = QSplitter(Qt.Horizontal)
        self.group_panel = self.build_template_group_panel()
        preview_splitter.addWidget(self.group_panel)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(420)
        self.preview_label.setFrameShape(QFrame.StyledPanel)
        self.preview_label.setStyleSheet("QLabel { background: rgba(0, 0, 0, 0.05); border-radius: 6px; }")
        preview_splitter.addWidget(self.preview_label)
        preview_splitter.setStretchFactor(0, 0)
        preview_splitter.setStretchFactor(1, 1)
        preview_splitter.setSizes([260, 900])
        layout.addWidget(preview_splitter, 1)

        self.match_status_label = BodyLabel(self.tr("模板匹配：未启用"))
        self.match_status_label.setWordWrap(False)
        layout.addWidget(self.match_status_label)
        self.reload_template_groups()

    def build_template_group_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.addWidget(BodyLabel(self.tr("模板分组")))
        self.template_group_tabs = QTabWidget()
        edit_tab = QWidget()
        edit_layout = QVBoxLayout(edit_tab)
        edit_layout.setContentsMargins(0, 0, 0, 0)
        edit_layout.setSpacing(8)
        self.template_group_search_edit = SearchLineEdit()
        self.template_group_search_edit.setPlaceholderText(self.tr("搜索模板"))
        self.template_group_search_edit.textChanged.connect(lambda _text: self.reload_template_groups())
        edit_layout.addWidget(self.template_group_search_edit)
        self.template_group_list = QListWidget()
        self.template_group_list.setIconSize(QSize(44, 32))
        self.template_group_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.template_group_list.itemDoubleClicked.connect(self.toggle_template_group_item)
        self.template_group_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.template_group_list.customContextMenuRequested.connect(self.show_template_group_context_menu)
        edit_layout.addWidget(self.template_group_list, 1)
        group_buttons = QHBoxLayout()
        create_btn = PushButton(FluentIcon.ADD, self.tr("新建分组"))
        create_btn.clicked.connect(self.create_template_group)
        delete_group_btn = PushButton(FluentIcon.DELETE, self.tr("删除分组"))
        delete_group_btn.clicked.connect(self.delete_selected_template_group)
        refresh_btn = PushButton(FluentIcon.SYNC, self.tr("刷新"))
        refresh_btn.clicked.connect(self.refresh_template_groups_from_source)
        group_buttons.addWidget(create_btn)
        group_buttons.addWidget(delete_group_btn)
        group_buttons.addWidget(refresh_btn)
        edit_layout.addLayout(group_buttons)
        self.template_group_tabs.addTab(edit_tab, self.tr("编辑分组优化"))

        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)
        self.preview_match_group_list = QListWidget()
        self.preview_match_group_list.setIconSize(QSize(44, 32))
        self.preview_match_group_list.setSelectionMode(QAbstractItemView.NoSelection)
        preview_layout.addWidget(self.preview_match_group_list, 1)
        self.preview_match_tab_index = self.template_group_tabs.addTab(preview_tab, self.tr("预览分组匹配"))

        specific_tab = QWidget()
        specific_layout = QVBoxLayout(specific_tab)
        specific_layout.setContentsMargins(0, 0, 0, 0)
        specific_layout.setSpacing(8)
        self.specific_preview_template_list = QListWidget()
        self.specific_preview_template_list.setIconSize(QSize(44, 32))
        self.specific_preview_template_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.specific_preview_template_list.itemDoubleClicked.connect(self.edit_specific_preview_template_item)
        specific_layout.addWidget(self.specific_preview_template_list, 1)
        specific_buttons = QHBoxLayout()
        add_specific_btn = PushButton(FluentIcon.ADD, self.tr("新增匹配模板"))
        add_specific_btn.clicked.connect(self.add_specific_preview_template)
        remove_specific_btn = PushButton(FluentIcon.REMOVE, self.tr("删除模板"))
        remove_specific_btn.clicked.connect(self.remove_selected_specific_preview_templates)
        refresh_specific_btn = PushButton(FluentIcon.SYNC, self.tr("刷新"))
        refresh_specific_btn.clicked.connect(self.refresh_template_groups_from_source)
        specific_buttons.addWidget(add_specific_btn)
        specific_buttons.addWidget(remove_specific_btn)
        specific_buttons.addWidget(refresh_specific_btn)
        specific_layout.addLayout(specific_buttons)
        self.specific_preview_tab_index = self.template_group_tabs.addTab(specific_tab, self.tr("预览指定模板匹配"))
        self.template_group_tabs.currentChanged.connect(self.on_template_group_tab_changed)
        layout.addWidget(self.template_group_tabs, 1)
        return panel

    def on_template_group_tab_changed(self, index):
        if index == getattr(self, "preview_match_tab_index", -1):
            self.reload_preview_match_groups()
        elif index == getattr(self, "specific_preview_tab_index", -1):
            self.reload_specific_preview_templates()

    def template_group_config_path(self):
        from ok.ui.qt.tasks.TemplateTab import ensure_template_folder
        return os.path.join(ensure_template_folder(), "template_groups.json")

    def load_template_group_config(self):
        path = self.template_group_config_path()
        try:
            with open(path, "r", encoding="utf-8-sig") as file:
                data = json.load(file)
            if isinstance(data, dict) and isinstance(data.get("groups"), list):
                return self.sanitize_template_group_config(data)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"failed to load template group config {path}: {e}")
        return {"groups": [], "default_order": [], "specified_templates": []}

    def save_template_group_config(self):
        path = self.template_group_config_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                json.dump({
                    "groups": self.configured_template_groups(),
                    "default_order": self.configured_default_template_order(),
                    "specified_templates": self.configured_specific_preview_template_refs(),
                }, file, ensure_ascii=False, indent=2)
        except Exception as e:
            alert_error(self.tr("保存模板分组失败: {e}").format(e=e))

    def configured_template_groups(self):
        return self.sanitize_template_groups(getattr(self, "template_group_config", {}).get("groups") or [])

    def configured_default_template_order(self):
        return self.sanitize_default_template_order(getattr(self, "template_group_config", {}).get("default_order") or [])

    def configured_specific_preview_templates(self):
        return [ref["name"] for ref in self.configured_specific_preview_template_refs()]

    def configured_specific_preview_template_refs(self):
        return sanitize_template_refs(getattr(self, "template_group_config", {}).get("specified_templates") or [])

    def template_group_config_from(self, groups=None, default_order=None, specified_templates=None):
        return {
            "groups": self.sanitize_template_groups(
                self.configured_template_groups() if groups is None else groups
            ),
            "default_order": self.sanitize_default_template_order(
                self.configured_default_template_order() if default_order is None else default_order
            ),
            "specified_templates": self.configured_specific_preview_template_refs()
            if specified_templates is None else sanitize_template_refs(specified_templates),
        }

    @classmethod
    def sanitize_template_group_config(cls, data):
        return {
            "groups": cls.sanitize_template_groups((data or {}).get("groups") or []),
            "default_order": cls.sanitize_default_template_order((data or {}).get("default_order") or []),
            "specified_templates": sanitize_template_refs((data or {}).get("specified_templates") or []),
        }

    @classmethod
    def sanitize_template_groups(cls, groups):
        sanitized = []
        seen_groups = set()
        used_templates = set()
        for group in groups or []:
            if not isinstance(group, dict):
                continue
            name = str(group.get("name") or "").strip()
            if not name or name == DEFAULT_TEMPLATE_GROUP_NAME or name in seen_groups:
                continue
            seen_groups.add(name)
            if isinstance(group.get("templates"), list):
                raw_templates = group.get("templates") or []
            elif isinstance(group.get("children"), list):
                raw_templates = cls.flatten_legacy_group_node_names([group])
            else:
                raw_templates = []
            templates = []
            for template_name in raw_templates:
                template_name = str(template_name or "").strip()
                if not template_name or template_name in used_templates:
                    continue
                used_templates.add(template_name)
                templates.append(template_name)
            sanitized.append({
                "name": name,
                "templates": templates,
            })
        return sanitized

    @staticmethod
    def sanitize_default_template_order(template_names):
        sanitized = []
        seen = set()
        for template_name in template_names or []:
            template_name = str(template_name or "").strip()
            if not template_name or template_name in seen:
                continue
            seen.add(template_name)
            sanitized.append(template_name)
        return sanitized

    @classmethod
    def flatten_legacy_group_node_names(cls, nodes):
        names = []
        for node in nodes or []:
            name = str((node or {}).get("name") or "").strip()
            if name:
                names.append(name)
            names.extend(cls.flatten_legacy_group_node_names((node or {}).get("children") or []))
        return names

    def available_record_template_names(self):
        return template_choice_names()

    def available_record_coco_template_names(self):
        return coco_template_names()

    def available_record_visual_rule_template_names(self):
        return visual_rule_template_names()

    def refresh_template_groups_from_source(self):
        self.reload_record_template_source()
        self.reload_template_groups()
        self.reload_specific_preview_templates()

    def reload_record_template_source(self):
        self.template_icon_cache = {}
        self.preview_match_icon_queue = []
        if hasattr(self, "preview_match_icon_timer"):
            self.preview_match_icon_timer.stop()
        try:
            feature_set = getattr(getattr(og, "executor", None), "feature_set", None)
            if feature_set is not None and hasattr(feature_set, "process_data"):
                feature_set.process_data()
        except Exception as e:
            logger.warning(f"Could not reload record template feature set: {e}")

    def complete_template_groups(self):
        configured = self.configured_template_groups()
        available = self.available_record_template_names()
        available_set = set(available)
        assigned = set(self.flatten_group_template_names(configured))
        groups = []
        for group in configured:
            templates = [
                name for name in group.get("templates", [])
                if not available_set or name in available_set
            ]
            groups.append({"name": group.get("name", ""), "templates": templates})
        ordered_default_templates = [
            name for name in self.configured_default_template_order()
            if name in available_set and name not in assigned
        ]
        default_templates = ordered_default_templates + [
            name for name in available
            if name not in assigned and name not in ordered_default_templates
        ]
        groups.append({
            "name": DEFAULT_TEMPLATE_GROUP_NAME,
            "templates": default_templates,
            "default": True,
        })
        return groups

    @classmethod
    def flatten_group_template_names(cls, groups):
        names = []
        for group in groups or []:
            for name in (group or {}).get("templates") or []:
                name = str(name or "").strip()
                if name:
                    names.append(name)
        return names

    def template_group_search_text(self):
        if not hasattr(self, "template_group_search_edit"):
            return ""
        return self.template_group_search_edit.text().strip()

    @staticmethod
    def template_group_template_matches_search(template_name, search_text, display_name=""):
        search_text = str(search_text or "").strip().casefold()
        if not search_text:
            return True
        haystack = f"{template_name or ''} {display_name or ''}".casefold()
        return search_text in haystack

    def reload_template_groups(self, keep_group_name=None, keep_template_name=None):
        if not hasattr(self, "template_group_list"):
            return
        selected_group_name = keep_group_name or self.selected_template_group_name()
        selected_template_name = keep_template_name if keep_template_name is not None else self.selected_group_template_name()
        self.template_group_config = self.load_template_group_config()
        self.template_group_list.clear()
        select_row = 0
        current_row = 0
        rule_names = self.available_record_visual_rule_template_names()
        search_text = self.template_group_search_text()
        for group in self.complete_template_groups():
            name = str(group.get("name") or "").strip()
            if not name:
                continue
            templates = list(group.get("templates") or [])
            total_count = len(templates)
            if search_text:
                templates = [
                    template_name
                    for template_name in templates
                    if self.template_group_template_matches_search(
                        template_name,
                        search_text,
                        self.record_template_display_name(template_name, template_name in rule_names),
                    )
                ]
                if not templates:
                    continue
            visible_group = dict(group)
            visible_group["templates"] = templates
            visible_group["total_count"] = total_count
            visible_group["searching"] = bool(search_text)
            item = self.template_group_header_item(visible_group)
            self.template_group_list.addItem(item)
            if selected_group_name and name == selected_group_name and not selected_template_name:
                select_row = current_row
            current_row += 1
            if name in self.collapsed_template_groups and not search_text:
                continue
            if not templates:
                empty_item = QListWidgetItem(self.tr("    暂无模板"))
                empty_item.setData(TEMPLATE_GROUP_ITEM_KIND_ROLE, TEMPLATE_GROUP_KIND_TEMPLATE)
                empty_item.setData(TEMPLATE_GROUP_NAME_ROLE, name)
                empty_item.setData(PACKAGE_ROLE, bool(group.get("default")))
                empty_item.setFlags(empty_item.flags() & ~Qt.ItemIsSelectable)
                self.template_group_list.addItem(empty_item)
                current_row += 1
                continue
            for template_name in templates:
                template_item = self.template_group_template_item(
                    template_name,
                    name,
                    bool(group.get("default")),
                    rule_names,
                )
                self.template_group_list.addItem(template_item)
                if (
                    selected_group_name
                    and selected_template_name
                    and name == selected_group_name
                    and template_name == selected_template_name
                ):
                    select_row = current_row
                current_row += 1
        if self.template_group_list.count() == 0:
            item = QListWidgetItem(self.tr("未找到模板"))
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.template_group_list.addItem(item)
        if self.template_group_list.count() > 0:
            self.template_group_list.setCurrentRow(min(select_row, self.template_group_list.count() - 1))
        if self.is_preview_match_tab_active():
            self.reload_preview_match_groups()
        if self.is_specific_preview_tab_active():
            self.reload_specific_preview_templates()

    def reload_specific_preview_templates(self):
        if not hasattr(self, "specific_preview_template_list"):
            return
        self.specific_preview_template_list.clear()
        refs = self.configured_specific_preview_template_refs()
        for ref in refs:
            self.specific_preview_template_list.addItem(self.specific_preview_template_item(ref))
        if not refs:
            item = QListWidgetItem(self.tr("暂无指定模板"))
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.specific_preview_template_list.addItem(item)

    def specific_preview_template_item(self, ref):
        ref = sanitize_template_ref(ref) or template_ref("")
        name = ref["name"]
        is_rule = ref["type"] == TEMPLATE_REF_TYPE_RULE
        item = QListWidgetItem(self.record_template_display_name(name, is_rule))
        item.setData(FRAME_ROLE, name)
        item.setData(TEMPLATE_REF_TYPE_ROLE, ref["type"])
        icon = self.template_icon_for_name(name)
        if not icon.isNull():
            item.setIcon(icon)
        item.setToolTip(f"{name} ({ref['type']})")
        if is_rule:
            item.setForeground(QColor(0, 120, 215))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        return item

    def add_specific_preview_template(self):
        ref = self.choose_specific_preview_template(None)
        if not ref:
            return
        refs = self.configured_specific_preview_template_refs()
        if template_ref_key(ref) not in {template_ref_key(item) for item in refs}:
            refs.append(ref)
        self.save_specific_preview_templates(refs, ref)

    def remove_selected_specific_preview_templates(self):
        if not hasattr(self, "specific_preview_template_list"):
            return
        selected_keys = {
            (
                str(item.data(TEMPLATE_REF_TYPE_ROLE) or TEMPLATE_REF_TYPE_TEMPLATE).strip(),
                str(item.data(FRAME_ROLE) or "").strip(),
            )
            for item in self.specific_preview_template_list.selectedItems()
            if item and str(item.data(FRAME_ROLE) or "").strip()
        }
        if not selected_keys:
            alert_error(self.tr("请先选择要删除的指定模板"))
            return
        refs = [
            ref for ref in self.configured_specific_preview_template_refs()
            if template_ref_key(ref) not in selected_keys
        ]
        self.save_specific_preview_templates(refs)

    def edit_specific_preview_template_item(self, item):
        if item is None:
            return
        old_ref = template_ref(
            str(item.data(FRAME_ROLE) or "").strip(),
            str(item.data(TEMPLATE_REF_TYPE_ROLE) or TEMPLATE_REF_TYPE_TEMPLATE).strip(),
        )
        if not old_ref["name"]:
            return
        new_ref = self.choose_specific_preview_template(old_ref)
        if not new_ref:
            return
        refs = self.configured_specific_preview_template_refs()
        old_key = template_ref_key(old_ref)
        new_key = template_ref_key(new_ref)
        try:
            index = [template_ref_key(ref) for ref in refs].index(old_key)
        except ValueError:
            index = len(refs)
        refs = [ref for ref in refs if template_ref_key(ref) not in {old_key, new_key}]
        refs.insert(min(index, len(refs)), new_ref)
        self.save_specific_preview_templates(refs, new_ref)

    def save_specific_preview_templates(self, refs, keep_ref=None):
        self.template_group_config = self.template_group_config_from(specified_templates=refs)
        self.save_template_group_config()
        self.reload_specific_preview_templates()
        if keep_ref and hasattr(self, "specific_preview_template_list"):
            keep_key = template_ref_key(keep_ref)
            for row in range(self.specific_preview_template_list.count()):
                item = self.specific_preview_template_list.item(row)
                item_key = (
                    str(item.data(TEMPLATE_REF_TYPE_ROLE) or TEMPLATE_REF_TYPE_TEMPLATE).strip(),
                    str(item.data(FRAME_ROLE) or "").strip(),
                )
                if item_key == keep_key:
                    self.specific_preview_template_list.setCurrentRow(row)
                    break

    def choose_specific_preview_template(self, current_ref):
        dialog = MessageBoxBase(self.window())
        dialog.titleLabel = SubtitleLabel(self.tr("选择匹配模板"), dialog)
        dialog.viewLayout.addWidget(dialog.titleLabel)
        combo = QComboBox(dialog)
        make_combo_searchable(combo)
        combo.lineEdit().setPlaceholderText(self.tr("输入或选择模板名"))
        for ref in template_choice_refs():
            add_template_choice_to_combo(
                combo,
                ref["name"],
                ref["type"] == TEMPLATE_REF_TYPE_RULE,
                self.tr("规则模板：由同名前缀的下划线样本自动合成"),
                data=dict(ref),
            )
        current_ref = sanitize_template_ref(current_ref)
        if current_ref:
            index = self.find_template_ref_combo_index(combo, current_ref)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                combo.setCurrentText(current_ref["name"])
        else:
            combo.setCurrentText("")
        dialog.viewLayout.addWidget(combo)
        dialog.yesButton.setText(self.tr("确定"))
        dialog.cancelButton.setText(self.tr("取消"))
        dialog.widget.setMinimumWidth(420)
        if not dialog.exec():
            return None
        return self.template_ref_from_combo(combo)

    @staticmethod
    def find_template_ref_combo_index(combo, ref):
        target_key = template_ref_key(ref)
        for index in range(combo.count()):
            if template_ref_key(combo.itemData(index)) == target_key:
                return index
        return -1

    @staticmethod
    def template_ref_from_combo(combo):
        return template_ref_from_combo_choice(combo)

    @staticmethod
    def template_name_from_combo(combo):
        ref = template_ref_from_combo_choice(combo)
        return ref["name"] if ref else ""

    def reload_preview_match_groups(self):
        if not hasattr(self, "preview_match_group_list"):
            return
        self.preview_match_icon_timer.stop()
        self.preview_match_icon_queue = []
        self.preview_match_group_list.clear()
        self.preview_match_group_items = {}
        groups = list(getattr(self, "preview_match_render_groups", []) or [])
        if not groups:
            item = QListWidgetItem(getattr(self, "preview_match_current_text", "") or self.tr("等待匹配进度..."))
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.preview_match_group_list.addItem(item)
            return
        rule_names = set(getattr(self, "preview_match_render_rule_names", set()) or set())
        self.preview_match_group_list.setUpdatesEnabled(False)
        try:
            for group in groups:
                self.add_preview_match_group_items(group, rule_names)
        finally:
            self.preview_match_group_list.setUpdatesEnabled(True)

    def add_preview_match_group_items(self, group, rule_names):
        name = str(group.get("name") or "").strip()
        if not name:
            return
        header = self.preview_match_group_header_item(group)
        self.preview_match_group_list.addItem(header)
        templates = list(group.get("templates") or [])
        if not templates:
            empty_item = QListWidgetItem(self.tr("    暂无模板"))
            empty_item.setFlags(empty_item.flags() & ~Qt.ItemIsSelectable)
            self.preview_match_group_list.addItem(empty_item)
            return
        for template_name in templates:
            item = self.template_group_template_item(
                template_name,
                name,
                bool(group.get("default")),
                rule_names,
                with_icon=False,
            )
            self.apply_preview_match_template_state(item, name, template_name)
            self.preview_match_group_list.addItem(item)
            self.preview_match_group_items[(name, template_name)] = item
            self.enqueue_preview_match_icon(template_name, item)

    def enqueue_preview_match_icon(self, template_name, item):
        template_name = str(template_name or "").strip()
        if not template_name or item is None:
            return
        if template_name in self.template_icon_cache:
            icon = self.template_icon_cache.get(template_name) or QIcon()
            if not icon.isNull():
                item.setIcon(icon)
            return
        self.preview_match_icon_queue.append((template_name, item))
        if self.is_preview_match_tab_active() and not self.preview_match_icon_timer.isActive():
            self.preview_match_icon_timer.start()

    def load_preview_match_icons_batch(self):
        if not self.is_preview_match_tab_active():
            self.preview_match_icon_timer.stop()
            self.preview_match_icon_queue = []
            return
        loaded = 0
        while self.preview_match_icon_queue and loaded < PREVIEW_MATCH_ICON_BATCH_SIZE:
            template_name, item = self.preview_match_icon_queue.pop(0)
            try:
                if self.preview_match_group_list.row(item) < 0:
                    continue
                icon = self.cached_template_icon_for_name(template_name)
                if not icon.isNull():
                    item.setIcon(icon)
            except RuntimeError:
                continue
            loaded += 1
        if not self.preview_match_icon_queue:
            self.preview_match_icon_timer.stop()

    def cached_template_icon_for_name(self, name):
        name = str(name or "").strip()
        if name in self.template_icon_cache:
            return self.template_icon_cache.get(name) or QIcon()
        icon = self.template_icon_for_name(name)
        self.template_icon_cache[name] = icon
        return icon

    def set_preview_match_render_groups(self, groups, feature_set=None):
        self.preview_match_render_groups = list(groups or [])
        feature_set = feature_set or getattr(getattr(og, "executor", None), "feature_set", None)
        self.preview_match_render_rule_names = {
            str(name)
            for name in getattr(feature_set, "visual_rule_dict", {}).keys()
            if name
        } if feature_set is not None else set()
        if self.is_preview_match_tab_active():
            self.reload_preview_match_groups()

    def refresh_preview_match_item_states(self, template_names=None):
        if not self.is_preview_match_tab_active() or not getattr(self, "preview_match_group_items", None):
            return
        template_name_set = None
        if template_names is not None:
            template_name_set = {
                str(name or "").strip()
                for name in template_names
                if str(name or "").strip()
            }
        self.preview_match_group_list.setUpdatesEnabled(False)
        try:
            for (group_name, template_name), item in list(self.preview_match_group_items.items()):
                if template_name_set is not None and template_name not in template_name_set:
                    continue
                self.apply_preview_match_template_state(item, group_name, template_name)
        finally:
            self.preview_match_group_list.setUpdatesEnabled(True)

    def preview_match_group_header_item(self, group):
        name = str((group or {}).get("name") or "").strip()
        display_name = self.tr(DEFAULT_TEMPLATE_GROUP_NAME) if (group or {}).get("default") else name
        item = QListWidgetItem(self.tr("- {name} ({count})").format(
            name=display_name,
            count=len((group or {}).get("templates") or []),
        ))
        item.setData(FRAME_ROLE, name)
        item.setData(TEMPLATE_GROUP_ITEM_KIND_ROLE, TEMPLATE_GROUP_KIND_GROUP)
        item.setData(TEMPLATE_GROUP_NAME_ROLE, name)
        font = item.font()
        font.setBold(True)
        if (group or {}).get("default"):
            font.setItalic(True)
        item.setFont(font)
        item.setBackground(QColor(55, 55, 55))
        return item

    def apply_preview_match_template_state(self, item, group_name, template_name):
        is_rule = bool(item.data(TEMPLATE_GROUP_RULE_ROLE))
        item.setBackground(QBrush())
        item.setForeground(QColor(0, 120, 215) if is_rule else QBrush())
        font = item.font()
        font.setBold(is_rule)
        item.setFont(font)
        state = self.preview_match_template_state(group_name, template_name)
        if state == "current":
            item.setBackground(QColor(255, 193, 7))
            item.setForeground(QColor(30, 30, 30))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        elif state == "hit":
            item.setBackground(QColor(40, 167, 69))
            item.setForeground(QColor(255, 255, 255))
        elif state == "miss":
            item.setBackground(QColor(255, 228, 230))
            item.setForeground(QColor(90, 45, 45))

    def preview_match_template_state(self, group_name, template_name):
        group_name = str(group_name or "").strip()
        template_name = str(template_name or "").strip()
        current_group = str(getattr(self, "preview_match_current_group", "") or "").strip()
        current_template = str(getattr(self, "preview_match_current_template", "") or "").strip()
        if template_name and template_name == current_template and (not current_group or group_name == current_group):
            return "current"
        hit_counts = getattr(self, "preview_match_hit_counts", {}) or {}
        if template_name in hit_counts:
            return "hit"
        scanned = getattr(self, "preview_match_scanned_templates", set()) or set()
        if template_name in scanned:
            return "miss"
        return ""

    def template_group_header_item(self, group):
        name = str((group or {}).get("name") or "").strip()
        item = QListWidgetItem(self.template_group_display_name(group))
        item.setData(FRAME_ROLE, name)
        item.setData(TEMPLATE_GROUP_ITEM_KIND_ROLE, TEMPLATE_GROUP_KIND_GROUP)
        item.setData(TEMPLATE_GROUP_NAME_ROLE, name)
        item.setData(PACKAGE_ROLE, bool((group or {}).get("default")))
        item.setToolTip(self.tr("双击折叠/展开"))
        font = item.font()
        font.setBold(True)
        if (group or {}).get("default"):
            font.setItalic(True)
        item.setFont(font)
        item.setBackground(QColor(55, 55, 55))
        return item

    def template_group_display_name(self, group):
        name = str((group or {}).get("name") or "").strip()
        if (group or {}).get("default"):
            name = self.tr(DEFAULT_TEMPLATE_GROUP_NAME)
        is_collapsed = str((group or {}).get("name") or "").strip() in self.collapsed_template_groups
        prefix = "+" if is_collapsed and not (group or {}).get("searching") else "-"
        count = len((group or {}).get("templates") or [])
        total_count = int((group or {}).get("total_count", count) or 0)
        if (group or {}).get("searching") and total_count != count:
            count_text = f"{count}/{total_count}"
        else:
            count_text = str(count)
        return self.tr("{prefix} {name} ({count})").format(
            prefix=prefix,
            name=name,
            count=count_text,
        )

    def template_group_template_item(self, name, group_name, is_default_group=False, rule_names=None, with_icon=True):
        rule_names = rule_names if rule_names is not None else self.available_record_visual_rule_template_names()
        is_rule = name in rule_names
        item = QListWidgetItem("    " + self.record_template_display_name(name, is_rule))
        item.setData(FRAME_ROLE, name)
        item.setData(TEMPLATE_GROUP_ITEM_KIND_ROLE, TEMPLATE_GROUP_KIND_TEMPLATE)
        item.setData(TEMPLATE_GROUP_NAME_ROLE, group_name)
        item.setData(PACKAGE_ROLE, bool(is_default_group))
        item.setData(TEMPLATE_GROUP_RULE_ROLE, bool(is_rule))
        if with_icon:
            icon = self.template_icon_for_name(name)
            if not icon.isNull():
                item.setIcon(icon)
        item.setToolTip(name)
        if is_rule:
            item.setForeground(QColor(0, 120, 215))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        return item

    def template_icon_for_name(self, name):
        feature_set = getattr(getattr(og, "executor", None), "feature_set", None)
        if feature_set is None:
            return QIcon()
        feature = getattr(feature_set, "feature_dict", {}).get(name)
        if feature is None:
            rule = getattr(feature_set, "visual_rule_dict", {}).get(name)
            feature = getattr(rule, "feature", None) if rule is not None else None
        mat = getattr(feature, "mat", None)
        if mat is None:
            return QIcon()
        return QIcon(frame_to_pixmap(mat).scaled(44, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def record_template_display_name(self, name, is_rule_template=None):
        if is_rule_template is None:
            is_rule_template = name in self.available_record_visual_rule_template_names()
        return template_choice_display_name(name, is_rule_template)

    def selected_template_group_name(self):
        item = self.template_group_list.currentItem() if hasattr(self, "template_group_list") else None
        return str(item.data(TEMPLATE_GROUP_NAME_ROLE) or "").strip() if item is not None else ""

    def selected_template_group(self):
        name = self.selected_template_group_name()
        return self.template_group_by_name(name)

    def template_group_by_name(self, name):
        for group in self.complete_template_groups():
            if str(group.get("name") or "") == name:
                return group
        return None

    def selected_template_group_is_default(self):
        item = self.template_group_list.currentItem() if hasattr(self, "template_group_list") else None
        return bool(item.data(PACKAGE_ROLE)) if item is not None else False

    def selected_group_template_name(self):
        item = self.template_group_list.currentItem() if hasattr(self, "template_group_list") else None
        if item is None or item.data(TEMPLATE_GROUP_ITEM_KIND_ROLE) != TEMPLATE_GROUP_KIND_TEMPLATE:
            return ""
        return str(item.data(FRAME_ROLE) or "").strip()

    def selected_group_template_records(self, fallback_item=None):
        if not hasattr(self, "template_group_list"):
            return []
        items = list(self.template_group_list.selectedItems() or [])
        if fallback_item is not None and fallback_item not in items:
            items = [fallback_item]
        records = []
        seen = set()
        for item in items:
            if item is None or item.data(TEMPLATE_GROUP_ITEM_KIND_ROLE) != TEMPLATE_GROUP_KIND_TEMPLATE:
                continue
            template_name = str(item.data(FRAME_ROLE) or "").strip()
            group_name = str(item.data(TEMPLATE_GROUP_NAME_ROLE) or "").strip()
            if not template_name or not group_name:
                continue
            key = (group_name, template_name)
            if key in seen:
                continue
            seen.add(key)
            records.append({"group": group_name, "template": template_name})
        return records

    def create_template_group(self):
        name, ok = QInputDialog.getText(
            self,
            self.tr("新建分组"),
            self.tr("分组名"),
        )
        name = str(name or "").strip()
        if not ok or not name:
            return
        if name == DEFAULT_TEMPLATE_GROUP_NAME:
            alert_error(self.tr("默认分组会自动生成，请换一个分组名"))
            return
        groups = self.configured_template_groups()
        if any(group.get("name") == name for group in groups):
            alert_error(self.tr("分组已存在: {name}").format(name=name))
            return
        groups.append({"name": name, "templates": []})
        self.template_group_config = self.template_group_config_from(groups=groups)
        self.save_template_group_config()
        self.reload_template_groups(name)

    def toggle_template_group_item(self, item):
        if item is None or item.data(TEMPLATE_GROUP_ITEM_KIND_ROLE) != TEMPLATE_GROUP_KIND_GROUP:
            return
        group_name = str(item.data(TEMPLATE_GROUP_NAME_ROLE) or "").strip()
        if not group_name:
            return
        if group_name in self.collapsed_template_groups:
            self.collapsed_template_groups.remove(group_name)
        else:
            self.collapsed_template_groups.add(group_name)
        self.reload_template_groups(group_name)

    def show_template_group_context_menu(self, pos):
        item = self.template_group_list.itemAt(pos)
        if item is None:
            return
        if item.data(TEMPLATE_GROUP_ITEM_KIND_ROLE) != TEMPLATE_GROUP_KIND_TEMPLATE:
            return
        if not item.isSelected():
            self.template_group_list.clearSelection()
            item.setSelected(True)
            self.template_group_list.setCurrentItem(item)
        records = self.selected_group_template_records(item)
        template_names = [record["template"] for record in records]
        source_groups = {record["group"] for record in records}
        if not template_names:
            return
        menu = RoundMenu(parent=self.template_group_list)
        if len(source_groups) == 1:
            group_name = next(iter(source_groups))
            menu.addAction(Action(
                self.tr("上移"),
                triggered=lambda checked=False, names=list(template_names), group=group_name: self.reorder_selected_templates(group, names, -1),
            ))
            menu.addAction(Action(
                self.tr("下移"),
                triggered=lambda checked=False, names=list(template_names), group=group_name: self.reorder_selected_templates(group, names, 1),
            ))
        move_menu = RoundMenu(self.tr("移动到分组"), parent=menu)
        target_groups = []
        for group in self.complete_template_groups():
            group_name = str(group.get("name") or "").strip()
            if not group_name:
                continue
            if len(source_groups) == 1 and group_name in source_groups:
                continue
            target_groups.append(group)
        if target_groups:
            for group in target_groups:
                group_name = str(group.get("name") or "").strip()
                display_name = self.tr(DEFAULT_TEMPLATE_GROUP_NAME) if group.get("default") else group_name
                action = Action(
                    display_name,
                    triggered=lambda checked=False, target=group_name, names=list(template_names): self.move_selected_templates_to_group(names, target),
                )
                move_menu.addAction(action)
        else:
            empty_action = Action(self.tr("暂无可移动分组"))
            empty_action.setEnabled(False)
            move_menu.addAction(empty_action)
        menu.addMenu(move_menu)
        menu.exec_(self.template_group_list.viewport().mapToGlobal(pos))

    def reorder_selected_templates(self, group_name, template_names, direction):
        group_name = str(group_name or "").strip()
        template_names = [str(name or "").strip() for name in template_names or [] if str(name or "").strip()]
        group = self.template_group_by_name(group_name)
        if not group or not template_names:
            return
        current_order = list(group.get("templates") or [])
        new_order = self.reorder_template_names(current_order, template_names, direction)
        if new_order == current_order:
            return
        if group_name == DEFAULT_TEMPLATE_GROUP_NAME:
            self.template_group_config = self.template_group_config_from(default_order=new_order)
        else:
            groups = self.configured_template_groups()
            for configured_group in groups:
                if configured_group.get("name") == group_name:
                    configured_group["templates"] = new_order
                    break
            self.template_group_config = self.template_group_config_from(groups=groups)
        self.save_template_group_config()
        self.reload_template_groups(group_name, template_names[0])

    @staticmethod
    def reorder_template_names(template_names, selected_names, direction):
        reordered = [str(name or "").strip() for name in template_names or [] if str(name or "").strip()]
        selected = {
            str(name or "").strip()
            for name in selected_names or []
            if str(name or "").strip()
        }
        if not reordered or not selected or direction == 0:
            return reordered
        if direction < 0:
            indexes = range(1, len(reordered))
            for index in indexes:
                if reordered[index] in selected and reordered[index - 1] not in selected:
                    reordered[index - 1], reordered[index] = reordered[index], reordered[index - 1]
        else:
            indexes = range(len(reordered) - 2, -1, -1)
            for index in indexes:
                if reordered[index] in selected and reordered[index + 1] not in selected:
                    reordered[index + 1], reordered[index] = reordered[index], reordered[index + 1]
        return reordered

    def move_selected_template_to_group(self, template_name, target_group_name):
        self.move_selected_templates_to_group([template_name], target_group_name)

    def move_selected_templates_to_group(self, template_names, target_group_name):
        template_names = [str(name or "").strip() for name in template_names or []]
        template_names = [name for index, name in enumerate(template_names) if name and name not in template_names[:index]]
        target_group_name = str(target_group_name or "").strip()
        if not template_names or not target_group_name:
            return
        self.template_group_config = self.template_group_config_from(
            groups=self.move_templates_to_group(
                self.configured_template_groups(),
                target_group_name,
                template_names,
            ),
            default_order=self.next_default_template_order(
                self.configured_default_template_order(),
                target_group_name,
                template_names,
            ),
        )
        self.save_template_group_config()
        self.collapsed_template_groups.discard(target_group_name)
        self.reload_template_groups(target_group_name, template_names[0])

    def delete_selected_template_group(self):
        group_name = self.selected_template_group_name()
        if not group_name:
            alert_error(self.tr("请先选择分组"))
            return
        item = self.template_group_list.currentItem() if hasattr(self, "template_group_list") else None
        if item is None or item.data(TEMPLATE_GROUP_ITEM_KIND_ROLE) != TEMPLATE_GROUP_KIND_GROUP:
            alert_error(self.tr("请先选择分组标题"))
            return
        if self.selected_template_group_is_default():
            alert_error(self.tr("默认分组不能删除"))
            return
        dialog = MessageBox(
            self.tr("删除分组"),
            self.tr("分组内模板会回到默认分组：{name}").format(name=group_name),
            self.window(),
        )
        if not dialog.exec():
            return
        configured_groups = self.configured_template_groups()
        deleted_templates = []
        groups = []
        for group in configured_groups:
            if group.get("name") == group_name:
                deleted_templates = list(group.get("templates") or [])
            else:
                groups.append(group)
        self.template_group_config = self.template_group_config_from(
            groups=groups,
            default_order=self.next_default_template_order(
                self.configured_default_template_order(),
                DEFAULT_TEMPLATE_GROUP_NAME,
                deleted_templates,
            ),
        )
        self.save_template_group_config()
        self.reload_template_groups(DEFAULT_TEMPLATE_GROUP_NAME)

    def add_template_to_selected_group(self):
        group_name = self.selected_template_group_name()
        if not group_name:
            alert_error(self.tr("请先选择分组"))
            return
        if self.selected_template_group_is_default():
            alert_error(self.tr("默认分组会自动收纳未分配模板，请先选择自定义分组"))
            return
        groups = self.configured_template_groups()
        current_templates = set()
        for group in groups:
            if group.get("name") == group_name:
                current_templates = set(group.get("templates") or [])
                break
        candidates = [name for name in self.available_record_template_names() if name not in current_templates]
        if not candidates:
            alert_error(self.tr("没有可加入的模板"))
            return
        display_to_name = {self.record_template_display_name(name): name for name in candidates}
        display_name, ok = QInputDialog.getItem(
            self,
            self.tr("加入模板"),
            self.tr("模板"),
            list(display_to_name.keys()),
            0,
            False,
        )
        if not ok or not display_name:
            return
        template_name = display_to_name.get(display_name, display_name)
        self.template_group_config = self.template_group_config_from(
            groups=self.add_template_to_group(groups, group_name, template_name),
            default_order=self.next_default_template_order(
                self.configured_default_template_order(),
                group_name,
                [template_name],
            ),
        )
        self.save_template_group_config()
        self.collapsed_template_groups.discard(group_name)
        self.reload_template_groups(group_name, template_name)

    def remove_template_from_selected_group(self):
        records = [
            record for record in self.selected_group_template_records()
            if record.get("group") != DEFAULT_TEMPLATE_GROUP_NAME
        ]
        if not records:
            alert_error(self.tr("请先选择要移出的模板"))
            return
        group_name = records[0]["group"]
        template_names = [record["template"] for record in records]
        self.template_group_config = self.template_group_config_from(
            groups=self.move_templates_to_group(
                self.configured_template_groups(),
                DEFAULT_TEMPLATE_GROUP_NAME,
                template_names,
            ),
            default_order=self.next_default_template_order(
                self.configured_default_template_order(),
                DEFAULT_TEMPLATE_GROUP_NAME,
                template_names,
            ),
        )
        self.save_template_group_config()
        self.reload_template_groups(group_name)

    @classmethod
    def add_template_to_group(cls, groups, group_name, template_name):
        group_name = str(group_name or "").strip()
        template_name = str(template_name or "").strip()
        if not group_name or not template_name or group_name == DEFAULT_TEMPLATE_GROUP_NAME:
            return cls.sanitize_template_groups(groups)
        sanitized = cls.sanitize_template_groups(groups)
        found = False
        for group in sanitized:
            templates = [name for name in group.get("templates", []) if name != template_name]
            if group.get("name") == group_name:
                found = True
                templates.append(template_name)
            group["templates"] = templates
        if not found:
            sanitized.append({"name": group_name, "templates": [template_name]})
        return cls.sanitize_template_groups(sanitized)

    @classmethod
    def remove_template_from_group(cls, groups, group_name, template_name):
        group_name = str(group_name or "").strip()
        template_name = str(template_name or "").strip()
        sanitized = cls.sanitize_template_groups(groups)
        for group in sanitized:
            if group.get("name") == group_name:
                group["templates"] = [name for name in group.get("templates", []) if name != template_name]
                break
        return cls.sanitize_template_groups(sanitized)

    @classmethod
    def move_template_to_group(cls, groups, target_group_name, template_name):
        return cls.move_templates_to_group(groups, target_group_name, [template_name])

    @classmethod
    def next_default_template_order(cls, default_order, target_group_name, template_names):
        target_group_name = str(target_group_name or "").strip()
        template_names = cls.sanitize_default_template_order(template_names)
        order = [
            name for name in cls.sanitize_default_template_order(default_order)
            if name not in template_names
        ]
        if target_group_name == DEFAULT_TEMPLATE_GROUP_NAME:
            order.extend(template_names)
        return cls.sanitize_default_template_order(order)

    @classmethod
    def move_templates_to_group(cls, groups, target_group_name, template_names):
        target_group_name = str(target_group_name or "").strip()
        template_names = [str(name or "").strip() for name in template_names or []]
        templates_to_move = []
        for template_name in template_names:
            if template_name and template_name not in templates_to_move:
                templates_to_move.append(template_name)
        if not target_group_name or not templates_to_move:
            return cls.sanitize_template_groups(groups)
        sanitized = cls.sanitize_template_groups(groups)
        for group in sanitized:
            group["templates"] = [name for name in group.get("templates", []) if name not in templates_to_move]
        if target_group_name != DEFAULT_TEMPLATE_GROUP_NAME:
            target = next((group for group in sanitized if group.get("name") == target_group_name), None)
            if target is None:
                target = {"name": target_group_name, "templates": []}
                sanitized.append(target)
            target.setdefault("templates", []).extend(templates_to_move)
        return cls.sanitize_template_groups(sanitized)

    def preview_match_groups(self):
        if self.is_specific_preview_tab_active() or not self.show_all_templates_check.isChecked():
            return None
        return self.complete_template_groups()

    def is_specific_preview_tab_active(self):
        return (
            hasattr(self, "template_group_tabs")
            and hasattr(self, "specific_preview_tab_index")
            and self.template_group_tabs.currentIndex() == self.specific_preview_tab_index
        )

    def is_preview_match_tab_active(self):
        return (
            hasattr(self, "template_group_tabs")
            and hasattr(self, "preview_match_tab_index")
            and self.template_group_tabs.currentIndex() == self.preview_match_tab_index
        )

    def start_recording(self):
        try:
            settings = {
                "show_all_templates": self.show_all_templates_check.isChecked(),
                "specified_preview_templates": self.configured_specific_preview_template_refs(),
                "confidence_threshold": self.preview_confidence_threshold(),
                "preview_fps": self.preview_fps(),
                "preview_recognition_resolution_percent": self.preview_resolution_percent(),
            }
            self.recorder.start(preview_settings=settings)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.record_timer.start(500)
            alert_info(self.tr("录屏模板录制已开始"))
        except Exception as e:
            alert_error(self.tr("开始录制失败: {e}").format(e=e))

    def stop_recording(self):
        try:
            package = self.recorder.stop()
            self.record_timer.stop()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.update_record_status()
            if package is not None:
                alert_info(self.tr("录屏模板包已生成: {name}").format(name=package.session_id))
                self.package_created.emit()
        except Exception as e:
            alert_error(self.tr("停止录制失败: {e}").format(e=e))

    def update_record_status(self):
        self.duration_label.setText(self.tr("时长 {time}").format(time=format_seconds(self.recorder.elapsed_seconds())))
        self.event_count_label.setText(self.tr("事件 {count}").format(count=self.recorder.event_count()))

    def update_preview_timer(self, value=None):
        self.preview_timer.start(max(1, round(1000 / self.preview_fps())))

    def preview_fps(self):
        return max(1, min(30, self._int_from_edit(self.preview_fps_edit, 8)))

    def preview_resolution_percent(self):
        return max(25, min(100, self._int_from_edit(self.preview_resolution_edit, 50)))

    def preview_confidence_threshold(self):
        return max(0.1, min(1.0, self._float_from_edit(self.confidence_edit, 0.72)))

    @staticmethod
    def _float_from_edit(edit, default):
        text = edit.text().strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default

    def preview_target_height(self, frame):
        if frame is None:
            return 0
        percent = self.preview_resolution_percent()
        if percent >= 100:
            return 0
        return max(1, round(frame.shape[0] * percent / 100))

    @staticmethod
    def _int_from_edit(edit, default):
        text = edit.text().strip()
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            return default

    def update_preview(self):
        try:
            capture_method = getattr(og.device_manager, "capture_method", None)
            frame = capture_method.get_frame() if capture_method is not None else None
            if frame is None:
                return
            pixmap = frame_to_pixmap(frame)
            self.schedule_template_preview_match(frame)
            self.draw_template_preview_boxes(pixmap)
            if not pixmap.isNull():
                self.preview_label.setPixmap(
                    pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        except Exception:
            pass

    def schedule_template_preview_match(self, frame):
        selected_refs = self.selected_preview_template_refs()
        selected_names = [ref["name"] for ref in selected_refs] if selected_refs is not None else None
        if self.is_specific_preview_tab_active() and not selected_refs:
            self.preview_match_boxes = []
            self.match_status_label.setText(self.tr("模板匹配：未指定模板"))
            self.preview_match_current_group = ""
            self.preview_match_current_template = ""
            self.set_preview_match_current(self.tr("未指定模板"))
            self.update_preview_match_hit_counts({})
            return
        if selected_names is None and not self.show_all_templates_check.isChecked():
            self.preview_match_boxes = []
            self.match_status_label.setText(self.tr("模板匹配：未启用"))
            self.preview_match_current_group = ""
            self.preview_match_current_template = ""
            self.set_preview_match_current(self.tr("未启用"))
            self.update_preview_match_hit_counts({})
            return
        if self.preview_match_worker is not None:
            return
        now = time.time()
        if now - self.last_preview_match_at < PREVIEW_MATCH_INTERVAL_SECONDS:
            return
        feature_set = ensure_feature_set_uses_template_coco(
            getattr(getattr(og, "executor", None), "feature_set", None)
        )
        if feature_set is None:
            self.preview_match_boxes = []
            self.match_status_label.setText(self.tr("模板匹配：未初始化"))
            self.preview_match_current_group = ""
            self.preview_match_current_template = ""
            self.set_preview_match_current(self.tr("未初始化"))
            self.update_preview_match_hit_counts({})
            return
        self.last_preview_match_at = now
        self.preview_match_seq += 1
        self.preview_match_progress = None
        self.match_status_label.setText(self.tr("模板匹配：准备中..."))
        self.reset_preview_match_detail()
        self.last_preview_progress_ui_at = 0.0
        self.last_preview_detail_ui_at = 0.0
        match_groups = self.preview_match_groups()
        if match_groups is not None:
            self.set_preview_match_render_groups(match_groups, feature_set)
        elif selected_names is not None:
            self.set_preview_match_render_groups([{
                "name": self.tr("指定模板"),
                "templates": selected_names,
            }], feature_set)
        worker = PreviewMatchWorker(
            self.preview_match_seq,
            feature_set,
            frame.copy(),
            selected_names=None if selected_refs is not None else selected_names,
            selected_refs=selected_refs,
            target_height=self.preview_target_height(frame),
            confidence_threshold=self.preview_confidence_threshold(),
            prefer_visual_rules=True,
            match_groups=match_groups,
        )
        worker.signals.finished.connect(self.on_preview_match_finished)
        worker.signals.failed.connect(self.on_preview_match_failed)
        worker.signals.progress.connect(self.on_preview_match_progress)
        worker.signals.detail.connect(self.on_preview_match_detail)
        self.preview_match_worker = worker
        QThreadPool.globalInstance().start(worker)

    def selected_preview_template_names(self):
        if self.is_specific_preview_tab_active():
            return self.configured_specific_preview_templates()
        return None

    def selected_preview_template_refs(self):
        if self.is_specific_preview_tab_active():
            return self.configured_specific_preview_template_refs()
        return None

    def on_preview_match_finished(self, seq, boxes):
        if seq == self.preview_match_seq:
            self.last_preview_match_at = time.time()
            self.preview_match_boxes = boxes or []
            count = len(boxes or [])
            progress = self.preview_match_progress or {}
            total = int(progress.get("total", 0) or 0)
            elapsed_ms = float(progress.get("elapsed_ms", 0) or 0)
            self.match_status_label.setText(
                self.tr("模板匹配：完成 {total}/{total}，命中 {count}，耗时 {elapsed:.0f}ms").format(
                    total=total,
                    count=count,
                    elapsed=elapsed_ms,
                )
            )
            self.preview_match_current_group = ""
            self.preview_match_current_template = ""
            self.update_preview_match_hits(boxes or [], final=True)
        self.preview_match_worker = None

    def on_preview_match_failed(self, seq, message):
        if seq == self.preview_match_seq:
            self.last_preview_match_at = time.time()
            self.preview_match_boxes = []
            self.match_status_label.setText(self.tr("模板匹配：失败 {message}").format(message=message))
            self.preview_match_current_group = ""
            self.preview_match_current_template = ""
            self.set_preview_match_current(self.tr("匹配失败：{message}").format(message=message))
            self.update_preview_match_hit_counts({})
        self.preview_match_worker = None

    def on_preview_match_progress(self, seq, current, total, matched_count, current_name, elapsed_ms):
        if seq != self.preview_match_seq:
            return
        self.preview_match_progress = {
            "current": current,
            "total": total,
            "matched_count": matched_count,
            "current_name": current_name,
            "elapsed_ms": elapsed_ms,
        }
        if current <= 0:
            self.match_status_label.setText(
                self.tr("模板匹配：开始 0/{total}，命中 0").format(total=total)
            )
            self.set_preview_match_current(self.tr("开始 0/{total}，命中 0").format(total=total))
            self.refresh_preview_match_item_states()
            return
        if not self.should_update_preview_match_progress_ui(current, total):
            return
        display_name = self.preview_match_display_name(current_name)
        self.match_status_label.setText(
            self.tr("模板匹配：{current}/{total}，命中 {matched}，正在匹配 {name}，耗时 {elapsed:.0f}ms").format(
                current=current,
                total=total,
                matched=matched_count,
                name=display_name,
                elapsed=elapsed_ms,
            )
        )
        self.set_preview_match_current(
            self.tr("{current}/{total} 正在匹配：{name}，命中 {matched}，耗时 {elapsed:.0f}ms").format(
                current=current,
                total=total,
                matched=matched_count,
                name=display_name,
                elapsed=elapsed_ms,
            )
        )

    def on_preview_match_detail(self, seq, detail):
        if seq != self.preview_match_seq:
            return
        group_name = str((detail or {}).get("group") or "").strip()
        template_name = str((detail or {}).get("template") or "").strip()
        current = int((detail or {}).get("current", 0) or 0)
        total = int((detail or {}).get("total", 0) or 0)
        searching = bool((detail or {}).get("searching"))
        affected_templates = {template_name} if template_name else set()
        previous_current_template = self.preview_match_current_template
        if previous_current_template:
            affected_templates.add(previous_current_template)
        if searching:
            self.preview_match_current_group = group_name
            self.preview_match_current_template = template_name
        elif template_name:
            self.preview_match_scanned_templates.add(template_name)
            if self.preview_match_current_template == template_name and (
                not self.preview_match_current_group or self.preview_match_current_group == group_name
            ):
                self.preview_match_current_group = ""
                self.preview_match_current_template = ""
        elif (detail or {}).get("skipped"):
            self.preview_match_current_group = ""
            self.preview_match_current_template = ""
        self.update_preview_match_hit_counts((detail or {}).get("hit_counts") or {}, changed_names=affected_templates)
        if not self.should_update_preview_match_detail_ui(current, total, bool((detail or {}).get("skipped"))):
            return
        if (detail or {}).get("skipped"):
            current_text = self.tr("{group} 已命中，后续分组已跳过").format(group=group_name or self.tr("当前分组"))
        else:
            current_text = self.tr("{group} / {template}").format(
                group=group_name or self.tr("全量"),
                template=template_name or "-",
            )
        self.set_preview_match_current(
            self.tr("{current}/{total} 正在匹配：{name}，命中 {matched}，耗时 {elapsed:.0f}ms").format(
                current=int((detail or {}).get("current", 0) or 0),
                total=int((detail or {}).get("total", 0) or 0),
                name=self.preview_match_display_name(current_text),
                matched=int((detail or {}).get("matched_count", 0) or 0),
                elapsed=float((detail or {}).get("elapsed_ms", 0) or 0),
            )
        )
        self.refresh_preview_match_item_states(affected_templates)

    def reset_preview_match_detail(self):
        self.preview_match_current_group = ""
        self.preview_match_current_template = ""
        self.preview_match_scanned_templates = set()
        self.set_preview_match_current(self.tr("准备中..."))
        self.update_preview_match_hit_counts({})

    def set_preview_match_current(self, text):
        self.preview_match_current_text = str(text or "")

    def should_update_preview_match_progress_ui(self, current, total):
        return self.should_update_preview_match_ui("last_preview_progress_ui_at", current, total)

    def should_update_preview_match_detail_ui(self, current, total, skipped=False):
        return bool(skipped) or self.should_update_preview_match_ui("last_preview_detail_ui_at", current, total)

    def should_update_preview_match_ui(self, timestamp_attr, current, total):
        now = time.monotonic()
        last = float(getattr(self, timestamp_attr, 0.0) or 0.0)
        should_update = current <= 1 or (total > 0 and current >= total)
        if not should_update:
            should_update = now - last >= PREVIEW_MATCH_UI_UPDATE_INTERVAL_SECONDS
        if should_update:
            setattr(self, timestamp_attr, now)
        return should_update

    def update_preview_match_hits(self, boxes, final=False):
        hit_counts = {}
        for box in boxes or []:
            name = str(getattr(box, "name", "") or "").strip()
            if not name:
                continue
            hit_counts[name] = hit_counts.get(name, 0) + 1
        self.update_preview_match_hit_counts(hit_counts, final=final)

    def update_preview_match_hit_counts(self, hit_counts, final=False, changed_names=None):
        self.preview_match_hit_counts = {
            str(name): int(count or 0)
            for name, count in (hit_counts or {}).items()
            if str(name or "").strip() and int(count or 0) > 0
        }
        if self.is_preview_match_tab_active():
            self.refresh_preview_match_item_states(changed_names)

    @staticmethod
    def preview_match_display_name(name):
        text = str(name or "")
        if len(text) <= 40:
            return text
        return f"{text[:37]}..."

    def draw_template_preview_boxes(self, pixmap):
        if pixmap.isNull() or not self.preview_match_boxes:
            return
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        for box in self.preview_match_boxes:
            draw_outlined_rect(painter, box.x, box.y, box.width, box.height, QColor(0, 170, 255), 2)
            name = str(getattr(box, "name", "") or "")
            confidence = max(0, min(100, round(float(getattr(box, "confidence", 0) or 0) * 100)))
            label = f"{name} {confidence}%" if name else f"{confidence}%"
            if label:
                draw_outlined_text(painter, int(box.x), max(14, int(box.y) - 5), label)
        painter.end()


class EditVideoTemplatePage(QWidget):
    template_exported = Signal(list)

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self.current_package: VideoTemplatePackage | None = None
        self.current_frame = None
        self.current_frame_item = None
        self.video = None
        self.duration = 0
        self.events = []
        self.ocr_worker = None
        self.export_template_worker = None
        self.skip_export_template_conflict_prompt = False
        self.current_preview_frame = None
        self.current_preview_seconds = 0
        self.validation_match_boxes = []
        self.validation_match_worker = None
        self.validation_match_seq = 0
        self.validation_match_progress = None
        self.validation_match_timer = QTimer(self)
        self.validation_match_timer.setSingleShot(True)
        self.validation_match_timer.timeout.connect(self.run_template_validation_match)
        self.undo_stack = []
        self.redo_stack = []
        self.restoring_history = False
        self.loading_frame_fields = False
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.setSingleShot(True)
        self.auto_save_timer.timeout.connect(self.auto_confirm_current_frame)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.advance_video)
        self.init_ui()
        self.init_shortcuts()
        self.refresh_packages()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.stack.addWidget(self._build_package_list())
        self.stack.addWidget(self._build_editor())

    def init_shortcuts(self):
        self.undo_shortcut = QShortcut(QKeySequence.Undo, self)
        self.undo_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.undo_shortcut.activated.connect(self.undo_operation)
        self.redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.redo_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.redo_shortcut.activated.connect(self.redo_operation)
        self.ctrl_shift_z_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self.ctrl_shift_z_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.ctrl_shift_z_shortcut.activated.connect(self.redo_operation)
        self.delete_keyframe_shortcut = QShortcut(QKeySequence.Delete, self)
        self.delete_keyframe_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.delete_keyframe_shortcut.activated.connect(self.delete_shortcut_target)
        self.draw_annotation_shortcut = QShortcut(QKeySequence(Qt.Key_R), self)
        self.draw_annotation_shortcut.setContext(Qt.ApplicationShortcut)
        self.draw_annotation_shortcut.activated.connect(self.toggle_video_draw_mode_shortcut)
        self.copy_annotation_shortcut = QShortcut(QKeySequence.Copy, self)
        self.copy_annotation_shortcut.setContext(Qt.ApplicationShortcut)
        self.copy_annotation_shortcut.activated.connect(self.copy_video_annotation_shortcut)
        self.duplicate_annotation_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        self.duplicate_annotation_shortcut.setContext(Qt.ApplicationShortcut)
        self.duplicate_annotation_shortcut.activated.connect(self.copy_video_annotation_shortcut)

    def _build_package_list(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        self.search_box = SearchLineEdit()
        self.search_box.setPlaceholderText(self.tr("搜索录屏包..."))
        self.search_box.textChanged.connect(self.refresh_packages)
        toolbar.addWidget(self.search_box, 1)
        refresh_btn = PushButton(FluentIcon.SYNC, self.tr("刷新"))
        refresh_btn.clicked.connect(self.refresh_packages)
        toolbar.addWidget(refresh_btn)
        open_btn = PrimaryPushButton(FluentIcon.PLAY, self.tr("打开录屏包"))
        open_btn.clicked.connect(self.open_selected_package)
        toolbar.addWidget(open_btn)
        layout.addLayout(toolbar)

        self.package_table = QTableWidget(0, 7)
        self.package_table.setHorizontalHeaderLabels([
            self.tr("录制时间"),
            self.tr("时长"),
            self.tr("事件数"),
            self.tr("图像识别状态"),
            self.tr("标注状态"),
            self.tr("导出状态"),
            self.tr("包状态"),
        ])
        self.package_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.package_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.package_table.doubleClicked.connect(self.open_selected_package)
        self.package_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.package_table, 1)
        return page

    def _build_editor(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        back_btn = PushButton(FluentIcon.LEFT_ARROW, self.tr("返回列表"))
        back_btn.clicked.connect(self.back_to_list)
        toolbar.addWidget(back_btn)
        self.ocr_all_btn = PrimaryPushButton(FluentIcon.SEARCH, self.tr("对所有操作帧进行图像识别加工"))
        self.ocr_all_btn.clicked.connect(self.ocr_all_operation_frames)
        toolbar.addWidget(self.ocr_all_btn)
        self.ocr_current_btn = PushButton(FluentIcon.CAMERA, self.tr("对当前帧进行图像识别加工"))
        self.ocr_current_btn.clicked.connect(self.ocr_current_frame)
        toolbar.addWidget(self.ocr_current_btn)
        self.export_btn = PushButton(FluentIcon.SAVE, self.tr("导出确认结果"))
        self.export_btn.clicked.connect(self.export_confirmed)
        toolbar.addWidget(self.export_btn)
        self.cancel_btn = PushButton(FluentIcon.CANCEL, self.tr("取消耗时操作"))
        self.cancel_btn.clicked.connect(self.cancel_worker)
        self.cancel_btn.setVisible(False)
        toolbar.addWidget(self.cancel_btn)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        progress_row = QHBoxLayout()
        self.progress_label = BodyLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_row.addWidget(self.progress_label)
        progress_row.addWidget(self.progress_bar, 1)
        layout.addLayout(progress_row)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_keyframe_panel())
        splitter.addWidget(self._build_video_panel())
        splitter.addWidget(self._build_detail_panel())
        splitter.setSizes([260, 620, 320])
        layout.addWidget(splitter, 1)
        return page

    def _build_keyframe_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(BodyLabel(self.tr("图像识别加工关键帧")))
        self.keyframe_list = QListWidget()
        self.keyframe_list.setIconSize(QPixmap(160, 90).size())
        self.keyframe_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.keyframe_list.currentItemChanged.connect(self.on_keyframe_selected)
        layout.addWidget(self.keyframe_list, 1)
        batch_row = QHBoxLayout()
        select_btn = PushButton(FluentIcon.ADD, self.tr("批量挑选"))
        select_btn.clicked.connect(lambda: self.bulk_update_selected_frames(FRAME_STATUS_SELECTED))
        batch_row.addWidget(select_btn)
        discard_btn = PushButton(FluentIcon.DELETE, self.tr("批量丢弃"))
        discard_btn.clicked.connect(lambda: self.bulk_update_selected_frames(FRAME_STATUS_DISCARDED))
        batch_row.addWidget(discard_btn)
        delete_btn = PushButton(FluentIcon.REMOVE, self.tr("删除"))
        delete_btn.clicked.connect(self.delete_selected_keyframes)
        batch_row.addWidget(delete_btn)
        layout.addLayout(batch_row)
        return panel

    def _build_video_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        edit_row = QHBoxLayout()
        self.video_draw_btn = PushButton(FluentIcon.EDIT, self.tr("新建标注 (R)"))
        self.video_draw_btn.setCheckable(True)
        self.video_draw_btn.toggled.connect(self.set_video_draw_mode)
        edit_row.addWidget(self.video_draw_btn)
        self.video_copy_btn = PushButton(FluentIcon.COPY, self.tr("复制标注"))
        self.video_copy_btn.clicked.connect(self.copy_selected_video_annotation)
        edit_row.addWidget(self.video_copy_btn)
        self.video_delete_btn = PushButton(FluentIcon.DELETE, self.tr("删除标注"))
        self.video_delete_btn.clicked.connect(self.delete_selected_video_annotation)
        edit_row.addWidget(self.video_delete_btn)
        self.video_edit_btn = PushButton(FluentIcon.SETTING, self.tr("修改标注名"))
        self.video_edit_btn.clicked.connect(self.edit_selected_video_annotation)
        edit_row.addWidget(self.video_edit_btn)
        edit_row.addStretch(1)
        layout.addLayout(edit_row)
        self.video_label = AnnotationVideoLabel()
        self.video_label.annotation_created.connect(self.add_annotation_box)
        self.video_label.annotation_boxes_changed.connect(self.replace_annotation_boxes_from_canvas)
        self.video_label.annotation_selection_changed.connect(self.sync_annotation_selection_from_canvas)
        self.video_label.annotation_edit_requested.connect(self.edit_annotation_at_index)
        self.video_label.setMinimumHeight(420)
        self.video_label.setFrameShape(QFrame.StyledPanel)
        self.video_label.setStyleSheet("QLabel { background: rgba(0, 0, 0, 0.06); border-radius: 6px; }")
        layout.addWidget(self.video_label, 1)

        self.detail_label = BodyLabel(self.tr("未选择关键帧"))
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)
        self.ocr_text = QPlainTextEdit()
        self.ocr_text.setReadOnly(True)
        self.ocr_text.setMaximumHeight(92)
        self.ocr_text.setPlaceholderText(self.tr("文字识别文本会显示在这里"))
        layout.addWidget(self.ocr_text)

        controls = QHBoxLayout()
        self.play_btn = PushButton(FluentIcon.PLAY, self.tr("播放"))
        self.play_btn.clicked.connect(self.toggle_playback)
        controls.addWidget(self.play_btn)
        self.time_label = BodyLabel("00:00 / 00:00")
        controls.addWidget(self.time_label)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(lambda value: self.seek_to(value / 1000 * self.duration))
        layout.addWidget(self.slider)
        self.timeline = TimelineWidget()
        self.timeline.seek_requested.connect(self.seek_to)
        layout.addWidget(self.timeline)
        return panel

    def _build_detail_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(BodyLabel(self.tr("模板标注列表")))
        self.annotation_list = QListWidget()
        self.annotation_list.setMinimumHeight(160)
        self.annotation_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.annotation_list.currentItemChanged.connect(self.on_annotation_list_current_changed)
        self.annotation_list.itemDoubleClicked.connect(self.edit_annotation_label)
        layout.addWidget(self.annotation_list, 1)

        draft_row = QHBoxLayout()
        draft_btn = PushButton(FluentIcon.ADD_TO, self.tr("使用文字识别框生成草稿"))
        draft_btn.clicked.connect(self.use_ocr_boxes_as_draft)
        draft_row.addWidget(draft_btn)
        candidate_btn = PushButton(FluentIcon.ADD_TO, self.tr("使用 OpenCV ROI 候选生成草稿"))
        candidate_btn.clicked.connect(self.use_candidate_boxes_as_draft)
        draft_row.addWidget(candidate_btn)
        remove_selected_btn = PushButton(FluentIcon.REMOVE, self.tr("删除选中标注"))
        remove_selected_btn.clicked.connect(self.remove_selected_annotation_boxes)
        draft_row.addWidget(remove_selected_btn)
        layout.addLayout(draft_row)

        agent_row = QHBoxLayout()
        agent_assist_btn = PushButton(FluentIcon.COMMAND_PROMPT, self.tr("使用 Agent 辅助进行加工标注"))
        agent_assist_btn.clicked.connect(self.open_agent_assist_prompt_dialog)
        agent_row.addWidget(agent_assist_btn)
        layout.addLayout(agent_row)

        layout.addWidget(BodyLabel(self.tr("模板匹配校验")))
        self.validation_template_combo = QComboBox()
        make_combo_searchable(self.validation_template_combo)
        self.validation_template_combo.lineEdit().setPlaceholderText(self.tr("输入或选择模板名"))
        self.validation_template_combo.currentTextChanged.connect(self.schedule_template_validation_match)
        layout.addWidget(self.validation_template_combo)
        validation_settings = QHBoxLayout()
        self.validation_confidence_edit = LineEdit()
        validation_confidence_validator = QDoubleValidator(0.1, 1.0, 2, self.validation_confidence_edit)
        validation_confidence_validator.setNotation(QDoubleValidator.StandardNotation)
        self.validation_confidence_edit.setValidator(validation_confidence_validator)
        self.validation_confidence_edit.setText("0.72")
        self.validation_confidence_edit.setFixedWidth(76)
        self.validation_confidence_edit.textChanged.connect(self.schedule_template_validation_match)
        self.validation_resolution_edit = LineEdit()
        self.validation_resolution_edit.setValidator(QIntValidator(25, 100, self.validation_resolution_edit))
        self.validation_resolution_edit.setText("50")
        self.validation_resolution_edit.setFixedWidth(72)
        self.validation_resolution_edit.textChanged.connect(self.schedule_template_validation_match)
        validation_settings.addWidget(BodyLabel(self.tr("置信度阈值")))
        validation_settings.addWidget(self.validation_confidence_edit)
        validation_settings.addWidget(BodyLabel(self.tr("识别分辨率")))
        validation_settings.addWidget(self.validation_resolution_edit)
        validation_settings.addWidget(BodyLabel("%"))
        validation_settings.addStretch(1)
        layout.addLayout(validation_settings)
        self.validation_status_label = BodyLabel(self.tr("未选择模板"))
        self.validation_status_label.setWordWrap(True)
        layout.addWidget(self.validation_status_label)

        self.boxes_edit = QPlainTextEdit()
        self.boxes_edit.setPlaceholderText(self.tr("标注框 JSON，例如 [{\"label\":\"按钮\",\"bbox\":[x,y,w,h]}]"))
        self.boxes_edit.setMinimumHeight(110)
        self.boxes_edit.textChanged.connect(self.refresh_annotation_overlay)
        self.boxes_edit.textChanged.connect(self.schedule_auto_confirm_current_frame)
        layout.addWidget(self.boxes_edit)

        buttons = QHBoxLayout()
        self.export_template_btn = PrimaryPushButton(FluentIcon.SAVE, self.tr("导出模板列表"))
        self.export_template_btn.clicked.connect(self.export_current_annotation_templates)
        buttons.addWidget(self.export_template_btn)
        layout.addLayout(buttons)
        return panel

    def refresh_packages(self):
        query = self.search_box.text().strip().lower() if hasattr(self, "search_box") else ""
        packages = []
        for package in self.store.list_packages():
            summary = package.summary()
            text = " ".join(str(value) for value in summary.values()).lower()
            if not query or query in text:
                packages.append((package, summary))
        self.package_table.setRowCount(len(packages))
        for row, (package, summary) in enumerate(packages):
            values = [
                summary["created_at"],
                format_seconds(summary["duration_seconds"]),
                str(summary["event_count"]),
                summary["ocr_status"],
                summary["annotation_status"],
                summary["export_status"],
                summary["status"],
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 0:
                    item.setData(PACKAGE_ROLE, package.session_id)
                self.package_table.setItem(row, col, item)
        self.package_table.resizeColumnsToContents()

    def reload_template_validation_choices(self):
        if not hasattr(self, "validation_template_combo"):
            return
        current_ref = self.current_validation_template_ref()
        refs = template_choice_refs()
        self.validation_template_combo.blockSignals(True)
        try:
            self.validation_template_combo.clear()
            for ref in refs:
                self.add_validation_template_choice(ref["name"], ref["type"] == TEMPLATE_REF_TYPE_RULE)
            if current_ref:
                self.set_validation_template_current_ref(current_ref)
            else:
                self.validation_template_combo.setCurrentText("")
        finally:
            self.validation_template_combo.blockSignals(False)

    def set_validation_template_current_name(self, name):
        combo = self.validation_template_combo
        index = self.find_validation_template_combo_index(combo, name)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setCurrentText(name)

    def set_validation_template_current_ref(self, ref):
        combo = self.validation_template_combo
        index = self.find_validation_template_ref_combo_index(combo, ref)
        if index >= 0:
            combo.setCurrentIndex(index)
            return
        name = str((ref or {}).get("name") or "").strip() if isinstance(ref, dict) else str(ref or "").strip()
        combo.setCurrentText(name)

    @staticmethod
    def find_validation_template_combo_index(combo, name):
        target = str(name or "").strip()
        if not target:
            return -1
        for index in range(combo.count()):
            if template_name_from_combo_data(combo.itemData(index)) == target:
                return index
        return -1

    @staticmethod
    def find_validation_template_ref_combo_index(combo, ref):
        target_key = template_ref_key(ref)
        if not target_key[1]:
            return -1
        for index in range(combo.count()):
            if template_ref_key(combo.itemData(index)) == target_key:
                return index
        return -1

    def add_validation_template_choice(self, name, is_rule_template):
        add_template_choice_to_combo(
            self.validation_template_combo,
            name,
            is_rule_template,
            self.tr("规则模板：由同名前缀的下划线样本自动合成"),
        )

    def validation_template_display_name(self, name, is_rule_template):
        return template_choice_display_name(name, is_rule_template)

    def available_visual_rule_template_names(self, choices=None):
        return visual_rule_template_names(choices)

    @staticmethod
    def rule_template_names_from_samples(template_names, min_samples=MIN_VISUAL_RULE_SAMPLES):
        return visual_rule_template_names_from_samples(template_names, min_samples)

    def available_template_names(self):
        return template_choice_names()

    @staticmethod
    def group_template_names(names):
        return template_name_groups(names)

    def open_selected_package(self):
        row = self.package_table.currentRow()
        if row < 0:
            alert_error(self.tr("请先选择一个录屏包"))
            return
        item = self.package_table.item(row, 0)
        session_id = item.data(PACKAGE_ROLE)
        self.open_package(self.store.package(session_id))

    def open_package(self, package):
        self.current_package = package
        self.clear_history()
        self.reload_template_validation_choices()
        meta = package.load_meta()
        self.duration = float(meta.get("duration_seconds", 0) or 0)
        self.events = package.load_events()
        self.timeline.set_data(self.duration, self.events, 0)
        self.load_keyframes()
        self.open_video()
        self.seek_to(0)
        self.stack.setCurrentIndex(1)

    def back_to_list(self):
        self.flush_pending_auto_confirm()
        self.validation_match_seq += 1
        self.validation_match_timer.stop()
        self.validation_match_boxes = []
        self.current_preview_frame = None
        self.stop_playback()
        self.clear_history()
        if self.video is not None:
            self.video.release()
            self.video = None
        self.refresh_packages()
        self.stack.setCurrentIndex(0)

    def open_video(self):
        if self.video is not None:
            self.video.release()
        self.video = cv2.VideoCapture(self.current_package.recording_path)
        if self.video.isOpened():
            real_duration = self.video.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, self.video.get(cv2.CAP_PROP_FPS))
            if real_duration > 0:
                self.duration = real_duration
                self.timeline.set_data(self.duration, self.events, 0)

    def seek_to(self, seconds):
        seconds = max(0, min(float(seconds or 0), self.duration))
        if self.video is None:
            return
        self.video.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = self.video.read()
        if ok:
            self.show_video_frame(frame, seconds)

    def show_video_frame(self, frame, seconds):
        self.current_preview_frame = frame.copy()
        self.current_preview_seconds = seconds
        pixmap = frame_to_pixmap(frame)
        ocr_boxes = []
        candidate_boxes = []
        if self.current_frame and abs(seconds - float(self.current_frame.get("time", 0) or 0)) < 0.25:
            ocr_boxes = self.current_frame.get("ocr_boxes") or []
            candidate_boxes = self.current_frame.get("candidate_boxes") or []
        self.video_label.set_frame_pixmap(
            pixmap,
            ocr_boxes,
            self.current_annotation_boxes(),
            candidate_boxes=candidate_boxes,
        )
        self.slider.blockSignals(True)
        value = int(seconds / self.duration * 1000) if self.duration > 0 else 0
        self.slider.setValue(max(0, min(1000, value)))
        self.slider.blockSignals(False)
        self.timeline.set_current_time(seconds)
        self.time_label.setText(f"{format_seconds(seconds)} / {format_seconds(self.duration)}")
        self.schedule_template_validation_match()

    def current_validation_template_name(self):
        ref = self.current_validation_template_ref()
        return ref["name"] if ref else ""

    def current_validation_template_ref(self):
        combo = getattr(self, "validation_template_combo", None)
        return self.validation_combo_template_ref(combo)

    @staticmethod
    def validation_combo_template_name(combo):
        ref = EditVideoTemplatePage.validation_combo_template_ref(combo)
        return ref["name"] if ref else ""

    @staticmethod
    def validation_combo_template_ref(combo):
        if combo is None:
            return None
        return template_ref_from_combo_choice(combo)

    def validation_template_names_for_match(self, name):
        if not name:
            return []
        if self.is_visual_rule_template_name(name):
            return [name]
        choices = [choice for choice in self.available_template_names() if choice == name or choice.startswith(f"{name}_")]
        exact_first = [choice for choice in choices if choice == name]
        grouped = [choice for choice in choices if choice != name]
        return exact_first + grouped if choices else [name]

    def is_visual_rule_template_name(self, name):
        return str(name or "") in self.available_visual_rule_template_names()

    def validation_confidence_threshold(self):
        return max(0.1, min(1.0, self._float_from_edit(self.validation_confidence_edit, 0.72)))

    def validation_resolution_percent(self):
        return max(25, min(100, self._int_from_edit(self.validation_resolution_edit, 50)))

    def validation_target_height(self, frame):
        if frame is None:
            return 0
        percent = self.validation_resolution_percent()
        if percent >= 100:
            return 0
        return max(1, round(frame.shape[0] * percent / 100))

    @staticmethod
    def _float_from_edit(edit, default):
        text = edit.text().strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default

    @staticmethod
    def _int_from_edit(edit, default):
        text = edit.text().strip()
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            return default

    def schedule_template_validation_match(self, *args):
        self.validation_match_seq += 1
        name = self.current_validation_template_name()
        if not name:
            self.validation_match_boxes = []
            if hasattr(self, "validation_status_label"):
                self.validation_status_label.setText(self.tr("未选择模板"))
            if hasattr(self, "video_label"):
                self.video_label.set_template_validation_boxes([], False)
            return
        if self.current_preview_frame is None:
            if hasattr(self, "validation_status_label"):
                self.validation_status_label.setText(self.tr("等待当前帧"))
            return
        self.validation_match_boxes = []
        self.video_label.set_template_validation_boxes([], True)
        self.validation_status_label.setText(self.tr("模板匹配：准备中..."))
        self.validation_match_timer.start(250)

    def run_template_validation_match(self):
        current_ref = self.current_validation_template_ref()
        name = current_ref["name"] if current_ref else ""
        if not name or self.current_preview_frame is None:
            self.schedule_template_validation_match()
            return
        if self.validation_match_worker is not None:
            self.validation_match_timer.start(300)
            return
        feature_set = ensure_feature_set_uses_template_coco(
            getattr(getattr(og, "executor", None), "feature_set", None)
        )
        if feature_set is None:
            self.validation_match_boxes = []
            self.video_label.set_template_validation_boxes([], True)
            self.validation_status_label.setText(self.tr("模板匹配：未初始化"))
            return
        self.validation_match_seq += 1
        self.validation_match_progress = None
        seq = self.validation_match_seq
        selected_refs = [current_ref]
        display_name = self.validation_template_display_name(name, current_ref["type"] == TEMPLATE_REF_TYPE_RULE)
        self.validation_status_label.setText(self.tr("模板匹配：正在匹配 {name}").format(name=display_name))
        worker = PreviewMatchWorker(
            seq,
            feature_set,
            self.current_preview_frame.copy(),
            selected_names=None,
            target_height=self.validation_target_height(self.current_preview_frame),
            confidence_threshold=self.validation_confidence_threshold(),
            horizontal_variance=1,
            vertical_variance=1,
            prefer_visual_rules=False,
            selected_refs=selected_refs,
        )
        worker.signals.finished.connect(self.on_validation_match_finished)
        worker.signals.failed.connect(self.on_validation_match_failed)
        worker.signals.progress.connect(self.on_validation_match_progress)
        self.validation_match_worker = worker
        QThreadPool.globalInstance().start(worker)

    def on_validation_match_finished(self, seq, boxes):
        if seq == self.validation_match_seq:
            self.validation_match_boxes = boxes or []
            self.video_label.set_template_validation_boxes(self.validation_match_boxes, bool(self.current_validation_template_name()))
            count = len(self.validation_match_boxes)
            progress = self.validation_match_progress or {}
            elapsed_ms = float(progress.get("elapsed_ms", 0) or 0)
            self.validation_status_label.setText(
                self.tr("模板匹配：命中 {count}，耗时 {elapsed:.0f}ms").format(
                    count=count,
                    elapsed=elapsed_ms,
                )
            )
        self.validation_match_worker = None

    def on_validation_match_failed(self, seq, message):
        if seq == self.validation_match_seq:
            self.validation_match_boxes = []
            self.video_label.set_template_validation_boxes([], bool(self.current_validation_template_name()))
            self.validation_status_label.setText(self.tr("模板匹配：失败 {message}").format(message=message))
        self.validation_match_worker = None

    def on_validation_match_progress(self, seq, current, total, matched_count, current_name, elapsed_ms):
        if seq != self.validation_match_seq:
            return
        self.validation_match_progress = {
            "current": current,
            "total": total,
            "matched_count": matched_count,
            "current_name": current_name,
            "elapsed_ms": elapsed_ms,
        }
        if current > 0:
            self.validation_status_label.setText(
                self.tr("模板匹配：{current}/{total}，命中 {matched}，耗时 {elapsed:.0f}ms").format(
                    current=current,
                    total=total,
                    matched=matched_count,
                    elapsed=elapsed_ms,
                )
            )

    def toggle_playback(self):
        if self.play_timer.isActive():
            self.stop_playback()
        else:
            self.play_btn.setText(self.tr("暂停"))
            self.play_timer.start(100)

    def stop_playback(self):
        self.play_timer.stop()
        self.play_btn.setText(self.tr("播放"))

    def advance_video(self):
        if self.video is None:
            self.stop_playback()
            return
        ok, frame = self.video.read()
        if not ok:
            self.stop_playback()
            return
        seconds = self.video.get(cv2.CAP_PROP_POS_MSEC) / 1000
        self.show_video_frame(frame, seconds)

    def load_keyframes(self):
        self.keyframe_list.clear()
        if not self.current_package:
            return
        frames = self.current_package.load_ocr_frames()
        total = len(frames)
        for index, frame in enumerate(frames, 1):
            item = QListWidgetItem()
            item.setData(FRAME_ROLE, frame.get("id"))
            item.setSizeHint(QSize(240, self.keyframe_item_height(frame)))
            image_path = os.path.join(self.current_package.folder, frame.get("image_file", ""))
            if os.path.exists(image_path):
                item.setIcon(QIcon(QPixmap(image_path).scaled(160, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            item.setToolTip(self.frame_tooltip(frame, index, total))
            self.keyframe_list.addItem(item)
            self.keyframe_list.setItemWidget(item, self.build_keyframe_item_widget(frame, index, total))

    def build_keyframe_item_widget(self, frame, index, total):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(2)
        title = BodyLabel(
            f"{index}/{total}  {format_seconds(frame.get('time'))}  {self.display_frame_status(frame.get('status'))}"
        )
        title.setWordWrap(False)
        layout.addWidget(title)
        text = self.frame_ocr_summary(frame)
        if text:
            summary = BodyLabel(text)
            summary.setWordWrap(True)
            summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            layout.addWidget(summary)
        return widget

    def keyframe_item_height(self, frame):
        text = self.frame_ocr_summary(frame)
        lines = 1 if text else 0
        if text:
            lines += max(1, min(4, (len(text) + 18) // 19))
        return 30 + lines * 18

    def keyframe_list_text(self, frame, index, total):
        text = self.frame_ocr_summary(frame)
        prefix = f"{index}/{total}  {format_seconds(frame.get('time'))}  {self.display_frame_status(frame.get('status'))}"
        return f"{prefix}\n{text}" if text else prefix

    def frame_tooltip(self, frame, index=None, total=None):
        frame_no = f"{index}/{total}\n" if index is not None and total is not None else ""
        text = frame.get("ocr_text") or "\n".join((frame.get("candidate_context") or {}).get("texts") or [])
        return (
            frame_no +
            f"{frame.get('kind')} {frame.get('source_stage')}\n"
            f"{format_seconds(frame.get('time'))}\n"
            f"{self.display_frame_status(frame.get('status'))}\n"
            f"{text}"
        )

    @staticmethod
    def display_frame_status(status):
        status = str(status or "")
        if status == "OCR 已完成":
            return "图像识别已完成"
        return status

    @staticmethod
    def frame_ocr_summary(frame, max_chars=0):
        text = frame.get("ocr_text") or "\n".join((frame.get("candidate_context") or {}).get("texts") or [])
        text = " / ".join(part.strip() for part in str(text or "").splitlines() if part.strip())
        if max_chars and len(text) > max_chars:
            return f"{text[:max_chars - 3]}..."
        return text

    def on_keyframe_selected(self, current, previous):
        if current is None or not self.current_package:
            return
        self.flush_pending_auto_confirm()
        frame_id = current.data(FRAME_ROLE)
        frame = self.find_frame(frame_id)
        if not frame:
            return
        self.current_frame = frame
        self.current_frame_item = current
        self.seek_to(float(frame.get("time", 0) or 0))
        text = frame.get("ocr_text") or ""
        context = frame.get("candidate_context") or {}
        self.detail_label.setText(
            f"{frame.get('id')}\n{frame.get('kind')} / {frame.get('source_stage')}\n"
            f"{self.display_frame_status(frame.get('status'))}\n"
            f"文字识别框 {len(frame.get('ocr_boxes') or [])} 个，OpenCV ROI 候选 {len(frame.get('candidate_boxes') or [])} 个"
        )
        self.ocr_text.setPlainText(text or "\n".join(context.get("texts") or []))
        annotation = self.find_annotation(frame.get("id"))
        self.loading_frame_fields = True
        try:
            self.boxes_edit.setPlainText(json.dumps((annotation or {}).get("boxes", []), ensure_ascii=False, indent=2))
        finally:
            self.loading_frame_fields = False

    def current_frame_annotation_metadata(self):
        if not self.current_frame:
            return "", ""
        annotation = self.find_annotation(self.current_frame.get("id")) or {}
        return str(annotation.get("template_name") or ""), str(annotation.get("notes") or "")

    def current_default_annotation_label(self):
        boxes = self.current_annotation_boxes()
        selected_index = getattr(getattr(self, "video_label", None), "selected_index", -1)
        if 0 <= selected_index < len(boxes):
            label = str(boxes[selected_index].get("label") or boxes[selected_index].get("source_text") or "").strip()
            if label:
                return label
        for box in boxes:
            label = str(box.get("label") or box.get("source_text") or "").strip()
            if label:
                return label
        template_name, _notes = self.current_frame_annotation_metadata()
        return template_name.strip()

    def current_annotation_boxes(self):
        try:
            data = json.loads(self.boxes_edit.toPlainText().strip() or "[]")
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def open_agent_assist_prompt_dialog(self):
        if not self.current_package or not self.current_frame:
            alert_error(self.tr("请先选择一个图像识别关键帧"))
            return
        dialog = AgentAssistPromptDialog(self, self.agent_assist_prompt_context())
        dialog.exec()

    def agent_assist_prompt_context(self):
        frame = self.current_frame or {}
        package = self.current_package
        frames = package.load_ocr_frames() if package else []
        frame_id = frame.get("id")
        frame_index = 0
        for index, item in enumerate(frames, 1):
            if item.get("id") == frame_id:
                frame_index = index
                break
        image_file = str(frame.get("image_file") or "")
        image_path = os.path.abspath(os.path.join(package.folder, image_file)) if package and image_file else ""
        annotation = self.find_annotation(frame_id) or {}
        return {
            "package_folder": os.path.abspath(package.folder) if package else "",
            "recording_path": os.path.abspath(package.recording_path) if package else "",
            "frames_path": os.path.abspath(package.ocr_frames_path) if package else "",
            "annotations_path": os.path.abspath(package.annotations_path) if package else "",
            "frame_id": str(frame_id or ""),
            "frame_index": frame_index,
            "frame_total": len(frames),
            "time_seconds": float(frame.get("time", 0) or 0),
            "kind": str(frame.get("kind") or ""),
            "source_stage": str(frame.get("source_stage") or ""),
            "image_file": image_file,
            "image_path": image_path,
            "existing_template_name": str(annotation.get("template_name") or ""),
            "existing_notes": str(annotation.get("notes") or ""),
            "existing_boxes": self.current_annotation_boxes(),
        }

    @staticmethod
    def build_agent_assist_prompt(context, target_text):
        context = dict(context or {})
        target = str(target_text or "").strip() or "（请先填写希望识别的画面或对象）"
        existing_boxes = context.get("existing_boxes") or []
        existing_boxes_text = json.dumps(existing_boxes, ensure_ascii=False, indent=2)
        time_seconds = float(context.get("time_seconds", 0) or 0)
        frame_index = int(context.get("frame_index", 0) or 0)
        frame_total = int(context.get("frame_total", 0) or 0)
        return "\n".join([
            "请辅助完成录屏模板的图像识别加工标注。",
            "",
            f"识别目标：{target}",
            "",
            "当前视频工程与指定帧：",
            f"- 录屏包目录：{context.get('package_folder', '')}",
            f"- 视频文件：{context.get('recording_path', '')}",
            f"- 图像识别帧数据：{context.get('frames_path', '')}",
            f"- 标注草稿数据：{context.get('annotations_path', '')}",
            f"- 当前 frame_id：{context.get('frame_id', '')}",
            f"- 当前加工帧序号：第 {frame_index}/{frame_total} 个",
            f"- 当前视频时间：{time_seconds:.3f} 秒",
            f"- 当前帧类型：{context.get('kind', '')} / {context.get('source_stage', '')}",
            f"- 当前帧截图：{context.get('image_path', '')}",
            "",
            "工具使用规范：",
            "1. 优先读取“当前帧截图”路径进行视觉检查；只有截图不存在时，才从视频文件按当前视频时间重新截帧。",
            "2. 如果需要从视频截帧，使用 OpenCV 打开视频，将 CAP_PROP_POS_MSEC 设置为当前视频时间 * 1000 后读取画面。",
            "3. 只分析当前 frame_id 对应的画面，不要把其他帧的画面或标注混进来。",
            "4. 标注框坐标必须使用当前帧截图的原始像素坐标，格式为 [x, y, width, height]，x/y 是左上角。",
            "5. 如果画面里有多个符合识别目标的对象，要分别写成多个 boxes 条目。",
            "6. 写回时只修改标注草稿数据 annotations_draft.json 中当前 frame_id 对应的记录；其他 frame_id 的记录必须原样保留。",
            "7. 如果当前 frame_id 没有标注记录，则新增一条；如果已有记录，则保留 template_name 和 notes，更新 boxes。",
            "8. boxes 条目建议格式：{\"label\":\"标注名\",\"bbox\":[x,y,width,height],\"source\":\"agent\"}。",
            "9. 写入 JSON 时使用 UTF-8，保持 JSON 为数组结构，完成后简单校验文件能被 json.load 读取。",
            "",
            "当前已有标注框：",
            existing_boxes_text,
            "",
            "请完成后说明：识别了哪些对象、写回了几个 boxes、写回的是哪个 frame_id。",
        ])


    def refresh_annotation_overlay(self):
        if hasattr(self, "video_label"):
            self.video_label.set_annotation_boxes(self.current_annotation_boxes())
        self.refresh_annotation_list()

    def refresh_annotation_list(self):
        if not hasattr(self, "annotation_list"):
            return
        self.annotation_list.clear()
        boxes = self.current_annotation_boxes()
        if not boxes:
            item = QListWidgetItem(self.tr("暂无模板标注"))
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.annotation_list.addItem(item)
            return
        for index, box in enumerate(boxes, 1):
            item = QListWidgetItem(self.annotation_list_text(index, box))
            item.setData(FRAME_ROLE, index - 1)
            self.annotation_list.addItem(item)
        selected_index = getattr(getattr(self, "video_label", None), "selected_index", -1)
        if selected_index >= 0:
            self.sync_annotation_selection_from_canvas(selected_index)

    def on_annotation_list_current_changed(self, current, previous):
        if not hasattr(self, "video_label") or current is None:
            return
        index = current.data(FRAME_ROLE)
        if isinstance(index, int):
            self.video_label.select_annotation(index)

    def sync_annotation_selection_from_canvas(self, index):
        if not hasattr(self, "annotation_list"):
            return
        self.annotation_list.blockSignals(True)
        try:
            self.annotation_list.clearSelection()
            self.annotation_list.setCurrentItem(None)
            if index < 0:
                return
            for row in range(self.annotation_list.count()):
                item = self.annotation_list.item(row)
                if item.data(FRAME_ROLE) == index:
                    item.setSelected(True)
                    self.annotation_list.setCurrentItem(item)
                    break
        finally:
            self.annotation_list.blockSignals(False)

    def set_video_draw_mode(self, checked):
        if hasattr(self, "video_label"):
            if checked:
                self.validation_match_seq += 1
                self.validation_match_timer.stop()
                self.validation_match_boxes = []
                self.video_label.set_template_validation_boxes([], False)
                if hasattr(self, "validation_status_label"):
                    self.validation_status_label.setText(self.tr("新建标注中"))
            self.video_label.set_draw_mode(checked)

    def replace_annotation_boxes_from_canvas(self, boxes, history_label):
        if not self.current_frame:
            return
        self.push_history(history_label or "edit_annotation_boxes")
        self.boxes_edit.setPlainText(json.dumps(boxes, ensure_ascii=False, indent=2))

    def copy_selected_video_annotation(self):
        if hasattr(self, "video_label"):
            self.video_label.copy_selected_annotation()

    def delete_selected_video_annotation(self):
        if hasattr(self, "video_label"):
            self.video_label.delete_selected_annotation()

    def edit_selected_video_annotation(self):
        if hasattr(self, "video_label"):
            self.edit_annotation_at_index(self.video_label.selected_index)

    @staticmethod
    def annotation_list_text(index, box):
        bbox = box.get("bbox") or []
        label = str(box.get("label") or box.get("source_text") or "未命名")
        if len(label) > 24:
            label = f"{label[:21]}..."
        bbox_text = ", ".join(str(int(v)) for v in bbox[:4]) if len(bbox) >= 4 else "bbox?"
        extras = []
        source = box.get("source") or ("ocr" if box.get("source_text") else "")
        if source:
            extras.append(str(source))
        if box.get("confidence") is not None:
            try:
                extras.append(f"{float(box.get('confidence')):.2f}")
            except (TypeError, ValueError):
                pass
        suffix = f"  {' / '.join(extras)}" if extras else ""
        return f"{index}. {label}\n[{bbox_text}]{suffix}"

    def push_history(self, label):
        if self.restoring_history or not self.current_package:
            return
        state = self.capture_history_state(label)
        if state is None:
            return
        self.undo_stack.append(state)
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def capture_history_state(self, label):
        if not self.current_package:
            return None
        return {
            "label": label,
            "frames": copy.deepcopy(self.current_package.load_ocr_frames()),
            "annotations": copy.deepcopy(self.current_package.load_annotations()),
            "current_frame_id": self.current_frame.get("id") if self.current_frame else None,
            "boxes_text": self.boxes_edit.toPlainText() if hasattr(self, "boxes_edit") else "",
        }

    def clear_history(self):
        self.undo_stack.clear()
        self.redo_stack.clear()

    def undo_operation(self):
        if self.is_text_input_focused():
            return
        if not self.undo_stack or not self.current_package:
            return
        current = self.capture_history_state("redo")
        state = self.undo_stack.pop()
        if current is not None:
            self.redo_stack.append(current)
        self.restore_history_state(state)

    def redo_operation(self):
        if self.is_text_input_focused():
            return
        if not self.redo_stack or not self.current_package:
            return
        current = self.capture_history_state("undo")
        state = self.redo_stack.pop()
        if current is not None:
            self.undo_stack.append(current)
        self.restore_history_state(state)

    def is_text_input_focused(self):
        focus = self.window().focusWidget() if self.window() else None
        targets = [
            getattr(self, "search_box", None),
            getattr(self, "boxes_edit", None),
            getattr(self, "ocr_text", None),
        ]
        for widget in targets:
            if widget is not None and (focus is widget or widget.isAncestorOf(focus)):
                return True
        return False

    def delete_shortcut_target(self):
        if self.is_text_input_focused():
            return
        focus = self.window().focusWidget() if self.window() else None
        if hasattr(self, "video_label") and self.video_label.selected_index >= 0:
            self.delete_selected_video_annotation()
            return
        if hasattr(self, "annotation_list") and (
            focus is self.annotation_list or self.annotation_list.isAncestorOf(focus)
        ):
            self.remove_selected_annotation_boxes()
            return
        self.delete_selected_keyframes()

    def toggle_video_draw_mode_shortcut(self):
        if self.is_text_input_focused() or not hasattr(self, "video_draw_btn"):
            return
        self.video_draw_btn.setChecked(not self.video_draw_btn.isChecked())

    def copy_video_annotation_shortcut(self):
        if self.is_text_input_focused():
            return
        self.copy_selected_video_annotation()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R and event.modifiers() == Qt.NoModifier:
            self.toggle_video_draw_mode_shortcut()
            event.accept()
            return
        super().keyPressEvent(event)

    def restore_history_state(self, state):
        if not self.current_package:
            return
        self.restoring_history = True
        try:
            self.current_package.save_ocr_frames(copy.deepcopy(state.get("frames") or []))
            self.current_package.save_annotations(copy.deepcopy(state.get("annotations") or []))
            self.load_keyframes()
            frame_id = state.get("current_frame_id")
            self.current_frame_item = self.select_keyframe(frame_id)
            self.current_frame = self.find_frame(frame_id) if frame_id else None
            self.update_current_frame_detail_labels()
            self.loading_frame_fields = True
            try:
                self.boxes_edit.setPlainText(state.get("boxes_text", ""))
            finally:
                self.loading_frame_fields = False
            if self.current_frame:
                self.seek_to(float(self.current_frame.get("time", 0) or 0))
            else:
                self.refresh_annotation_overlay()
        finally:
            self.restoring_history = False

    def select_keyframe(self, frame_id):
        self.keyframe_list.blockSignals(True)
        try:
            self.keyframe_list.clearSelection()
            for index in range(self.keyframe_list.count()):
                item = self.keyframe_list.item(index)
                if item.data(FRAME_ROLE) == frame_id:
                    self.keyframe_list.setCurrentItem(item)
                    item.setSelected(True)
                    return item
            self.keyframe_list.setCurrentItem(None)
            return None
        finally:
            self.keyframe_list.blockSignals(False)

    def update_current_frame_detail_labels(self):
        if not self.current_frame:
            self.detail_label.setText(self.tr("未选择关键帧"))
            self.ocr_text.setPlainText("")
            return
        frame = self.current_frame
        context = frame.get("candidate_context") or {}
        text = frame.get("ocr_text") or ""
        self.detail_label.setText(
            f"{frame.get('id')}\n{frame.get('kind')} / {frame.get('source_stage')}\n"
            f"{self.display_frame_status(frame.get('status'))}\n"
            f"文字识别框 {len(frame.get('ocr_boxes') or [])} 个，OpenCV ROI 候选 {len(frame.get('candidate_boxes') or [])} 个"
        )
        self.ocr_text.setPlainText(text or "\n".join(context.get("texts") or []))
        self.refresh_annotation_list()

    def add_annotation_box(self, box):
        if not self.current_frame:
            return
        self.push_history("add_annotation_box")
        boxes = self.current_annotation_boxes()
        label = self.current_default_annotation_label()
        box["label"] = label
        boxes.append(box)
        self.boxes_edit.setPlainText(json.dumps(boxes, ensure_ascii=False, indent=2))
        if hasattr(self, "video_draw_btn"):
            self.video_draw_btn.setChecked(False)
        if hasattr(self, "video_label"):
            self.video_label.select_annotation(len(boxes) - 1)

    def remove_last_annotation_box(self):
        boxes = self.current_annotation_boxes()
        if not boxes:
            return
        self.push_history("remove_last_annotation_box")
        boxes.pop()
        self.boxes_edit.setPlainText(json.dumps(boxes, ensure_ascii=False, indent=2))

    def remove_selected_annotation_boxes(self):
        if not self.current_frame:
            return
        selected_indexes = sorted(
            {
                item.data(FRAME_ROLE)
                for item in self.annotation_list.selectedItems()
                if isinstance(item.data(FRAME_ROLE), int)
            },
            reverse=True,
        )
        if not selected_indexes:
            alert_error(self.tr("请先选择要删除的模板标注"))
            return
        boxes = self.current_annotation_boxes()
        if not boxes:
            return
        self.push_history("remove_selected_annotation_boxes")
        for index in selected_indexes:
            if 0 <= index < len(boxes):
                boxes.pop(index)
        self.boxes_edit.setPlainText(json.dumps(boxes, ensure_ascii=False, indent=2))

    def edit_annotation_label(self, item):
        if not self.current_frame or item is None:
            return
        index = item.data(FRAME_ROLE)
        if not isinstance(index, int):
            return
        self.edit_annotation_at_index(index)

    def edit_annotation_at_index(self, index):
        if not self.current_frame or not isinstance(index, int):
            return
        boxes = self.current_annotation_boxes()
        if not (0 <= index < len(boxes)):
            return
        box = boxes[index]
        current_label = str(box.get("label") or box.get("source_text") or "")
        new_label, ok = QInputDialog.getText(
            self,
            self.tr("修改标注名"),
            self.tr("标注名"),
            QtLineEdit.EchoMode.Normal,
            current_label,
        )
        if not ok:
            return
        new_label = new_label.strip()
        if new_label == current_label:
            return
        self.push_history("edit_annotation_label")
        box["label"] = new_label
        self.boxes_edit.setPlainText(json.dumps(boxes, ensure_ascii=False, indent=2))

    def find_frame(self, frame_id):
        for frame in self.current_package.load_ocr_frames():
            if frame.get("id") == frame_id:
                return frame
        return None

    def find_annotation(self, frame_id):
        for annotation in self.current_package.load_annotations():
            if annotation.get("frame_id") == frame_id:
                return annotation
        return None

    def selected_frame_ids(self):
        return [
            item.data(FRAME_ROLE)
            for item in self.keyframe_list.selectedItems()
            if item.data(FRAME_ROLE)
        ]

    def delete_selected_keyframes(self):
        if self.is_text_input_focused() or not self.current_package:
            return
        frame_ids = self.selected_frame_ids()
        if not frame_ids and self.current_frame:
            frame_ids = [self.current_frame.get("id")]
        frame_ids = [frame_id for frame_id in frame_ids if frame_id]
        if not frame_ids:
            alert_error(self.tr("请先选择要删除的图像识别关键帧"))
            return
        dialog = MessageBox(
            self.tr("删除图像识别关键帧"),
            self.tr("将从工程中删除 {count} 个图像识别关键帧及对应草稿。可用 Ctrl+Z 撤回。").format(count=len(frame_ids)),
            self.window(),
        )
        if not dialog.exec():
            return
        self.push_history("delete_selected_keyframes")
        frame_id_set = set(frame_ids)
        frames = [
            frame for frame in self.current_package.load_ocr_frames()
            if frame.get("id") not in frame_id_set
        ]
        annotations = [
            annotation for annotation in self.current_package.load_annotations()
            if annotation.get("frame_id") not in frame_id_set
        ]
        self.current_package.save_ocr_frames(frames)
        self.current_package.save_annotations(annotations)
        if self.current_frame and self.current_frame.get("id") in frame_id_set:
            self.current_frame = None
            self.current_frame_item = None
            self.boxes_edit.setPlainText("[]")
            self.update_current_frame_detail_labels()
        self.load_keyframes()
        if self.current_frame:
            self.select_keyframe(self.current_frame.get("id"))
            self.seek_to(float(self.current_frame.get("time", 0) or 0))
        self.refresh_annotation_overlay()
        alert_info(self.tr("已删除图像识别关键帧: {count} 个").format(count=len(frame_ids)))

    def bulk_update_selected_frames(self, status):
        if not self.current_package:
            return
        frame_ids = self.selected_frame_ids()
        if not frame_ids:
            alert_error(self.tr("请先选择关键帧"))
            return
        dialog = MessageBox(
            self.tr("批量操作"),
            self.tr("将影响 {count} 个关键帧，状态改为：{status}").format(count=len(frame_ids), status=status),
            self.window(),
        )
        if not dialog.exec():
            return
        self.push_history("bulk_update_selected_frames")
        frame_id_set = set(frame_ids)
        frames = self.current_package.load_ocr_frames()
        for frame in frames:
            if frame.get("id") in frame_id_set:
                frame["status"] = status
        annotations = self.current_package.load_annotations()
        annotation_by_frame = {item.get("frame_id"): item for item in annotations}
        for frame_id in frame_ids:
            annotation = annotation_by_frame.get(frame_id)
            if status == FRAME_STATUS_DISCARDED:
                if annotation:
                    annotation["status"] = FRAME_STATUS_DISCARDED
            elif annotation is None:
                annotations.append({
                    "frame_id": frame_id,
                    "status": ANNOTATION_STATUS_DRAFT,
                    "template_name": "",
                    "notes": "",
                    "boxes": [],
                })
        self.current_package.save_ocr_frames(frames)
        self.current_package.save_annotations(annotations)
        self.load_keyframes()
        alert_info(self.tr("批量操作完成: {count} 个关键帧").format(count=len(frame_ids)))

    def ocr_all_operation_frames(self):
        self.start_ocr_worker("all")

    def ocr_current_frame(self):
        seconds = self.slider.value() / 1000 * self.duration if self.duration > 0 else 0
        self.start_ocr_worker("current", seconds=seconds)

    def start_ocr_worker(self, mode, seconds=0):
        if not self.current_package:
            return
        self.set_busy(True, self.tr("准备图像识别加工..."), cancellable=True)
        self.ocr_worker = OcrWorker(self.current_package, mode, seconds=seconds)
        self.ocr_worker.signals.progress.connect(self.on_worker_progress)
        self.ocr_worker.signals.finished.connect(self.on_worker_finished)
        self.ocr_worker.signals.failed.connect(self.on_worker_failed)
        QThreadPool.globalInstance().start(self.ocr_worker)

    def cancel_worker(self):
        if self.ocr_worker is not None:
            self.ocr_worker.cancel_requested = True

    def on_worker_progress(self, done, total, message):
        self.progress_label.setText(message)
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(done)

    def on_worker_finished(self, frames):
        self.set_busy(False, "")
        self.ocr_worker = None
        self.load_keyframes()
        alert_info(self.tr("图像识别加工完成: {count} 帧").format(count=len(frames)))

    def on_worker_failed(self, message):
        self.set_busy(False, "")
        self.ocr_worker = None
        alert_error(self.tr("图像识别加工失败: {message}").format(message=message))

    def set_busy(self, busy, message, cancellable=True):
        self.progress_label.setText(message)
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setMaximum(1)
            self.progress_bar.setValue(0)
        self.cancel_btn.setVisible(bool(busy and cancellable))
        buttons = [self.ocr_all_btn, self.ocr_current_btn, self.export_btn]
        export_template_btn = getattr(self, "export_template_btn", None)
        if export_template_btn is not None:
            buttons.append(export_template_btn)
        for button in buttons:
            button.setEnabled(not busy)

    def update_current_frame_status(self, status):
        if not self.current_frame or not self.current_package:
            return
        self.push_history("update_current_frame_status")
        frame_id = self.current_frame.get("id")
        frames = self.current_package.load_ocr_frames()
        for frame in frames:
            if frame.get("id") == frame_id:
                frame["status"] = status
                self.current_frame = frame
                break
        annotations = self.current_package.load_annotations()
        annotation = None
        for item in annotations:
            if item.get("frame_id") == frame_id:
                annotation = item
                break
        if status == FRAME_STATUS_DISCARDED:
            if annotation:
                annotation["status"] = FRAME_STATUS_DISCARDED
        elif annotation is None:
            template_name, notes = self.current_frame_annotation_metadata()
            annotations.append({
                "frame_id": frame_id,
                "status": ANNOTATION_STATUS_DRAFT,
                "template_name": template_name,
                "notes": notes,
                "boxes": self.current_annotation_boxes(),
            })
        else:
            template_name = str(annotation.get("template_name") or "")
            notes = str(annotation.get("notes") or "")
            annotation.update({
                "status": ANNOTATION_STATUS_DRAFT,
                "template_name": template_name,
                "notes": notes,
                "boxes": self.current_annotation_boxes(),
            })
        self.current_package.save_ocr_frames(frames)
        self.current_package.save_annotations(annotations)
        self.load_keyframes()
        alert_info(self.tr("关键帧状态已更新: {status}").format(status=status))

    def use_ocr_boxes_as_draft(self):
        if not self.current_frame:
            return
        self.push_history("use_ocr_boxes_as_draft")
        boxes = []
        fallback_label = self.current_default_annotation_label()
        for item in self.current_frame.get("ocr_boxes") or []:
            bbox = item.get("bbox") or []
            if len(bbox) != 4:
                continue
            boxes.append({
                "label": fallback_label or item.get("text") or "",
                "bbox": [int(v) for v in bbox],
                "source_text": item.get("text", ""),
                "confidence": item.get("confidence", 0),
            })
        self.boxes_edit.setPlainText(json.dumps(boxes, ensure_ascii=False, indent=2))
        self.seek_to(float(self.current_frame.get("time", 0) or 0))

    def use_candidate_boxes_as_draft(self):
        if not self.current_frame:
            return
        self.push_history("use_candidate_boxes_as_draft")
        boxes = []
        fallback_label = self.current_default_annotation_label()
        for item in self.current_frame.get("candidate_boxes") or []:
            bbox = item.get("bbox") or []
            if len(bbox) != 4:
                continue
            boxes.append({
                "label": fallback_label or item.get("label") or "",
                "bbox": [int(v) for v in bbox],
                "source": item.get("source", "operation_point"),
            })
        self.boxes_edit.setPlainText(json.dumps(boxes, ensure_ascii=False, indent=2))
        self.seek_to(float(self.current_frame.get("time", 0) or 0))

    def confirm_current_frame(self):
        self.save_current_frame_annotation(push_history=True, show_alert=True, show_parse_error=True)

    def auto_confirm_current_frame(self):
        self.save_current_frame_annotation(push_history=False, show_alert=False, show_parse_error=False)

    def flush_pending_auto_confirm(self):
        if self.auto_save_timer.isActive():
            self.auto_save_timer.stop()
            self.auto_confirm_current_frame()

    def schedule_auto_confirm_current_frame(self, *args):
        if self.restoring_history or self.loading_frame_fields:
            return
        if not self.current_frame or not self.current_package:
            return
        self.auto_save_timer.start(350)

    def parse_annotation_boxes(self, show_error):
        try:
            boxes = json.loads(self.boxes_edit.toPlainText().strip() or "[]")
            if not isinstance(boxes, list):
                raise ValueError("boxes must be a JSON list")
            return boxes
        except Exception as e:
            if show_error:
                alert_error(self.tr("标注框 JSON 无效: {e}").format(e=e))
            return None

    def refresh_current_keyframe_item(self):
        if not self.current_frame_item or not self.current_frame:
            return
        index = self.keyframe_list.row(self.current_frame_item)
        total = self.keyframe_list.count()
        if index < 0:
            return
        self.current_frame_item.setSizeHint(QSize(240, self.keyframe_item_height(self.current_frame)))
        self.current_frame_item.setToolTip(self.frame_tooltip(self.current_frame, index + 1, total))
        self.keyframe_list.setItemWidget(
            self.current_frame_item,
            self.build_keyframe_item_widget(self.current_frame, index + 1, total),
        )
        self.update_current_frame_detail_labels()

    def save_current_frame_annotation(self, push_history=False, show_alert=False, show_parse_error=True):
        if not self.current_frame or not self.current_package:
            return False
        boxes = self.parse_annotation_boxes(show_parse_error)
        if boxes is None:
            return False
        frame_id = self.current_frame.get("id")
        frames = self.current_package.load_ocr_frames()
        for frame in frames:
            if frame.get("id") == frame_id:
                frame["status"] = FRAME_STATUS_CONFIRMED
                self.current_frame = frame
                break
        annotations = self.current_package.load_annotations()
        annotation = None
        for item in annotations:
            if item.get("frame_id") == frame_id:
                annotation = item
                break
        if annotation is None:
            annotation = {"frame_id": frame_id, "boxes": []}
            annotations.append(annotation)
        if push_history:
            self.push_history("confirm_current_frame")
        template_name = str(annotation.get("template_name") or "")
        notes = str(annotation.get("notes") or "")
        annotation.update({
            "status": ANNOTATION_STATUS_CONFIRMED,
            "template_name": template_name,
            "notes": notes,
            "boxes": boxes,
        })
        self.current_package.save_ocr_frames(frames)
        self.current_package.save_annotations(annotations)
        self.refresh_current_keyframe_item()
        if show_alert:
            alert_info(self.tr("关键帧已确认"))
        return True

    def export_current_annotation_templates(self):
        if not self.current_package or not self.current_frame:
            alert_error(self.tr("请先选择一个图像识别关键帧"))
            return
        if self.export_template_worker is not None:
            return
        self.flush_pending_auto_confirm()
        boxes = self.parse_annotation_boxes(show_error=True)
        if boxes is None:
            return
        valid_boxes = self.valid_template_annotation_boxes(boxes)
        if not valid_boxes:
            alert_error(self.tr("模板标注列表为空，或缺少标注名/bbox"))
            return
        image_file = self.current_frame.get("image_file")
        if not image_file:
            alert_error(self.tr("当前关键帧没有截图文件"))
            return
        source_path = os.path.join(self.current_package.folder, image_file)
        if not os.path.exists(source_path):
            alert_error(self.tr("关键帧截图不存在: {path}").format(path=source_path))
            return

        target_name = self.current_template_export_image_name()
        self.set_busy(True, self.tr("导出模板列表：检查同名模板..."), cancellable=False)
        conflicts = self.find_export_template_conflicts(valid_boxes)
        overwrite_labels = [conflict["label"] for conflict in conflicts]
        if conflicts and not getattr(self, "skip_export_template_conflict_prompt", False):
            self.on_worker_progress(1, 2, self.tr("导出模板列表：发现同名模板，等待确认..."))
            dialog = ExportTemplateConflictDialog(self, conflicts, source_path, target_name)
            if not dialog.exec():
                self.set_busy(False, self.tr("导出模板列表：已取消"))
                return
            if dialog.apply_to_session:
                self.skip_export_template_conflict_prompt = True

        worker = ExportTemplateWorker(source_path, valid_boxes, target_name, overwrite_labels=overwrite_labels)
        worker.signals.progress.connect(self.on_export_template_progress)
        worker.signals.finished.connect(self.on_export_template_finished)
        worker.signals.failed.connect(self.on_export_template_failed)
        self.export_template_worker = worker
        self.on_worker_progress(0, 4, self.tr("导出模板列表：准备中..."))
        QThreadPool.globalInstance().start(worker)

    def on_export_template_progress(self, done, total, message):
        self.on_worker_progress(done, total, message)

    def on_export_template_finished(self, result):
        self.export_template_worker = None
        self.set_busy(False, "")
        labels = list((result or {}).get("labels") or [])
        exported_path = str((result or {}).get("exported_path") or "")
        self.reload_template_validation_choices()
        selected_name = self.template_name_to_validate_after_export(labels)
        if selected_name and hasattr(self, "validation_template_combo"):
            self.set_validation_template_current_name(selected_name)
        self.schedule_template_validation_match()
        self.template_exported.emit(labels)
        alert_info(
            self.tr("已导出模板列表: {count} 个标注 -> {name}\n已写入: {path}").format(
                count=int((result or {}).get("count", len(labels)) or 0),
                name=os.path.basename(exported_path),
                path=exported_path,
            )
        )

    def on_export_template_failed(self, message):
        self.export_template_worker = None
        self.set_busy(False, "")
        alert_error(self.tr("导出模板列表失败: {e}").format(e=message))

    def template_name_to_validate_after_export(self, labels):
        return self.template_name_to_validate_from_rules(labels, self.available_visual_rule_template_names())

    @staticmethod
    def template_name_to_validate_from_rules(labels, rule_names):
        labels = [str(label or "").strip() for label in labels or [] if str(label or "").strip()]
        if not labels:
            return ""
        for label in labels:
            rule_name = EditVideoTemplatePage.rule_name_from_sample_label(label)
            if rule_name and rule_name in rule_names:
                return rule_name
        return labels[0]

    @staticmethod
    def rule_name_from_sample_label(label):
        text = str(label or "")
        if "_" not in text:
            return ""
        prefix, suffix = text.rsplit("_", 1)
        return prefix if prefix and suffix else ""

    def find_export_template_conflicts(self, boxes):
        try:
            from ok.ui.qt.tasks.TemplateTab import load_coco
            return self.find_export_template_conflicts_in_coco(load_coco(), boxes)
        except Exception as e:
            logger.warning(f"Could not inspect template export conflicts: {e}")
            return []

    @staticmethod
    def find_export_template_conflicts_in_coco(coco, boxes):
        labels = []
        seen_labels = set()
        for box in boxes or []:
            label = str((box or {}).get("label") or "").strip()
            if label and label not in seen_labels:
                seen_labels.add(label)
                labels.append(label)
        if not labels:
            return []

        image_map = {
            image.get("id"): image
            for image in (coco or {}).get("images", [])
        }
        category_name_by_id = {
            category.get("id"): str(category.get("name") or "").strip()
            for category in (coco or {}).get("categories", [])
        }
        existing_by_label = {label: [] for label in labels}
        for annotation in (coco or {}).get("annotations", []):
            label = category_name_by_id.get(annotation.get("category_id"), "")
            if label not in existing_by_label:
                continue
            image = image_map.get(annotation.get("image_id"), {}) or {}
            existing_by_label[label].append({
                "annotation_id": annotation.get("id"),
                "image_file": image.get("file_name", ""),
                "bbox": annotation.get("bbox") or [],
            })

        conflicts = []
        for label in labels:
            existing = existing_by_label.get(label) or []
            if not existing:
                continue
            new_boxes = [
                {
                    "label": label,
                    "bbox": (box or {}).get("bbox") or [],
                }
                for box in boxes or []
                if str((box or {}).get("label") or "").strip() == label
            ]
            conflicts.append({
                "label": label,
                "existing": existing,
                "new": new_boxes,
            })
        return conflicts

    @staticmethod
    def export_conflict_detail_text(conflicts, source_path, target_name):
        lines = [
            f"本次新截图：{source_path}",
            f"本次导出图片名：{target_name}",
            "",
        ]
        for conflict in conflicts or []:
            label = str((conflict or {}).get("label") or "")
            lines.append(f"同名模板：{label}")
            lines.append("已有标注：")
            for existing in (conflict or {}).get("existing") or []:
                lines.append(
                    f"  - 图片：{existing.get('image_file', '')}，bbox：{existing.get('bbox') or []}"
                )
            lines.append("本次新标注：")
            for new_box in (conflict or {}).get("new") or []:
                lines.append(f"  - bbox：{new_box.get('bbox') or []}")
            lines.append("")
        return "\n".join(lines).strip()

    def valid_template_annotation_boxes(self, boxes):
        valid = []
        fallback_label = self.current_default_annotation_label()
        for box in boxes or []:
            if not isinstance(box, dict):
                continue
            bbox = box.get("bbox") or []
            label = str(box.get("label") or box.get("source_text") or fallback_label or "").strip()
            if len(bbox) != 4 or not label:
                continue
            try:
                x, y, width, height = [float(value) for value in bbox[:4]]
            except (TypeError, ValueError):
                continue
            if width <= 0 or height <= 0:
                continue
            item = dict(box)
            item["label"] = label
            item["bbox"] = [x, y, width, height]
            valid.append(item)
        return valid

    def write_current_annotations_to_template_coco(self, source_path, boxes, overwrite_labels=None):
        return self.write_annotations_to_template_coco(
            source_path,
            boxes,
            self.current_template_export_image_name(),
            overwrite_labels=overwrite_labels,
        )

    @staticmethod
    def write_annotations_to_template_coco(source_path, boxes, target_name, overwrite_labels=None):
        from ok.ui.qt.tasks.TemplateTab import ensure_template_folder, load_coco, save_coco

        target_folder = ensure_template_folder()
        os.makedirs(target_folder, exist_ok=True)
        target_path = os.path.join(target_folder, target_name)
        shutil.copy2(source_path, target_path)

        image = cv_imread(target_path)
        if image is None:
            raise ValueError("无法读取导出的截图")
        height, width = image.shape[:2]

        coco = load_coco()
        image_entry = EditVideoTemplatePage.upsert_coco_image(coco, target_name, width, height)
        EditVideoTemplatePage.replace_coco_annotations_for_image(
            coco,
            image_entry["id"],
            boxes,
            overwrite_labels=overwrite_labels,
        )
        save_coco(coco)
        return target_path, [box["label"] for box in boxes]

    def current_template_export_image_name(self):
        video_name = self.safe_template_file_stem(self.current_package.session_id)
        sequence = self.current_frame_sequence_number()
        return f"{video_name}_ocr{sequence:04d}.png"

    @staticmethod
    def safe_template_file_stem(text):
        safe = []
        for char in str(text or "video"):
            if char.isalnum() or char in ("-", "_"):
                safe.append(char)
            else:
                safe.append("_")
        value = "".join(safe).strip("_")
        return value or "video"

    def current_frame_sequence_number(self):
        frame_id = self.current_frame.get("id") if self.current_frame else None
        for index, frame in enumerate(self.current_package.load_ocr_frames(), 1):
            if frame.get("id") == frame_id:
                return index
        return 1

    @staticmethod
    def upsert_coco_image(coco, file_name, width, height):
        for image in coco.get("images", []):
            if os.path.basename(image.get("file_name", "")) == file_name:
                image.update({"file_name": file_name, "width": int(width), "height": int(height)})
                return image
        next_id = max([int(image.get("id", 0) or 0) for image in coco.get("images", [])] or [0]) + 1
        image = {"id": next_id, "file_name": file_name, "width": int(width), "height": int(height)}
        coco.setdefault("images", []).append(image)
        return image

    @staticmethod
    def replace_coco_annotations_for_image(coco, image_id, boxes, overwrite_labels=None):
        overwrite_labels = {
            str(label or "").strip()
            for label in overwrite_labels or []
            if str(label or "").strip()
        }
        category_name_by_id = {
            category.get("id"): str(category.get("name") or "").strip()
            for category in coco.get("categories", [])
        }
        coco["annotations"] = [
            annotation for annotation in coco.get("annotations", [])
            if (
                annotation.get("image_id") != image_id
                and category_name_by_id.get(annotation.get("category_id"), "") not in overwrite_labels
            )
        ]
        next_annotation_id = max(
            [int(annotation.get("id", 0) or 0) for annotation in coco.get("annotations", [])] or [0]
        ) + 1
        for box in boxes:
            category_id = EditVideoTemplatePage.get_or_create_coco_category_id(coco, box["label"])
            x, y, width, height = box["bbox"]
            coco.setdefault("annotations", []).append({
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0,
            })
            next_annotation_id += 1

    @staticmethod
    def get_or_create_coco_category_id(coco, name):
        for category in coco.get("categories", []):
            if category.get("name") == name:
                return category.get("id")
        next_id = max([int(category.get("id", 0) or 0) for category in coco.get("categories", [])] or [0]) + 1
        coco.setdefault("categories", []).append({
            "id": next_id,
            "name": name,
            "supercategory": "",
        })
        return next_id

    @staticmethod
    def reload_feature_set_after_template_export():
        try:
            feature_set = ensure_feature_set_uses_template_coco(
                getattr(getattr(og, "executor", None), "feature_set", None)
            )
            if feature_set is not None and hasattr(feature_set, "process_data"):
                feature_set.process_data()
        except Exception as e:
            logger.warning(f"Could not reload feature set after video template export: {e}")

    def export_confirmed(self):
        if not self.current_package:
            return
        self.flush_pending_auto_confirm()
        try:
            manifest = self.current_package.export_confirmed()
            alert_info(self.tr("导出完成: {count} 帧").format(count=len(manifest.get("frames", []))))
            self.refresh_packages()
        except Exception as e:
            alert_error(self.tr("导出失败: {e}").format(e=e))


class VideoTemplateTab(QWidget):
    def __init__(self, config=None):
        super().__init__()
        self.setObjectName("VideoTemplateTab")
        self.store = VideoTemplateStore()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.tabs = QTabWidget()
        self.record_page = RecordVideoTemplatePage(self.store)
        self.edit_page = EditVideoTemplatePage(self.store)
        self.record_page.package_created.connect(self.edit_page.refresh_packages)
        self.edit_page.template_exported.connect(lambda _labels: self.record_page.refresh_template_groups_from_source())
        self.tabs.addTab(self.record_page, self.tr("录屏"))
        self.tabs.addTab(self.edit_page, self.tr("编辑录屏模板"))
        layout.addWidget(self.tabs, 1)
