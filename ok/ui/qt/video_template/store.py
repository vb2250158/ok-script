import json
import os
import shutil
from datetime import datetime
from typing import Any


VIDEO_TEMPLATE_ROOT = os.path.join("recordings", "video_templates")
META_JSON = "meta.json"
EVENTS_JSON = "events.json"
OCR_FRAMES_JSON = "ocr_frames.json"
ANNOTATIONS_DRAFT_JSON = "annotations_draft.json"
RECORDING_MP4 = "recording.mp4"
KEYFRAMES_DIR = "keyframes"
EXPORTS_DIR = "exports"


PACKAGE_STATUS_RAW = "未加工"
PACKAGE_STATUS_OCR = "已图像识别"
PACKAGE_STATUS_KEYFRAMES = "有关键帧"
PACKAGE_STATUS_DRAFT = "有标注草稿"
PACKAGE_STATUS_EXPORTED = "已导出"

FRAME_STATUS_OCR_DONE = "图像识别已完成"
FRAME_STATUS_SELECTED = "已被挑选"
FRAME_STATUS_ANNOTATED = "已标注"
FRAME_STATUS_CONFIRMED = "已确认"
FRAME_STATUS_DISCARDED = "已丢弃"

ANNOTATION_STATUS_DRAFT = "草稿"
ANNOTATION_STATUS_NEEDS_FIX = "需要矫正"
ANNOTATION_STATUS_CONFIRMED = "已确认"
ANNOTATION_STATUS_EXPORTED = "已导出"


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def safe_session_id(created_at=None):
    text = created_at or datetime.now().strftime("%Y%m%d_%H%M%S")
    return text.replace(":", "").replace("-", "").replace("T", "_").replace(" ", "_")


def read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


class VideoTemplatePackage:
    """Filesystem-backed package for one recording session."""

    def __init__(self, folder):
        self.folder = os.path.abspath(folder)

    @property
    def session_id(self):
        return os.path.basename(self.folder)

    @property
    def recording_path(self):
        return os.path.join(self.folder, RECORDING_MP4)

    @property
    def meta_path(self):
        return os.path.join(self.folder, META_JSON)

    @property
    def events_path(self):
        return os.path.join(self.folder, EVENTS_JSON)

    @property
    def ocr_frames_path(self):
        return os.path.join(self.folder, OCR_FRAMES_JSON)

    @property
    def annotations_path(self):
        return os.path.join(self.folder, ANNOTATIONS_DRAFT_JSON)

    @property
    def keyframes_folder(self):
        return os.path.join(self.folder, KEYFRAMES_DIR)

    @property
    def exports_folder(self):
        return os.path.join(self.folder, EXPORTS_DIR)

    def load_meta(self):
        return read_json(self.meta_path, {})

    def save_meta(self, meta):
        write_json(self.meta_path, meta)

    def update_meta(self, **values):
        meta = self.load_meta()
        meta.update(values)
        meta["updated_at"] = now_iso()
        self.save_meta(meta)
        return meta

    def load_events(self):
        return read_json(self.events_path, [])

    def save_events(self, events):
        write_json(self.events_path, events)
        self.update_meta(event_count=len(events))

    def load_ocr_frames(self):
        return read_json(self.ocr_frames_path, [])

    def save_ocr_frames(self, frames):
        write_json(self.ocr_frames_path, frames)
        self.update_meta(ocr_frame_count=len(frames), status=self.derive_status(frames=frames))

    def load_annotations(self):
        return read_json(self.annotations_path, [])

    def save_annotations(self, annotations):
        write_json(self.annotations_path, annotations)
        self.update_meta(annotation_count=len(annotations), status=self.derive_status(annotations=annotations))

    def derive_status(self, frames=None, annotations=None):
        meta = self.load_meta()
        frames = self.load_ocr_frames() if frames is None else frames
        annotations = self.load_annotations() if annotations is None else annotations
        if meta.get("exported_at"):
            return PACKAGE_STATUS_EXPORTED
        if any(a.get("status") for a in annotations):
            return PACKAGE_STATUS_DRAFT
        if any(f.get("status") in (FRAME_STATUS_SELECTED, FRAME_STATUS_ANNOTATED, FRAME_STATUS_CONFIRMED)
               for f in frames):
            return PACKAGE_STATUS_KEYFRAMES
        if frames:
            return PACKAGE_STATUS_OCR
        return PACKAGE_STATUS_RAW

    def summary(self):
        meta = self.load_meta()
        events = self.load_events()
        frames = self.load_ocr_frames()
        annotations = self.load_annotations()
        duration = meta.get("duration_seconds", 0)
        status = self.derive_status(frames=frames, annotations=annotations)
        return {
            "session_id": self.session_id,
            "folder": self.folder,
            "created_at": meta.get("created_at", ""),
            "duration_seconds": duration,
            "event_count": len(events),
            "ocr_status": PACKAGE_STATUS_OCR if frames else PACKAGE_STATUS_RAW,
            "annotation_status": PACKAGE_STATUS_DRAFT if annotations else PACKAGE_STATUS_RAW,
            "export_status": PACKAGE_STATUS_EXPORTED if meta.get("exported_at") else PACKAGE_STATUS_RAW,
            "status": status,
        }

    def export_confirmed(self):
        frames = self.load_ocr_frames()
        annotations = self.load_annotations()
        confirmed_frame_ids = {
            a.get("frame_id")
            for a in annotations
            if a.get("status") == ANNOTATION_STATUS_CONFIRMED
        }
        confirmed_frames = [
            frame for frame in frames
            if frame.get("status") == FRAME_STATUS_CONFIRMED or frame.get("id") in confirmed_frame_ids
        ]
        os.makedirs(self.exports_folder, exist_ok=True)
        export_frames = []
        for frame in confirmed_frames:
            image_file = frame.get("image_file")
            if not image_file:
                continue
            src = os.path.join(self.folder, image_file)
            if not os.path.exists(src):
                continue
            dst = os.path.join(self.exports_folder, os.path.basename(image_file))
            shutil.copy2(src, dst)
            item = dict(frame)
            item["exported_image_file"] = os.path.relpath(dst, self.folder).replace("\\", "/")
            export_frames.append(item)

        export_annotations = [
            dict(a, status=ANNOTATION_STATUS_EXPORTED)
            for a in annotations
            if a.get("frame_id") in {frame.get("id") for frame in export_frames}
        ]
        manifest = {
            "session_id": self.session_id,
            "exported_at": now_iso(),
            "frames": export_frames,
            "annotations": export_annotations,
        }
        manifest_path = os.path.join(self.exports_folder, "manifest.json")
        write_json(manifest_path, manifest)
        self.update_meta(exported_at=manifest["exported_at"], export_count=len(export_frames),
                         status=PACKAGE_STATUS_EXPORTED)
        return manifest


class VideoTemplateStore:
    """Small interface for listing and creating recording packages."""

    def __init__(self, root=None):
        self.root = os.path.abspath(root or VIDEO_TEMPLATE_ROOT)

    def ensure_root(self):
        os.makedirs(self.root, exist_ok=True)

    def create_package(self, meta: dict[str, Any] | None = None):
        self.ensure_root()
        created_at = now_iso()
        session_id = safe_session_id(created_at)
        folder = os.path.join(self.root, session_id)
        index = 1
        while os.path.exists(folder):
            folder = os.path.join(self.root, f"{session_id}_{index}")
            index += 1
        os.makedirs(os.path.join(folder, KEYFRAMES_DIR), exist_ok=True)
        package = VideoTemplatePackage(folder)
        package.save_meta({
            "session_id": package.session_id,
            "created_at": created_at,
            "updated_at": created_at,
            "status": PACKAGE_STATUS_RAW,
            **(meta or {}),
        })
        package.save_events([])
        package.save_ocr_frames([])
        package.save_annotations([])
        return package

    def package(self, session_id):
        folder = os.path.join(self.root, session_id)
        return VideoTemplatePackage(folder)

    def list_packages(self):
        self.ensure_root()
        packages = []
        for name in os.listdir(self.root):
            folder = os.path.join(self.root, name)
            if not os.path.isdir(folder):
                continue
            if os.path.exists(os.path.join(folder, META_JSON)):
                packages.append(VideoTemplatePackage(folder))
        packages.sort(key=lambda package: package.load_meta().get("created_at", ""), reverse=True)
        return packages
