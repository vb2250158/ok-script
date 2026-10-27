import os

from ok import OK
from ok.util.file import workspace_path_candidates


def test_apply_working_directory_keeps_project_paths_relative(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    monkeypatch.chdir(project_root)

    config = {
        "working_directory": "panghu",
        "template_matching": {"coco_feature_json": "ok_templates/coco_annotations.json"},
    }

    OK.__new__(OK)._apply_working_directory(config)

    assert os.getcwd() == str(project_root / "panghu")
    assert config["project_root"] == str(project_root)
    assert config["resolved_working_directory"] == str(project_root / "panghu")
    assert config["template_matching"]["coco_feature_json"] == "ok_templates/coco_annotations.json"


def test_workspace_path_candidates_prefers_configured_workspace(tmp_path, monkeypatch):
    cwd = tmp_path / "current"
    workspace = tmp_path / "workspace"
    project_root = tmp_path / "repo"
    cwd.mkdir()
    workspace.mkdir()
    project_root.mkdir()
    monkeypatch.chdir(cwd)

    candidates = workspace_path_candidates(
        "ok_templates/coco_annotations.json",
        {
            "project_root": str(project_root),
            "resolved_working_directory": str(workspace),
        },
    )

    assert candidates == [
        str(workspace / "ok_templates" / "coco_annotations.json"),
        str(cwd / "ok_templates" / "coco_annotations.json"),
        str(project_root / "ok_templates" / "coco_annotations.json"),
    ]


def test_ok_resolves_template_matching_path_from_workspace(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    workspace = project_root / "panghu"
    root_coco = project_root / "ok_templates" / "coco_annotations.json"
    workspace_coco = workspace / "ok_templates" / "coco_annotations.json"
    root_coco.parent.mkdir(parents=True)
    workspace_coco.parent.mkdir(parents=True)
    root_coco.write_text("{}", encoding="utf-8")
    workspace_coco.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(workspace)

    ok = OK.__new__(OK)
    ok.config = {
        "project_root": str(project_root),
        "resolved_working_directory": str(workspace),
    }

    assert (
        ok._resolve_configured_workspace_path("ok_templates/coco_annotations.json")
        == str(workspace_coco)
    )
