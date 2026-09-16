from __future__ import annotations

from pathlib import Path

import pytest

from src.modules.artifacts import ArtifactStore, SUCCESS_FILE


def test_stage_is_reused_only_after_successful_publish(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("input")
    store = ArtifactStore(tmp_path, "bgl", Path(__file__).parents[1])

    def build(stage_dir: Path) -> dict[str, Path]:
        output = stage_dir / "result.txt"
        output.write_text("result")
        return {"result": output}

    outputs, manifest, reused = store.stage(
        stage="parse",
        stage_config={"version": 1},
        inputs=[source],
        build=build,
    )
    assert not reused
    assert outputs["result"].read_text() == "result"
    assert (outputs["result"].parent / SUCCESS_FILE).exists()
    assert manifest["outputs"] == {"result": "result.txt"}

    outputs, _, reused = store.stage(
        stage="parse",
        stage_config={"version": 1},
        inputs=[source],
        build=lambda _: pytest.fail("cached stage must not rebuild"),
    )
    assert reused
    assert outputs["result"].read_text() == "result"


def test_failed_stage_never_publishes_success_manifest(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("input")
    store = ArtifactStore(tmp_path, "hdfs", Path(__file__).parents[1])

    def fail(stage_dir: Path) -> dict[str, Path]:
        (stage_dir / "partial.txt").write_text("incomplete")
        raise RuntimeError("simulated runtime disconnect")

    with pytest.raises(RuntimeError, match="disconnect"):
        store.stage(stage="build", stage_config={}, inputs=[source], build=fail)

    assert not list((tmp_path / "artifacts").rglob(SUCCESS_FILE))
