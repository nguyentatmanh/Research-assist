"""Exercise dependency bootstrap without installing/changing the test environment."""

import importlib.util
import json
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_module():
    spec = importlib.util.spec_from_file_location(
        "kaggle_bootstrap_test", ROOT / "notebooks/kaggle_environment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def versions(module, numpy="2.2.6"):
    result = dict(r.split("==", 1) for r in module.TRAINING_REQUIREMENTS)
    result.update({name: result.get(name) for name in module.WATCHED_MODULES.values()})
    result["numpy"] = numpy
    return result


def test_stale_numpy_requires_actual_kernel_restart_once(tmp_path, monkeypatch):
    m = bootstrap_module()
    before, after = versions(m, "2.4.0"), versions(m)
    states = iter([before, after, after, after, after, after])
    monkeypatch.setattr(m, "installed_versions", lambda: next(states))
    loaded = {"numpy": "2.4.0"}
    monkeypatch.setattr(m, "loaded_versions", lambda: loaded)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return types.SimpleNamespace(returncode=0, stdout='{"imports_ok": true}', stderr="")

    monkeypatch.setattr(m.subprocess, "run", run)
    marker = tmp_path / "environment.json"
    first = m.install_training_environment(marker, runtime_id="kernel-A")
    assert not first["ready"] and first["restart_required_for"] == "kernel-A"
    # Running the setup cell again in the same process must not bypass the guard.
    repeated = m.install_training_environment(marker, runtime_id="kernel-A")
    assert not repeated["ready"]
    loaded["numpy"] = "2.2.6"
    ready = m.install_training_environment(marker, runtime_id="kernel-B")
    assert ready["ready"] and ready["restart_required_for"] is None
    pip_calls = [args for args in calls if "pip" in args]
    assert len(pip_calls) == 1 and not any("torch==" in arg for arg in pip_calls[0])
    assert json.loads(marker.read_text())["ready"] is True


def test_unknown_preloaded_kernel_requests_restart_even_if_versions_match(tmp_path, monkeypatch):
    m = bootstrap_module()
    monkeypatch.setattr(m, "installed_versions", lambda: versions(m))
    monkeypatch.setattr(m, "loaded_versions", lambda: {"numpy": "2.2.6"})
    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(
            returncode=0,
            stdout='{"imports_ok": true}',
            stderr="",
        ),
    )
    status = m.install_training_environment(tmp_path / "state.json", runtime_id="old-loaded-kernel")
    assert not status["ready"]


def test_corrupt_numpy_on_disk_gets_one_repair_then_fresh_process_check(tmp_path, monkeypatch):
    m = bootstrap_module()
    monkeypatch.setattr(m, "installed_versions", lambda: versions(m))
    monkeypatch.setattr(m, "loaded_versions", dict)
    calls = []
    checks = 0

    def run(args, **kwargs):
        nonlocal checks
        calls.append(args)
        if "-c" in args:
            checks += 1
            if checks == 1:
                return types.SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="AttributeError: 'numpy.ufunc' object has no attribute '__module__'",
                )
        return types.SimpleNamespace(returncode=0, stdout='{"imports_ok": true}', stderr="")

    monkeypatch.setattr(m.subprocess, "run", run)
    status = m.install_training_environment(tmp_path / "state.json", runtime_id="fresh-kernel")
    assert status["ready"] and status["numpy_repaired"]
    repair = next(args for args in calls if "--force-reinstall" in args)
    assert "--no-deps" in repair and repair[-1] == "numpy==2.2.6" and checks == 2


def test_unrelated_import_failure_stops_before_training_without_numpy_reinstall(
    tmp_path, monkeypatch, capsys
):
    m = bootstrap_module()
    monkeypatch.setattr(m, "installed_versions", lambda: versions(m))
    monkeypatch.setattr(m, "loaded_versions", dict)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return types.SimpleNamespace(returncode=1, stdout="", stderr="original unrelated failure")

    monkeypatch.setattr(m.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Training has NOT started"):
        m.install_training_environment(tmp_path / "state.json")
    assert "original unrelated failure" in capsys.readouterr().out
    assert len(calls) == 1


def test_generated_notebook_blocks_imports_until_setup_is_ready():
    book = json.loads((ROOT / "notebooks/01_kaggle_finetune_e5.ipynb").read_text(encoding="utf-8"))
    # A pending restart is an ordinary setup status, not a fake traceback. Every
    # later code cell must skip all imports, downloads and training when not ready.
    code_cells = ["".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code"]
    for source in code_cells[1:]:
        namespace = {}
        exec(compile(source, "guarded-cell", "exec"), namespace)
        assert set(namespace) == {"__builtins__"}


def test_notebook_clock_survives_kernel_restart(tmp_path):
    import runpy
    import time

    setup = runpy.run_path(str(ROOT / "scripts/build_kaggle_notebooks.py"))["SETUP"]
    timer = setup.split("# Run in a fresh Kaggle GPU session", 1)[0]
    clock_file = tmp_path / "session-clock.json"
    clock_file.write_text(json.dumps({"started_epoch": time.time() - 3600}))
    timer = timer.replace(
        'Path("/kaggle/temp/researchdesk-session-clock.json")', f"Path({str(clock_file)!r})"
    )
    first, second = {"Path": Path, "json": json}, {"Path": Path, "json": json}
    exec(compile(timer, "setup-timer", "exec"), first)
    exec(compile(timer, "setup-timer", "exec"), second)
    assert time.monotonic() - second["_NOTEBOOK_STARTED_MONOTONIC"] >= 3600
    assert second["_NOTEBOOK_STARTED_MONOTONIC"] <= first["_NOTEBOOK_STARTED_MONOTONIC"] + 0.05


def test_kernel_identity_changes_after_restart_even_if_pid_is_reused(monkeypatch):
    m = bootstrap_module()
    monkeypatch.setattr(m.sys, "_researchdesk_runtime_id", None, raising=False)
    first = m.current_runtime_identity()
    assert m.current_runtime_identity() == first
    # Simulate fresh Python memory while keeping the same OS PID.
    m.sys._researchdesk_runtime_id = None
    assert m.current_runtime_identity() != first
