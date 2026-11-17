"""Print matching vLLM launch arguments and Research-assist per-agent routes."""

import argparse
import json
import shlex
from pathlib import Path

from training_lab.engine.contracts import LAB, ROLES


def deployment(weights: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    routes = {}
    modules = []
    bases = set()
    for role in ROLES:
        folder = weights / role
        if not folder.is_dir():
            continue
        adapter = folder / "adapter"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["role"] != role or not (adapter / "adapter_config.json").is_file():
            raise ValueError(f"Incomplete adapter export for {role}")
        if not list(adapter.glob("adapter_model.*")):
            raise ValueError(f"No adapter weights for {role}")
        if any(char.isspace() for char in str(adapter.resolve())):
            raise ValueError("Use an adapter path without spaces for vLLM --lora-modules")
        bases.add((manifest["base_model"], manifest["revision"]))
        name = f"ra-{role.replace('_', '-')}"
        routes[role] = {"provider": "openai", "model": name}
        modules.append(f"{name}={adapter.resolve()}")

    if len(bases) != 1:
        raise ValueError(
            "Serve at least one adapter; all adapters in one server must use the same base revision"
        )
    base, revision = bases.pop()
    command = [
        "vllm",
        "serve",
        base,
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--enable-lora",
        "--max-lora-rank",
        "16",
        "--lora-modules",
        *modules,
    ]
    if revision:
        command.extend(["--revision", revision])
    return command, routes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=LAB / "final_weights")
    args = parser.parse_args()
    command, routes = deployment(args.weights)
    print(shlex.join(command))
    print()
    print("RA_OPENAI_BASE_URL=http://127.0.0.1:8001/v1")
    print("RA_OPENAI_API_KEY=local-vllm")
    print("RA_AGENT_ROUTES=" + json.dumps(routes, separators=(",", ":")))


if __name__ == "__main__":
    main()
