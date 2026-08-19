from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path("data/MachinePlan-10K")

def part_dirs(root: Path = ROOT) -> list[Path]: return sorted(p for p in root.iterdir() if p.is_dir())

def step_file(part: Path) -> Path: return part / f"{part.name}.stp"

def operations_file(part: Path) -> Path: return part / f"{part.name}_operations.json"

def load_operations(part: Path) -> list[dict[str, Any]]:
    return json.loads(operations_file(part).read_text())["operations"]
