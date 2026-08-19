from __future__ import annotations

from .features import Features, Hole

FINAL_D = ("INDEXABLE_INSERT_DRILL_THROUGH_HOLE_FROM_SOLID", "INDEXABLE_INSERT_DRILL_BLIND_HOLE_FROM_SOLID",
           "BORE_BLIND_HOLE", "MILL_THROUGH_HOLE_FROM_SOLID_MATERIAL")
TOOL_TYPE = {"SPOT_DRILL": "spot_drill", "BORE_BLIND_HOLE": "boring_tool"}

_MODELS: dict | None = None
def models() -> dict:
    global _MODELS
    if _MODELS is None:
        import pickle
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "derived/tool_dia_models.pkl"
        _MODELS = pickle.load(p.open("rb")) if p.exists() else {}
    return _MODELS

def tool_type(name: str) -> str:
    if name in TOOL_TYPE: return TOOL_TYPE[name]
    if "GUN_DRILL" in name: return "gun_drill"
    if "SPADE" in name: return "spade_drill"
    if "INDEXABLE" in name: return "insert_drill"
    if name == "AREA_MILL": return "chamfer_mill"
    if name.startswith("MILL_"): return "end_mill"
    return "twist_drill"

def hole_tool_dia(name: str, h: Hole, found: Features, pos: int, chain_len: int) -> float:
    if name == "SPOT_DRILL": return 12.0
    if name in FINAL_D: return h.diameter
    kind = models().get(name)
    if not kind: return h.diameter
    if kind[0] == "const": return float(kind[1])
    sx, sy, _ = found.stock
    X = [[h.diameter, h.depth, h.depth / h.diameter, float(h.through), pos, chain_len,
          found.stock[2], found.top_z - h.mouth_z, h.bottom_z, h.diameter % 1.0,
          float(abs(h.diameter - round(h.diameter)) < 0.01), min(h.x, sx - h.x, h.y, sy - h.y)]]
    return float(kind[1].predict(X)[0])
