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

LIB = [3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20, 22, 24, 25, 28, 30, 32, 36, 40, 45, 50]
KINDS = ["corner", "center", "slot", "edge"]

_POCKET: object = None
def pocket_model():
    global _POCKET
    if _POCKET is None:
        import pickle
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "derived/pocket_tool_model.pkl"
        _POCKET = pickle.load(p.open("rb")) if p.exists() else False
    return _POCKET

def pocket_cap(pk) -> float:
    fr = pk.fillet_radius or 0.0
    return pk.w if fr == 0 else min(pk.w, 2 * fr)

def pocket_dia_feats(pk) -> list[float]:
    fr, w, l = pk.fillet_radius or 0.0, pk.w, pk.l
    return [KINDS.index(pk.kind) if pk.kind in KINDS else -1, fr, pk.depth, pk.area,
            float(pk.open_sides or 0), w, l, w / max(l, 0.1), pocket_cap(pk), pk.depth / max(w, 0.1),
            pk.area / max(w * l, 0.1), getattr(pk, "mi", 0.0), getattr(pk, "hull", 1.0)]

# sibling context: NX tool choice couples across a part's pockets
def pocket_dia_X(ps: list) -> list[list[float]]:
    pad = lambda v: (sorted(v) + [0.0] * 5)[:5]
    caps, mis, deps = [pocket_cap(p) for p in ps], [getattr(p, "mi", 0.0) for p in ps], [p.depth for p in ps]
    return [pocket_dia_feats(p) + [len(ps), sum(1 for v in caps if v < pocket_cap(p) - 1e-9)]
            + pad(caps) + pad(mis) + pad(deps) for p in ps]

def pocket_tool_dia(pk) -> float:
    fr = pk.fillet_radius or 5.0
    return float(max((t for t in LIB if t < 2 * fr - 1e-9), default=round(2 * fr)))

def pocket_dias(ps: list) -> list[float]:
    m = pocket_model()
    if not ps: return []
    if not m or any(p.w <= 0 for p in ps): return [pocket_tool_dia(p) for p in ps]
    X = pocket_dia_X(ps)
    if isinstance(m, dict):
        proba = m["rf"].predict_proba(X) + m["hgb"].predict_proba(X)
        cls = m["rf"].classes_.astype(float)
        return [m["mode"][cls[i]] for i in proba.argmax(1)]
    return [float(v) for v in m.predict(X)]

_PREBORE: object = None
def prebore_mill_dia(fd: float) -> float:
    global _PREBORE
    if _PREBORE is None:
        import pickle
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "derived/prebore_mill_dia.pkl"
        _PREBORE = pickle.load(p.open("rb")) if p.exists() else False
    if not _PREBORE: return fd
    return float(_PREBORE.predict([[fd, fd % 1.0]])[0])

_GP: dict | None = None
def gun_pilot(gun_dia: float, final_dia: float) -> float | None:
    global _GP
    if _GP is None:
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "derived/gun_pilot_lookup.json"
        _GP = json.loads(p.read_text()) if p.exists() else {}
    return _GP.get(f"{round(gun_dia, 2)}|{round(final_dia, 1)}")

_CLUT: dict | None = None
def chain_dias(names: list[str], final_d: float) -> list[float] | None:
    """Modal GT dia vector for a whole hole chain, keyed on (chain, final dia)."""
    global _CLUT
    if _CLUT is None:
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "derived/chain_dia_lookup.json"
        _CLUT = json.loads(p.read_text()) if p.exists() else {}
    v = _CLUT.get(f"{'>'.join(names)}|{round(final_d, 1)}")
    return list(v) if v and len(v) == len(names) else None

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
