"""Rule-based CAM process-plan baseline for MachinePlan-10K.

Design principle: reproduce the parts of the NX pipeline we have *confirmed*
deterministic, and fall back on learned priors only where geometry extraction is
still unsolved.

  confirmed exact
    - chamfer features  == non-axis-aligned planar faces (1500/1500 parts)
    - every chamfer      -> one AREA_MILL with chamfer mill UGT0205_001
    - holes              == closed (360 deg) cylindrical faces
    - hole diameter      == the diameter of the tool that made it (0.000 mm)
    - operation name     -> tool class (Type, SubType), 100%

  recovered by alignment
    - each drilling/cylinder-milling step is matched to the hole it cuts via
      the X/Y positions in its G-code, giving true per-hole operation chains
    - pockets/slots/notches are recovered by clustering endmill corner fillets
      (same radius, height and depth); the fillet radius is the endmill radius

  learned prior
    - (diameter, through/blind, depth band) -> operation chain, with back-off

Usage
    python3 scripts/baseline.py fit
    python3 scripts/baseline.py predict <part_id>
"""
import collections, csv, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_parse import parse

ROOT = "/Users/vasusilawat/Desktop/stpd/data/MachinePlan-10K"
DER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "derived")
MODEL = os.path.join(DER, "baseline_model.json")

HOLE_FAMS = {"STEP1HOLE", "HOLE_FREE_SHAPED_STRAIGHT"}
CHAMFER_FAMS = {"FG_CHAMFER_SURFACE"}

# Operation kinds that target a hole rather than an area.
HOLE_OP_TYPES = {"Drilling", "Cylinder Milling"}

# NX order-group priority, used to sequence within a method block.
ORDER_PRI = {"CENTER": 0, "DRILL": 1, "DRILL_1": 2, "DRILL_2": 3,
             "BORE": 4, "BORE_1": 5,
             "MILL_ROUGH": 0, "MILL_SEMI_FINISH": 1, "MILL_FINISH": 2}

strip = lambda s: re.sub(r"_\d+$", "", s or "")


def load_ops():
    ops = list(csv.DictReader(open(os.path.join(DER, "op_details.csv"))))
    tools = {t["tool"]: t for t in
             csv.DictReader(open(os.path.join(DER, "tools.csv")))}
    by_part = collections.defaultdict(list)
    for o in ops:
        by_part[o["part_id"]].append(o)
    for v in by_part.values():
        v.sort(key=lambda r: int(r["seq"]))
    return by_part, tools


def dia(tools, t):
    d = tools.get(t, {}).get("diameter")
    return float(d) if d else None


def chains_of(part_ops):
    """Split a part's operations into per-feature chains, keyed by geometry group."""
    g = collections.OrderedDict()
    for o in part_ops:
        g.setdefault(o["geometry_group"], []).append(o)
    return g


RE_XY = re.compile(r"X(-?[\d.]+)\s+Y(-?[\d.]+)")


def part_holes(p):
    """Holes from the B-rep: centre, diameter, depth, through/blind."""
    out = []
    for c in p.hole_cylinders():
        ax = c.get("axis") or (0, 0, 1)
        org = c.get("origin")
        if not org:
            continue
        i = max(range(3), key=lambda k: abs(ax[k]))
        xy = tuple(org[k] for k in range(3) if k != i)
        through = False
        if c.get("depth") is not None and p.bbox:
            extent = p.bbox[i][1] - p.bbox[i][0]
            through = c["depth"] >= 0.97 * extent
        out.append({"xy": xy, "dia": 2 * c["radius"],
                    "depth": c.get("depth"), "through": through, "axis": i})
    return out


def fillet_clusters(p):
    """Group endmill corner fillets into pocket/slot/notch features.

    A rectangular pocket leaves several identical partial cylinders -- same
    radius, same height, same depth -- one at each corner. Collapsing them on
    that signature recovers the feature, and the fillet radius *is* the radius
    of the endmill that cut it.
    """
    out = {}
    for c in p.fillet_cylinders():
        ax = c.get("axis") or (0, 0, 1)
        org = c.get("origin") or (0, 0, 0)
        i = max(range(3), key=lambda k: abs(ax[k]))
        key = (round(c["radius"], 2), round(org[i], 1),
               round(c.get("depth") or 0, 1))
        out.setdefault(key, {"radius": c["radius"], "depth": c.get("depth"),
                             "n": 0})["n"] += 1
    return list(out.values())


def milled_features(p, tol=1e-3):
    """All pocket / slot / notch features on a part.

    Two detectors, because neither alone is complete:

      1. corner-fillet clusters -- catches enclosed pockets and notches, and
         additionally reveals the endmill radius;
      2. floor faces -- a horizontal face that is not the outermost one on its
         side is the bottom of *something* cut into the part. This catches
         through-slots, which are open at both ends and so leave no fillets.

    Blind holes also have flat bottoms, so one floor is discounted per blind
    hole. Floors already explained by a fillet cluster are discounted too, so
    the two detectors do not double-count.
    """
    clusters = fillet_clusters(p)

    up, dn = [], []
    for f in p.faces:
        if f["kind"] != "plane" or not f.get("axis") or not f.get("origin"):
            continue
        ax = f["axis"]
        if abs(abs(ax[2]) - 1.0) > 1e-6:      # horizontal faces only
            continue
        (up if ax[2] > 0 else dn).append(f["origin"][2])

    # express each floor as a depth below the surface it was cut from
    depths = []
    if up:
        top = max(up)
        depths += [top - z for z in up if z < top - tol]
    if dn:
        bot = min(dn)
        depths += [z - bot for z in dn if z > bot + tol]

    def consume(target):
        if target is None or not depths:
            return
        j = min(range(len(depths)), key=lambda k: abs(depths[k] - target))
        depths.pop(j)

    for h in part_holes(p):
        if not h["through"]:
            consume(h["depth"])
    for c in clusters:
        consume(c["depth"])

    feats = [{"radius": c["radius"], "depth": c["depth"], "n": c["n"]}
             for c in clusters]
    feats += [{"radius": None, "depth": d, "n": 0} for d in depths]
    return feats


def op_positions(part_id, seq, tool):
    """X/Y positions touched by one operation's toolpath (from its G-code)."""
    f = os.path.join(ROOT, part_id, f"{seq:03d}_{tool}.ptp")
    if not os.path.exists(f):
        return []
    try:
        txt = open(f, errors="replace").read()
    except OSError:
        return []
    return [(float(m.group(1)), float(m.group(2))) for m in RE_XY.finditer(txt)]


def depth_bucket(d):
    return None if d is None else int(d // 10) * 10


def split_ids(all_ids, val_frac=0.2):
    """Deterministic split -- hash-free, just every 5th part to validation."""
    ids = sorted(all_ids)
    val = set(ids[::5]) if val_frac else set()
    return [i for i in ids if i not in val], sorted(val)


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------
def _merge(keyed):
    """Merge a keyed lookup down to a single Counter."""
    out = collections.Counter()
    for c in keyed.values():
        out.update(c)
    return out


def _regroup(keyed, idx):
    """Re-key a lookup on a subset of its key components."""
    out = collections.defaultdict(collections.Counter)
    for k, c in keyed.items():
        out[tuple(k[i] for i in idx)].update(c)
    return out


def fit():
    by_part, tools = load_ops()
    train, val = split_ids(by_part)
    print(f"fitting on {len(train):,} parts (holding out {len(val):,})")

    hole_chain = collections.defaultdict(collections.Counter)
    chamfer_chain = collections.Counter()
    pocket_chain = collections.defaultdict(collections.Counter)
    slot_chain = collections.defaultdict(collections.Counter)
    # (operation, feature diameter) -> tool. Tool choice is a deterministic
    # function of the feature's dimensions in NX, so key it on the feature
    # rather than inheriting whichever tool the matched example happened to use.
    op_tool = collections.defaultdict(collections.Counter)
    op_tool_any = collections.defaultdict(collections.Counter)
    op_type = {}      # dataset JSON vocabulary (HoleDrilling, SurfaceContour, ...)
    is_drill = {}     # operation-card vocabulary, for method-block ordering
    op_order = {}
    op_time = collections.defaultdict(list)
    block_order = collections.Counter()

    times = {}
    for r in csv.DictReader(open(os.path.join(DER, "operations.csv"))):
        times[(r["part_id"], int(r["seq"]))] = float(r["time_min"] or 0)
        op_type[r["base_name"]] = r["type"]

    for pid in train:
        pops = by_part[pid]
        for o in pops:
            b = strip(o["op_name"])
            is_drill[b] = (o["op_type"] == "Drilling")
            op_tool_any[b][o["tool"]] += 1
            op_order[b] = o["order_group"]
            # op_details `seq` is the 1-based file index; operations.csv `seq`
            # is the 0-based JSON sequence_number.
            t = times.get((pid, int(o["seq"]) - 1))
            if t is not None:
                op_time[b].append(t)
        # method-block order
        blocks = [k for k, _ in __import__("itertools").groupby(
            o["method_group"] for o in pops)]
        block_order[tuple(blocks)] += 1

        for grp, gops in chains_of(pops).items():
            fam = strip(grp)
            if fam in CHAMFER_FAMS:
                chamfer_chain[tuple((strip(o["op_name"]), o["tool"])
                                    for o in gops)] += 1
                continue
        # --- drilling: align each operation to the hole it actually cuts ----
        # The G-code carries the X/Y of every drilled position, so operations
        # can be matched to a specific hole in the B-rep instead of guessed
        # from diameter alone.
        f = os.path.join(ROOT, pid, pid + ".stp")
        if not os.path.exists(f):
            continue
        try:
            p_geom = parse(f)
        except Exception:
            continue
        holes = part_holes(p_geom)
        per_hole = collections.defaultdict(list)
        for o in pops if holes else []:
            # Cylinder Milling also targets holes -- a hole can be milled
            # rather than drilled -- so both kinds align to the same features.
            if o["op_type"] not in HOLE_OP_TYPES:
                continue
            for x, y in set(op_positions(pid, int(o["seq"]), o["tool"])):
                j = min(range(len(holes)),
                        key=lambda k: (holes[k]["xy"][0] - x) ** 2 +
                                      (holes[k]["xy"][1] - y) ** 2)
                h = holes[j]
                if (h["xy"][0] - x) ** 2 + (h["xy"][1] - y) ** 2 <= 1.0:
                    per_hole[j].append((int(o["seq"]),
                                        strip(o["op_name"]), o["tool"]))
        for j, chain in per_hole.items():
            chain.sort()
            h = holes[j]
            ch = tuple((n, t) for _, n, t in chain)
            k_dia = round(h["dia"] * 2) / 2
            hole_chain[(k_dia, h["through"], depth_bucket(h["depth"]))][ch] += 1
            for _s, _n, _t in chain:
                op_tool[(_n, k_dia)][_t] += 1

        # --- pockets/slots/notches: match each milled feature to the corner
        # fillet cluster it left behind, keyed by the endmill radius ---------
        feats = milled_features(p_geom)
        with_r = [f for f in feats if f["radius"] is not None]
        without_r = [f for f in feats if f["radius"] is None]
        for grp, gops in chains_of(pops).items():
            vb = [o for o in gops
                  if o["op_type"] == "Volume Based 2.5D Milling"]
            if not vb:
                continue
            ds = [dia(tools, o["tool"]) for o in vb]
            ds = [d for d in ds if d]
            if not ds:
                continue
            td = max(ds)
            chain = tuple((strip(o["op_name"]), o["tool"]) for o in vb)
            # Prefer a fillet-bearing feature whose endmill radius matches this
            # operation's tool; otherwise this is a floor-only feature (a
            # through-slot), which is learned by depth alone.
            best = (min(with_r, key=lambda c: abs(2 * c["radius"] - td))
                    if with_r else None)
            if best is not None and abs(2 * best["radius"] - td) <= 1.0:
                with_r.remove(best)
                _kd = round(2 * best["radius"] * 2) / 2
                pocket_chain[(best["n"], _kd,
                              depth_bucket(best["depth"]))][chain] += 1
                for _n, _t in chain:
                    op_tool[(_n, _kd)][_t] += 1
            elif without_r:
                f_ = without_r.pop(0)
                slot_chain[depth_bucket(f_["depth"])][chain] += 1
            elif best is not None:
                with_r.remove(best)
                _kd = round(2 * best["radius"] * 2) / 2
                pocket_chain[(best["n"], _kd,
                              depth_bucket(best["depth"]))][chain] += 1
                for _n, _t in chain:
                    op_tool[(_n, _kd)][_t] += 1
            else:
                slot_chain[None][chain] += 1

    # Three lookup levels, most specific first; predict() backs off when a
    # part presents a hole we never saw at that depth/diameter.
    l3, l2, l1 = (collections.defaultdict(collections.Counter) for _ in range(3))
    for (d, th, db), counter in hole_chain.items():
        for ch, n in counter.items():
            l3[(d, th, db)][ch] += n
            l2[(d, th)][ch] += n
            l1[d][ch] += n

    model = {
        "hole_l3": {f"{d}|{int(th)}|{db}": list(v.most_common(1)[0][0])
                    for (d, th, db), v in l3.items()},
        "hole_l2": {f"{d}|{int(th)}": list(v.most_common(1)[0][0])
                    for (d, th), v in l2.items()},
        "hole_l1": {f"{d}": list(v.most_common(1)[0][0])
                    for d, v in l1.items()},
        "chamfer_chain": list(chamfer_chain.most_common(1)[0][0]),
        "pocket_l3": {f"{n}|{d}|{db}": list(v.most_common(1)[0][0])
                      for (n, d, db), v in pocket_chain.items()},
        "pocket_l2": {f"{n}|{d}": list(v.most_common(1)[0][0])
                      for (n, d), v in _regroup(pocket_chain, (0, 1)).items()},
        "pocket_l1": {f"{n}": list(v.most_common(1)[0][0])
                      for (n,), v in _regroup(pocket_chain, (0,)).items()},
        "pocket_l0": (list(_merge(pocket_chain).most_common(1)[0][0])
                      if pocket_chain else []),
        "op_tool": {f"{n}|{d}": v.most_common(1)[0][0]
                    for (n, d), v in op_tool.items()},
        "op_tool_any": {n: v.most_common(1)[0][0]
                        for n, v in op_tool_any.items()},
        "slot_l1": {str(db): list(v.most_common(1)[0][0])
                    for db, v in slot_chain.items()},
        "slot_l0": (list(_merge(slot_chain).most_common(1)[0][0])
                    if slot_chain else []),
        "op_type": op_type,
        "is_drill": is_drill,
        "op_order": op_order,
        "op_time": {k: sum(v) / len(v) for k, v in op_time.items()},
        "val_ids": val,
        "train_ids": train,
    }
    json.dump(model, open(MODEL, "w"), indent=1)
    print(f"  hole keys (dia,through,depth) : {len(model['hole_l3'])}")
    print(f"  hole keys (dia,through)       : {len(model['hole_l2'])}")
    print(f"  hole keys (dia)               : {len(model['hole_l1'])}")
    print(f"  modal chamfer chain           : {model['chamfer_chain']}")
    print(f"  pocket keys (corners,dia,dep) : {len(model['pocket_l3'])}")
    print(f"  pocket keys (corners,dia)     : {len(model['pocket_l2'])}")
    print(f"  pocket keys (corners)         : {len(model['pocket_l1'])}")
    print(f"  slot keys (depth)             : {len(model['slot_l1'])}")
    print(f"  modal slot chain              : {model['slot_l0']}")
    print(f"  most common method-block order: "
          f"{block_order.most_common(3)}")
    print(f"wrote {MODEL}")
    return model


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------
def load_model():
    m = json.load(open(MODEL))
    m["hole_l3"] = {(float(a), bool(int(b)), None if c == "None" else int(c)):
                    [tuple(x) for x in v]
                    for k, v in m["hole_l3"].items()
                    for a, b, c in [k.split("|")]}
    m["hole_l2"] = {(float(a), bool(int(b))): [tuple(x) for x in v]
                    for k, v in m["hole_l2"].items()
                    for a, b in [k.split("|")]}
    m["hole_l1"] = {float(k): [tuple(x) for x in v]
                    for k, v in m["hole_l1"].items()}
    m["chamfer_chain"] = [tuple(x) for x in m["chamfer_chain"]]
    m["pocket_l3"] = {(int(a), float(b), None if c == "None" else int(c)):
                      [tuple(x) for x in v]
                      for k, v in m["pocket_l3"].items()
                      for a, b, c in [k.split("|")]}
    m["pocket_l2"] = {(int(a), float(b)): [tuple(x) for x in v]
                      for k, v in m["pocket_l2"].items()
                      for a, b in [k.split("|")]}
    m["pocket_l1"] = {int(k): [tuple(x) for x in v]
                      for k, v in m["pocket_l1"].items()}
    m["pocket_l0"] = [tuple(x) for x in m["pocket_l0"]]
    m["slot_l1"] = {(None if k == "None" else int(k)): [tuple(x) for x in v]
                    for k, v in m["slot_l1"].items()}
    m["slot_l0"] = [tuple(x) for x in m["slot_l0"]]
    m["op_tool"] = {(a, float(b)): v for k, v in m["op_tool"].items()
                    for a, b in [k.rsplit("|", 1)]}
    bo = os.path.join(DER, "block_order.json")
    m["block_order"] = json.load(open(bo)) if os.path.exists(bo) else None
    return m


def resolve_tool(model, name, feat_dia, fallback):
    """Pick the tool for an operation from the feature's size.

    NX selects tools by querying its library on the feature's dimensions, so a
    tool memorised from a differently-sized example is wrong even when the
    operation is right. Re-derive it from this feature's actual diameter.
    """
    if feat_dia is not None:
        t = model["op_tool"].get((name, round(feat_dia * 2) / 2))
        if t:
            return t
        cands = [k for k in model["op_tool"] if k[0] == name]
        if cands:
            return model["op_tool"][
                min(cands, key=lambda k: abs(k[1] - feat_dia))]
    return model["op_tool_any"].get(name, fallback)


def predict(part_id, model=None, step_path=None):
    """Return an ordered list of {name, type, tool} for one part."""
    model = model or load_model()
    step_path = step_path or os.path.join(ROOT, part_id, part_id + ".stp")
    p = parse(step_path)

    plan = []

    # --- holes: one chain per closed cylinder (confirmed = one hole) --------
    # Look up by (diameter, through/blind, depth band), backing off to less
    # specific keys when that exact combination was never seen in training.
    l3, l2, l1 = model["hole_l3"], model["hole_l2"], model["hole_l1"]
    for h in part_holes(p):
        d, th, db = h["dia"], h["through"], depth_bucket(h["depth"])
        chain = l3.get((round(d * 2) / 2, th, db))
        exact = chain is not None
        if chain is None:
            cands = [k for k in l3 if k[1] == th and k[2] == db]
            if cands:
                chain = l3[min(cands, key=lambda k: abs(k[0] - d))]
        if chain is None:
            cands = [k for k in l2 if k[1] == th]
            if cands:
                chain = l2[min(cands, key=lambda k: abs(k[0] - d))]
        if chain is None and l1:
            chain = l1[min(l1, key=lambda k: abs(k - d))]
        plan += [(n, t, d, exact) for n, t in (chain or [])]

    # --- chamfers: exact ---------------------------------------------------
    n_cham = sum(1 for f in p.faces
                 if f["kind"] == "plane" and f.get("axis_aligned") is False)
    plan += [(n, t, None, True) for n, t in model["chamfer_chain"]] * n_cham

    # --- pockets/slots/notches --------------------------------------------
    p3, p2, p1, p0 = (model["pocket_l3"], model["pocket_l2"],
                      model["pocket_l1"], model["pocket_l0"])
    s1, s0 = model["slot_l1"], model["slot_l0"]
    for f in milled_features(p):
        db = depth_bucket(f["depth"])
        if f["radius"] is None:
            # floor-only feature (through-slot): no endmill radius to key on
            chain = s1.get(db)
            if chain is None and s1:
                cands = [k for k in s1 if k is not None]
                if cands and db is not None:
                    chain = s1[min(cands, key=lambda k: abs(k - db))]
            plan += [(n, t, None, True) for n, t in (chain or s0 or [])]
            continue
        d, nc = 2 * f["radius"], f["n"]
        chain = p3.get((nc, round(d * 2) / 2, db))
        exact = chain is not None
        if chain is None:
            cands = [k for k in p3 if k[0] == nc and k[2] == db]
            if cands:
                chain = p3[min(cands, key=lambda k: abs(k[1] - d))]
        if chain is None:
            cands = [k for k in p2 if k[0] == nc]
            if cands:
                chain = p2[min(cands, key=lambda k: abs(k[1] - d))]
        if chain is None:
            chain = p1.get(nc)
        plan += [(n, t, d, exact) for n, t in (chain or p0 or [])]

    # --- sequence: one method block then the other, ordered by order group --
    # Which block comes first is close to a coin flip in the data (58/42), but
    # is predictable from geometry at ~79% -- worth up to 4 rubric points.
    drill_first = True
    if model.get("block_order"):
        from block_order import extract, predict_drill_first
        try:
            drill_first = predict_drill_first(
                model["block_order"], extract(p)) >= 0.5
        except Exception:
            drill_first = True

    def sort_key(i_op):
        i, (name, _tool, _d, _e) = i_op
        is_d = bool(model["is_drill"].get(name))
        meth = 0 if is_d == drill_first else 1
        return (meth, ORDER_PRI.get(model["op_order"].get(name, ""), 9), i)

    plan = [op for _, op in sorted(enumerate(plan), key=sort_key)]
    return [{"name": n, "type": model["op_type"].get(n),
             "tool": t if e else resolve_tool(model, n, d, t),
             "toolpath_time_min": model["op_time"].get(n)}
            for n, t, d, e in plan]


def to_dataset_format(part_id, plan):
    """Emit a plan in the same schema as the dataset's *_operations.json."""
    ops = []
    seen = collections.Counter()
    total = 0.0
    tools_seq = []
    for i, op in enumerate(plan):
        base = op["name"]
        n = seen[base]
        seen[base] += 1
        t = op.get("toolpath_time_min") or 0.0
        total += t
        tools_seq.append(op["tool"])
        ops.append({
            "sequence_number": i,
            "name": base if n == 0 else f"{base}_{n}",
            "type": op["type"],
            "tool_name": op["tool"],
            "toolpath_time_min": round(t, 6),
        })
    changes = sum(1 for a, b in zip(tools_seq, tools_seq[1:]) if a != b)
    return {
        "part_id": part_id,
        "machining_summary": {
            "total_toolpath_time_min": round(total, 4),
            "tool_changes": changes,
            "num_operations": len(ops),
        },
        "operations": ops,
    }


def submit(out_dir, ids=None):
    """Write predicted plans for every part, ready to package as a submission."""
    model = load_model()
    ids = ids or sorted(
        d for d in os.listdir(ROOT) if d.startswith("featured_part_"))
    os.makedirs(out_dir, exist_ok=True)
    ok = bad = 0
    for i, pid in enumerate(ids, 1):
        f = os.path.join(ROOT, pid, pid + ".stp")
        if not os.path.exists(f):
            bad += 1
            continue
        try:
            plan = predict(pid, model)
        except Exception:
            bad += 1
            continue
        json.dump(to_dataset_format(pid, plan),
                  open(os.path.join(out_dir, f"{pid}_operations.json"), "w"),
                  indent=1)
        ok += 1
        if i % 2000 == 0:
            print(f"  {i:,}/{len(ids):,}", flush=True)
    print(f"wrote {ok:,} plans to {out_dir} ({bad} skipped)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "fit":
        fit()
    elif cmd == "predict":
        for op in predict(sys.argv[2]):
            print(f"  {op['name']:48s} {op['type']:30s} {op['tool']}")
    elif cmd == "submit":
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
            DER, "predictions")
        submit(out)
