import math, re, sys
from collections import defaultdict

RE_ENT = re.compile(r"#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*)\)\s*$", re.S)


def split_args(s):
    out, depth, cur, instr = [], 0, [], False
    i = 0
    while i < len(s):
        c = s[i]
        if instr:
            cur.append(c)
            if c == "'":
                if i + 1 < len(s) and s[i + 1] == "'":
                    cur.append(s[i + 1]); i += 1
                else:
                    instr = False
        elif c == "'":
            instr = True; cur.append(c)
        elif c == "(":
            depth += 1; cur.append(c)
        elif c == ")":
            depth -= 1; cur.append(c)
        elif c == "," and depth == 0:
            out.append("".join(cur).strip()); cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        out.append("".join(cur).strip())
    return out


class Part:
    def __init__(self, pid):
        self.part_id = pid
        self.faces = []
        self.bbox = None

    @property
    def n_planar(self):
        return sum(1 for f in self.faces if f["kind"] == "plane")

    @property
    def n_cyl(self):
        return sum(1 for f in self.faces if f["kind"] == "cylinder")

    def cyl_radii(self):
        return sorted(f["radius"] for f in self.faces if f["kind"] == "cylinder")

    def hole_cylinders(self):
        return [f for f in self.faces
                if f["kind"] == "cylinder" and f.get("closed")]

    def fillet_cylinders(self):
        return [f for f in self.faces
                if f["kind"] == "cylinder" and not f.get("closed")]

    def plane_normals(self):
        return [f["axis"] for f in self.faces if f["kind"] == "plane"]


def _is_axis_aligned(v, tol=1e-6):
    return sum(1 for c in v if abs(abs(c) - 1.0) < tol) == 1 and \
           sum(1 for c in v if abs(c) < tol) == 2


def parse(path):
    txt = open(path, errors="replace").read()
    if "DATA;" in txt:
        txt = txt.split("DATA;", 1)[1]
    txt = txt.split("ENDSEC", 1)[0]
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)

    ent = {}
    for chunk in txt.split(";"):
        chunk = chunk.strip()
        if not chunk.startswith("#"):
            continue
        m = RE_ENT.match(chunk)
        if m:
            ent[int(m.group(1))] = (m.group(2), m.group(3))

    def ref(a):
        a = a.strip()
        return int(a[1:]) if a.startswith("#") else None

    def triple(eid):
        if eid is None or eid not in ent:
            return None
        _, args = ent[eid]
        a = split_args(args)
        nums = a[1].strip()[1:-1]
        try:
            return tuple(float(x) for x in nums.split(","))
        except ValueError:
            return None

    def placement(eid):
        if eid is None or eid not in ent:
            return None, None
        _, args = ent[eid]
        a = split_args(args)
        origin = triple(ref(a[1])) if len(a) > 1 else None
        axis = triple(ref(a[2])) if len(a) > 2 and a[2].strip() != "$" else None
        return origin, axis

    def loop_edges(loop_id):
        if loop_id is None or loop_id not in ent:
            return []
        typ, args = ent[loop_id]
        if typ != "EDGE_LOOP":
            return []
        out = []
        inner = split_args(args)[1].strip()
        for tok in split_args(inner[1:-1]):
            oe = ref(tok)
            if oe is None or oe not in ent or ent[oe][0] != "ORIENTED_EDGE":
                continue
            ec = ref(split_args(ent[oe][1])[3])
            if ec is None or ec not in ent or ent[ec][0] != "EDGE_CURVE":
                continue
            ea = split_args(ent[ec][1])
            v1, v2, crv = ref(ea[1]), ref(ea[2]), ref(ea[3])
            ctype = ent[crv][0] if crv in ent else "?"
            out.append((v1, v2, ctype, crv))
        return out

    def face_bounds(a):
        edges = []
        inner = a[1].strip()
        for tok in split_args(inner[1:-1]):
            b = ref(tok)
            if b is None or b not in ent:
                continue
            btyp, bargs = ent[b]
            if btyp in ("FACE_OUTER_BOUND", "FACE_BOUND"):
                edges += loop_edges(ref(split_args(bargs)[1]))
        return edges

    pid = path.split("/")[-1].replace(".stp", "")
    part = Part(pid)

    for eid, (typ, args) in ent.items():
        if typ != "ADVANCED_FACE":
            continue
        a = split_args(args)
        surf = ref(a[2]) if len(a) > 2 else None
        if surf is None or surf not in ent:
            continue
        stype, sargs = ent[surf]
        sa = split_args(sargs)
        if stype == "CYLINDRICAL_SURFACE":
            org, ax = placement(ref(sa[1]))
            edges = face_bounds(a)
            circles = [e for e in edges if e[2] == "CIRCLE"]
            n_full = sum(1 for v1, v2, _, _ in circles if v1 == v2)
            depth = None
            lo_hi = None
            if ax:
                i = max(range(3), key=lambda k: abs(ax[k]))
                cs = []
                for _, _, _, crv in circles:
                    if crv in ent and ent[crv][0] == "CIRCLE":
                        o2, _a2 = placement(ref(split_args(ent[crv][1])[1]))
                        if o2:
                            cs.append(o2[i])
                if len(cs) >= 2:
                    depth = max(cs) - min(cs)
                    lo_hi = (min(cs), max(cs))
            part.faces.append({"kind": "cylinder", "radius": float(sa[2]),
                               "origin": org, "axis": ax, "id": eid,
                               "n_circle_edges": len(circles),
                               "n_full_circles": n_full,
                               "depth": depth,
                               "extent": lo_hi,
                               "closed": n_full > 0})
        elif stype == "PLANE":
            org, ax = placement(ref(sa[1]))
            part.faces.append({"kind": "plane", "radius": None,
                               "origin": org, "axis": ax, "id": eid,
                               "axis_aligned": _is_axis_aligned(ax) if ax else None})
        else:
            part.faces.append({"kind": stype.lower(), "radius": None,
                               "origin": None, "axis": None, "id": eid})

    pts = [triple(e) for e, (t, _) in ent.items() if t == "CARTESIAN_POINT"]
    pts = [p for p in pts if p and len(p) == 3]
    if pts:
        part.bbox = tuple(
            (min(p[i] for p in pts), max(p[i] for p in pts)) for i in range(3))
    return part


if __name__ == "__main__":
    p = parse(sys.argv[1])
    print(f"{p.part_id}: {len(p.faces)} faces "
          f"({p.n_planar} planar, {p.n_cyl} cylindrical)")
    print(f"  bbox: {p.bbox}")
    print(f"  cylinder radii: {p.cyl_radii()}")
    na = [f for f in p.faces if f["kind"] == "plane" and f["axis_aligned"] is False]
    print(f"  non-axis-aligned planes (chamfer candidates): {len(na)}")
    for f in na[:5]:
        print(f"    axis={tuple(round(c,4) for c in f['axis'])}")
