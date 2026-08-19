from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

DATA = Path("data/MachinePlan-10K")
OUT_OPDETAILS = Path("derived/opdetails.csv")
OUT_TOOLLIB = Path("derived/tool_library.csv")

FNAME_RE = re.compile(r"^(\d{3})_(.+)_details\.txt$")
OBJNAME_RE = re.compile(r"Object name:\s*(\S+)")
TEMPLATE_TYPE_RE = re.compile(r"^Template Type:\s*(.+?)\s*$", re.MULTILINE)
TEMPLATE_SUBTYPE_RE = re.compile(r"^Template Subtype:\s*(.+?)\s*$", re.MULTILINE)
GEOMETRY_GROUP_RE = re.compile(r"^Geometry Group\s+(\S+)", re.MULTILINE)
TOOL_TYPE_RE = re.compile(r"^Tool Type\s*:\s*(.+?)\s*$", re.MULTILINE)
DIAMETER_RE = re.compile(r"^\(D\)\s*Diameter\s*=\s*([\d.]+)\s*mm", re.MULTILINE)
LOWER_RADIUS_RE = re.compile(r"^\(R1\)\s*Lower Radius\s*=\s*([\d.]+)\s*mm", re.MULTILINE)
FLUTE_LENGTH_RE = re.compile(r"^\(FL\)\s*(?:Tool )?Flute Length\s*=\s*([\d.]+)\s*mm", re.MULTILINE)
GEO_SUFFIX_RE = re.compile(r"_\d+$")

def find_detail_files() -> list[Path]:
    return sorted(f for part_dir in DATA.glob("featured_part_*") for f in part_dir.glob("*_details.txt") if f.name != "workpiece_details.txt")

def parse_one(path: Path) -> dict | None:
    fname_match = FNAME_RE.match(path.name)
    if not fname_match: return None
    op_index, tool_name = int(fname_match.group(1)), fname_match.group(2)
    part = path.parent.name
    try: text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError: return {"part": part, "op_index": op_index, "tool_name": tool_name, "_error": "read_failed"}
    objnames = OBJNAME_RE.findall(text)
    tsubtypes = TEMPLATE_SUBTYPE_RE.findall(text)
    op_name = objnames[0] if objnames else ""
    tool_template_subtype = tsubtypes[1].strip() if len(tsubtypes) >= 2 else ""
    ttype_m, dia_m, r1_m, fl_m, geo_m, tt_m = (
        TEMPLATE_TYPE_RE.search(text), DIAMETER_RE.search(text), LOWER_RADIUS_RE.search(text),
        FLUTE_LENGTH_RE.search(text), GEOMETRY_GROUP_RE.search(text), TOOL_TYPE_RE.search(text))
    return {
        "part": part, "op_index": op_index, "tool_name": tool_name, "op_name": op_name,
        "template_type": ttype_m.group(1).strip() if ttype_m else "",
        "template_subtype": tsubtypes[0].strip() if tsubtypes else "",
        "geometry_group": geo_m.group(1) if geo_m else "",
        "tool_type": tt_m.group(1).strip() if tt_m else "",
        "tool_diameter_mm": dia_m.group(1) if dia_m else "",
        "lower_radius_mm": r1_m.group(1) if r1_m else "",
        "flute_length_mm": fl_m.group(1) if fl_m else "",
        "tool_template_subtype": tool_template_subtype,
    }

FIELDS = ["part", "op_index", "tool_name", "op_name", "template_type", "template_subtype", "geometry_group",
          "tool_type", "tool_diameter_mm", "lower_radius_mm", "flute_length_mm", "tool_template_subtype"]

def write_opdetails(rows: list[dict]) -> None:
    OUT_OPDETAILS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_OPDETAILS.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["part"], r["op_index"])): w.writerow(r)

def build_tool_library(rows: list[dict]) -> tuple[list[dict], list[tuple]]:
    by_tool: dict[str, dict[str, Counter]] = {}
    for r in rows:
        tn = r["tool_name"]
        slot = by_tool.setdefault(tn, {"tool_type": Counter(), "tool_diameter_mm": Counter(), "lower_radius_mm": Counter()})
        for f in ("tool_type", "tool_diameter_mm", "lower_radius_mm"):
            v = r.get(f, "")
            if v: slot[f][v] += 1
    conflicts = []
    lib_rows = []
    for tn, slot in sorted(by_tool.items()):
        vals = {}
        for f in ("tool_type", "tool_diameter_mm", "lower_radius_mm"):
            if len(slot[f]) > 1: conflicts.append((tn, f, dict(slot[f])))
            vals[f] = slot[f].most_common(1)[0][0] if slot[f] else ""
        lib_rows.append({"tool_name": tn, **vals})
    return lib_rows, conflicts

def write_tool_library(lib_rows: list[dict]) -> None:
    with OUT_TOOLLIB.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tool_name", "tool_type", "tool_diameter_mm", "lower_radius_mm"])
        w.writeheader()
        for r in lib_rows: w.writerow(r)

def geo_prefix(g: str) -> str: return GEO_SUFFIX_RE.sub("", g) if g else g

def verify_alignment(sample_size: int = 50) -> list[str]:
    problems = []
    part_dirs = sorted(DATA.glob("featured_part_*"))
    random.seed(0)
    sample = random.sample(part_dirs, min(sample_size, len(part_dirs)))
    for part_dir in sample:
        opsfile = part_dir / f"{part_dir.name}_operations.json"
        if not opsfile.exists(): problems.append(f"{part_dir.name}: missing operations.json"); continue
        ops = sorted(json.loads(opsfile.read_text())["operations"], key=lambda o: o["sequence_number"])
        detail_files = {int(FNAME_RE.match(f.name).group(1)): FNAME_RE.match(f.name).group(2)
                         for f in part_dir.glob("*_details.txt") if f.name != "workpiece_details.txt"}
        for op in ops:
            expect_idx = op["sequence_number"] + 1
            if expect_idx not in detail_files: problems.append(f"{part_dir.name}: seq {op['sequence_number']} -> op_index {expect_idx} missing details file"); continue
            if detail_files[expect_idx] != op.get("tool_name"):
                problems.append(f"{part_dir.name}: op_index {expect_idx} tool_name mismatch: file={detail_files[expect_idx]!r} json={op.get('tool_name')!r}")
    return problems

def main() -> None:
    files = find_detail_files()
    print(f"Found {len(files)} operation detail files across {len({f.parent.name for f in files})} parts.")
    with Pool() as pool: results = pool.map(parse_one, files, chunksize=200)
    rows = [r for r in results if r is not None]
    failures = [r for r in rows if "_error" in r]
    rows = [r for r in rows if "_error" not in r]
    print(f"Parsed ops: {len(rows)}  Parts represented: {len({r['part'] for r in rows})}  Parse failures: {len(failures)}")
    if failures: print("Sample failures:", failures[:5])
    write_opdetails(rows)
    lib_rows, conflicts = build_tool_library(rows)
    write_tool_library(lib_rows)
    print(f"\nWrote {OUT_OPDETAILS} ({len(rows)} rows) and {OUT_TOOLLIB} ({len(lib_rows)} unique tools).")
    print(f"Tool name conflicts (tool_name mapping to >1 distinct value for a field): {len(conflicts)}")
    for tn, f, vals in conflicts[:20]: print(f"  {tn} field={f} values={vals}")

    tt_counts = Counter(r["template_type"] for r in rows)
    ts_counts = Counter(r["template_subtype"] for r in rows)
    print(f"\n=== Distinct template_type ({len(tt_counts)}) ===")
    for k, v in tt_counts.most_common(): print(f"  {k!r:20s} {v}")
    print(f"\n=== Distinct template_subtype ({len(ts_counts)}) ===")
    for k, v in ts_counts.most_common(): print(f"  {k!r:20s} {v}")

    print("\n=== Cross-tab template_type x template_subtype ===")
    cross = Counter((r["template_type"], r["template_subtype"]) for r in rows)
    for k, v in sorted(cross.items(), key=lambda x: -x[1]): print(f"  {k[0]!r:15s} x {k[1]!r:25s} {v}")

    tool_type_counts = Counter(r["tool_type"] for r in rows)
    print(f"\n=== Distinct Tool Type strings ({len(tool_type_counts)}) ===")
    for k, v in tool_type_counts.most_common(): print(f"  {k!r:30s} {v}")

    print("\n=== Tool Type x tool_template_subtype prefix (for vocab mapping) ===")

    def subtype_prefix(s: str) -> str: return re.sub(r"[\d.]+.*$", "", s).strip()
    tt_ts_cross = Counter((r["tool_type"], subtype_prefix(r["tool_template_subtype"])) for r in rows)
    for k, v in sorted(tt_ts_cross.items(), key=lambda x: -x[1]): print(f"  {k[0]!r:30s} x {k[1]!r:30s} {v}")

    geo_counts = Counter(geo_prefix(r["geometry_group"]) for r in rows)
    print(f"\n=== Distinct geometry_group prefixes ({len(geo_counts)}) ===")
    for k, v in geo_counts.most_common(): print(f"  {k!r:35s} {v}")

    print("\n=== Cross-tab template_subtype x Tool Type string ===")
    sub_tool_cross = Counter((r["template_subtype"], r["tool_type"]) for r in rows)
    for k, v in sorted(sub_tool_cross.items(), key=lambda x: -x[1]): print(f"  {k[0]!r:20s} x {k[1]!r:30s} {v}")

    print("\n=== Verifying op_index/tool_name alignment with {part}_operations.json (50 random parts) ===")
    problems = verify_alignment(50)
    if problems:
        print(f"Found {len(problems)} mismatches:")
        for p in problems[:30]: print(f"  {p}")
    else: print("No mismatches found: op_index == sequence_number+1 and tool_name match for all checked parts.")

if __name__ == "__main__": main()
