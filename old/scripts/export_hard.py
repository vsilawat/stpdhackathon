"""Write the Hard-track tool submission: tool type and diameter per operation.

Format fixed by reference/validate_submission.py:

    <part_id>_tools.json
    {"part_id": ..., "summary": {"number_of_operations": N},
     "operations": [{"operation_number": 1, "tool_type": ...,
                     "tool_diameter_mm": 20.0}, ...]}

Our model predicts NX catalogue tool IDs. The ID encodes the tool class
directly -- `NXT0307_003` is type 03, subtype 07 = gun drill -- and this was
verified against the catalogue descriptions for all 431 tools with zero
disagreements.
"""
import csv, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline import ROOT, DER, load_model, predict

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(DER, "..", "submission", "hard")
VOCAB = os.path.join(DER, "..", "reference", "vocabularies.json")

# NX tool library (Type, SubType) code -> the organizers' tool_type vocabulary
CODE = {"0201": "end_mill", "0205": "chamfer_mill", "0301": "twist_drill",
        "0302": "insert_drill", "0306": "spade_drill", "0307": "gun_drill",
        "0321": "spot_drill", "0332": "boring_tool"}
RE_TOOL = re.compile(r"^(?:NX|UG)T(\d{4})_")


def tool_tables():
    dia, cat = {}, {}
    for t in csv.DictReader(open(os.path.join(DER, "tools.csv"))):
        if t["diameter"]:
            dia[t["tool"]] = float(t["diameter"])
        m = RE_TOOL.match(t["tool"])
        if m:
            cat[t["tool"]] = CODE.get(m.group(1))
    return dia, cat


def main():
    vocab = set(json.load(open(VOCAB))["tool_type"])
    dia, cat = tool_tables()
    model = load_model()
    ids = sorted(d for d in os.listdir(ROOT) if d.startswith("featured_part_"))
    os.makedirs(OUT, exist_ok=True)

    # fallbacks so we never emit an invalid record
    med_dia = sorted(dia.values())[len(dia) // 2]

    ok = skipped = bad_cat = bad_dia = 0
    for i, pid in enumerate(ids, 1):
        f = os.path.join(ROOT, pid, pid + ".stp")
        if not os.path.exists(f):
            skipped += 1
            continue
        try:
            plan = predict(pid, model)
        except Exception:
            skipped += 1
            continue

        operations = []
        for op in plan:
            t = op["tool"]
            tt = cat.get(t)
            if tt not in vocab:
                tt = "end_mill"
                bad_cat += 1
            d = dia.get(t)
            if not d or d <= 0:
                d = med_dia
                bad_dia += 1
            operations.append({"operation_number": len(operations) + 1,
                               "tool_type": tt,
                               "tool_diameter_mm": round(float(d), 4)})

        doc = {"part_id": pid,
               "summary": {"number_of_operations": len(operations)},
               "operations": operations}
        json.dump(doc, open(os.path.join(OUT, f"{pid}_tools.json"), "w"),
                  indent=2)
        ok += 1
        if i % 2500 == 0:
            print(f"  {i:,}/{len(ids):,}", flush=True)

    print(f"wrote {ok:,} tool files to {OUT}")
    if skipped:
        print(f"  skipped {skipped}")
    if bad_cat or bad_dia:
        print(f"  fallbacks used: {bad_cat} tool_type, {bad_dia} diameter")


if __name__ == "__main__":
    main()
