"""Write the Easy-track submission: ordered operation sequence per part.

Format is fixed by the organizers' validator (reference/validate_submission.py):

    <part_id>_sequence.json
    {"part_id": ..., "summary": {"number_of_operations": N},
     "operations": [{"operation_number": 1, "o1": ..., "o2": ...}, ...]}

`o1` / `o2` are coarse categories from reference/vocabularies.json. Our model
predicts NX's fine-grained operation names, which map onto them deterministically
(verified: all 29 names map to exactly one (o1, o2) pair).

Note the required keys are EXACT -- extra fields are a validation error, not
just missing ones.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline import ROOT, DER, load_model, predict

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(DER, "..", "submission", "easy")
MAP = os.path.join(DER, "o1o2_map.json")
VOCAB = os.path.join(DER, "..", "reference", "vocabularies.json")


def main():
    o1o2 = {k: tuple(v) for k, v in json.load(open(MAP)).items()}
    vocab = json.load(open(VOCAB))
    v1, v2 = set(vocab["o1"]), set(vocab["o2"])

    model = load_model()
    ids = sorted(d for d in os.listdir(ROOT) if d.startswith("featured_part_"))
    os.makedirs(OUT, exist_ok=True)

    ok = skipped = unmapped = 0
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
            pair = o1o2.get(op["name"])
            if pair is None or pair[0] not in v1 or pair[1] not in v2:
                # never seen -> fall back to the permitted catch-all rather
                # than emitting a value the validator would reject
                pair = ("OTHER", "OTHER")
                unmapped += 1
            operations.append({"operation_number": len(operations) + 1,
                               "o1": pair[0], "o2": pair[1]})

        doc = {"part_id": pid,
               "summary": {"number_of_operations": len(operations)},
               "operations": operations}
        json.dump(doc, open(os.path.join(OUT, f"{pid}_sequence.json"), "w"),
                  indent=2)
        ok += 1
        if i % 2500 == 0:
            print(f"  {i:,}/{len(ids):,}", flush=True)

    print(f"wrote {ok:,} sequence files to {OUT}")
    if skipped:
        print(f"  skipped {skipped}")
    if unmapped:
        print(f"  operations with no known (o1,o2): {unmapped} -> OTHER/OTHER")


if __name__ == "__main__":
    main()
