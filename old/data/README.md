# data/ — the dataset (not in git)

This directory is intentionally empty in the repository. Its contents are
**11 GB** and exceed GitHub's limits, so they are excluded by `.gitignore`
(the exception being this README).

## What belongs here

```
data/
  MachinePlan-10K.zip          5.2 GB   the archive as downloaded
  MachinePlan-10K/             6.4 GB   extracted, 10,000 part folders
    featured_part_00001/
      featured_part_00001.stp                 STEP B-rep  (the model INPUT)
      featured_part_00001_operations.json     process plan (the LABEL)
      workpiece_details.txt                   material, offsets
      000_BLANK.stl                           starting stock
      001_<TOOL>.stl                          in-process workpiece after op 1
      001_<TOOL>.ptp                          toolpath for op 1 (ISO G-code)
      001_<TOOL>_details.txt                  operation card: tool + parameters
      *.png                                   five preview renders
    ...
```

Each part folder holds `10 + 4 x (number of operations)` files.

## How to obtain it

```bash
# 1. download (~5.2 GB; Zenodo is slow, budget ~90 minutes)
mkdir -p data && cd data
curl -L -C - --retry 5 -o MachinePlan-10K.zip \
  "https://zenodo.org/api/records/21653081/files/MachinePlan-10K.zip/content"

# 2. verify — this MD5 is published on the Zenodo record
md5 -q MachinePlan-10K.zip     # expect 831ccc4bd0ee62759ec383556b8c95da

# 3. extract (skips redundant ASCII-STL duplicates: 35 GB -> 6.4 GB)
cd .. && python3 scripts/extract.py
```

`scripts/extract.py` deliberately skips the `*_text.stl.txt` members. They are
plain-text copies of the binary `.stl` meshes and account for ~28 GB of the
35 GB uncompressed archive with no information gain.

Source: **MachinePlan-10K**, DOI [10.5281/zenodo.21653081](https://doi.org/10.5281/zenodo.21653081),
CC-BY-4.0. C. Dharmarajan, Arvanitis, Ameta (2026).

## Gotchas

- Part IDs run to `featured_part_11416` **with gaps**; there are exactly 10,000
  folders. Do not assume ID == index.
- All dimensions are millimetres.
- Zenodo ignores HTTP `Range` on `HEAD` but honours it on `GET`, which is how
  the archive index was inspected before the download finished.
