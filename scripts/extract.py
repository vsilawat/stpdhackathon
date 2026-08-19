import argparse, os, sys, time, zipfile

DEFAULT_ZIP = "/Users/vasusilawat/Desktop/stpd/data/MachinePlan-10K.zip"
DEFAULT_OUT = "/Users/vasusilawat/Desktop/stpd/data/MachinePlan-10K"


def wanted(name, keep_png):
    if name.endswith("/"):
        return False
    if name.endswith("_text.stl.txt"):
        return False
    if not keep_png and name.endswith(".png"):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=DEFAULT_ZIP)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--keep-png", action="store_true",
                    help="also extract the 1.3 GB of preview renders")
    ap.add_argument("--limit", type=int, default=0,
                    help="only extract the first N part folders (smoke test)")
    a = ap.parse_args()

    zf = zipfile.ZipFile(a.zip)
    names = [n for n in zf.namelist() if wanted(n, a.keep_png)]

    if a.limit:
        keep = sorted({n.split("/")[0] for n in names})[: a.limit]
        keep = set(keep)
        names = [n for n in names if n.split("/")[0] in keep]

    total = sum(zf.getinfo(n).file_size for n in names)
    print(f"extracting {len(names):,} members ({total/1e9:.2f} GB) -> {a.out}")

    os.makedirs(a.out, exist_ok=True)
    done = 0
    t0 = time.time()
    for i, n in enumerate(names, 1):
        zf.extract(n, a.out)
        done += zf.getinfo(n).file_size
        if i % 5000 == 0 or i == len(names):
            el = time.time() - t0
            print(f"  {i:7,}/{len(names):,}  {done/1e9:6.2f} GB  {el:6.0f}s",
                  flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
