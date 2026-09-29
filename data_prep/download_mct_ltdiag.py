"""
download_mct_ltdiag.py — download MCT-LTDiag from Harvard Dataverse.

  Dataset : doi:10.7910/DVN/S3RW15  (CC0 1.0, no restricted files)
  Size    : ~180 GB — 517 per-case .tar archives + 2 metadata tables

Standard library only, so it runs on an AICR data-transfer node
(dtn0001.aicr.ai) without a Python environment. Resumable: files whose MD5
already matches are skipped; partial downloads are retried.

Usage
  python download_mct_ltdiag.py --out /work/<inst>/<group>/herald/raw
  python download_mct_ltdiag.py --out ./raw --cases 230906d12 230218a1   # subset
"""

import argparse, hashlib, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DATAVERSE = "https://dataverse.harvard.edu"
DOI       = "doi:10.7910/DVN/S3RW15"
CHUNK     = 1 << 20
# Dataverse rejects urllib's default User-Agent with HTTP 403
HEADERS   = {"User-Agent": "herald-mct-ltdiag-downloader/1.0"}


def open_url(url, timeout):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS),
                                  timeout=timeout)


def list_files():
    url = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={DOI}"
    with open_url(url, timeout=60) as r:
        version = json.load(r)["data"]["latestVersion"]
    files = []
    for f in version["files"]:
        df = f["dataFile"]
        tabular = "originalFileName" in df or df["filename"].endswith(".tab")
        name = df.get("originalFileName", df["filename"])
        files.append({"id": df["id"], "name": name, "tabular": tabular,
                      "size": df.get("originalFileSize", df["filesize"]),
                      "md5": (df.get("checksum") or {}).get("value") or df.get("md5")})
    print(f"Dataset {DOI} v{version['versionNumber']}.{version['versionMinorNumber']}"
          f" — {len(files)} files, {sum(f['size'] for f in files) / 1e9:.1f} GB")
    return files


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(f, out_dir, retries=5):
    dst = os.path.join(out_dir, f["name"])
    # tabular files are served in their original format; their stored MD5 is
    # for the ingested .tab, so they are verified by size only
    verify_md5 = f["md5"] and not f["tabular"]
    if os.path.exists(dst) and os.path.getsize(dst) == f["size"] and \
            (not verify_md5 or md5sum(dst) == f["md5"]):
        return f["name"], "ok (cached)"
    url = f"{DATAVERSE}/api/access/datafile/{f['id']}"
    if f["tabular"]:
        url += "?format=original"
    for attempt in range(1, retries + 1):
        try:
            tmp = dst + ".part"
            with open_url(url, timeout=120) as r, open(tmp, "wb") as fh:
                for block in iter(lambda: r.read(CHUNK), b""):
                    fh.write(block)
            if verify_md5 and md5sum(tmp) != f["md5"]:
                raise IOError("MD5 mismatch")
            os.replace(tmp, dst)
            return f["name"], "downloaded"
        except Exception as e:                       # noqa: BLE001 — retry anything
            print(f"  {f['name']}: attempt {attempt} failed ({e})", flush=True)
            time.sleep(min(60, 5 * attempt))
    return f["name"], "FAILED"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Directory for the raw archives")
    p.add_argument("--cases", nargs="*", help="Only these case IDs (default: all)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = list_files()
    if args.cases:
        wanted = {f"{c}.tar" for c in args.cases}
        files = [f for f in files if f["name"] in wanted or not f["name"].endswith(".tar")]

    failed = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, f, args.out) for f in files]
        for i, fut in enumerate(as_completed(futs), 1):
            name, status = fut.result()
            print(f"[{i}/{len(files)}] {name}: {status}", flush=True)
            if status == "FAILED":
                failed.append(name)

    if failed:
        print(f"\n{len(failed)} file(s) failed — rerun to retry: {failed}")
        sys.exit(1)
    print(f"\nAll {len(files)} files present and verified in {args.out}")


if __name__ == "__main__":
    main()
