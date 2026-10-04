#!/usr/bin/env python
"""Dem so hop nguoi duyet da THEM / XOA / CHINH so voi nhan nhap, moi anh 1 dong.

Nguoi gan khong phai tu dem tay (guideline muc 9): so lieu nay lay tu viec so file nhan nhap
(prelabel.py --out) voi file export CVAT sau khi duyet.

Ghep hop nhap <-> hop duyet theo IoU (tham lam, IoU cao truoc):
  IoU >= --match  : cung 1 xe. Neu IoU < --same thi tinh la "chinh" (adjusted).
  hop duyet khong ghep duoc : "them" (added)
  hop nhap khong ghep duoc  : "xoa" (deleted)
IoU tinh tren toa do chuan hoa [0,1] van dung (co gian rieng tung truc khong doi IoU).

Vi du:
  python tools/review_stats.py --prelabels prelabels --reviewed reviewed/r1_cuong.zip reviewed/r1_long.zip \\
      --image-pool frames --out logs/review_stats.csv
"""
import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import collect, read_excludes  # noqa: E402


def read_boxes(txt):
    """YOLO cx cy w h -> list (x1, y1, x2, y2); file khong co = []."""
    if txt is None or not Path(txt).exists():
        return []
    out = []
    for ln in Path(txt).read_text(encoding="utf-8").splitlines():
        v = ln.split()
        if len(v) >= 5:
            cx, cy, w, h = map(float, v[1:5])
            out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def compare(pre, rev, match, same):
    pairs = sorted(((iou(p, r), i, j) for i, p in enumerate(pre) for j, r in enumerate(rev)), reverse=True)
    used_p, used_r, adjusted = set(), set(), 0
    for v, i, j in pairs:
        if v < match:
            break
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        adjusted += v < same
    return len(rev) - len(used_r), len(pre) - len(used_p), adjusted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prelabels", required=True, help="thu muc --out cua prelabel.py (<stem>.txt)")
    ap.add_argument("--reviewed", nargs="+", required=True, help="zip/thu muc export CVAT sau khi duyet")
    ap.add_argument("--image-pool", nargs="*", default=[], help="anh goc, khi export CVAT khong kem anh")
    ap.add_argument("--exclude-log", default="logs/labeling_log.csv")
    ap.add_argument("--match", type=float, default=0.5)
    ap.add_argument("--same", type=float, default=0.98, help="IoU >= nguong nay coi nhu giu nguyen")
    ap.add_argument("--out", default="logs/review_stats.csv")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="review_stats_"))
    try:
        items = collect(a.reviewed, a.image_pool, tmp)
        excl = read_excludes(a.exclude_log)
        pre_dir = Path(a.prelabels)
        rows, tot = [], [0, 0, 0, 0, 0]
        for stem in sorted(items):
            img, lab, src = items[stem]
            pre_txt = pre_dir / f"{stem}.txt"
            pre, rev = read_boxes(pre_txt), read_boxes(lab)
            added, deleted, adjusted = compare(pre, rev, a.match, a.same)
            action = "exclude" if stem in excl else "keep"
            rows.append([img.name, src, len(pre), len(rev), added, deleted, adjusted, action,
                         "" if pre_txt.exists() else "khong co nhan nhap"])
            if action == "keep":
                for k, n in enumerate((len(pre), len(rev), added, deleted, adjusted)):
                    tot[k] += n
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "source", "n_prelabel", "n_final", "n_added", "n_deleted", "n_adjusted", "action", "note"])
        w.writerows(rows)
    n_keep = sum(r[7] == "keep" for r in rows)
    print(f"{len(rows)} anh ({n_keep} keep, {len(rows) - n_keep} exclude) -> {out}")
    print(f"Tren anh keep: nhap {tot[0]} hop -> sau duyet {tot[1]} hop | them {tot[2]} | xoa {tot[3]} | chinh {tot[4]}")
    touched = sum(1 for r in rows if r[7] == "keep" and (r[4] or r[5] or r[6]))
    if n_keep:
        print(f"{touched}/{n_keep} anh ({100 * touched / n_keep:.0f}%) nguoi duyet co sua it nhat 1 hop")


if __name__ == "__main__":
    main()
