#!/usr/bin/env python
"""Dong goi anh + nhan NHAP de dua vao CVAT (dinh dang "YOLO 1.1").

Moi phan (part) tao 2 file:
  <out>_images.zip   anh phang (flat) -> dung khi "Create new task" (Select files)
  <out>.zip          nhan YOLO 1.1: obj.names (GreenSM), obj.data, train.txt, obj_train_data/<stem>.txt
                     -> trong task: Actions > Upload annotations > chon format "YOLO 1.1"
CVAT ghep nhan voi frame theo TEN FILE (frame_000001.jpg <-> frame_000001.txt), nen
khong doi ten anh giua 2 buoc.

--split-by-video K : chia anh thanh K phan theo video goc (video stem = phan truoc '_f<so>'),
                     can bang so anh, de K nguoi duyet song song -> <out>_part1.zip ...
--with-images      : nhet ca anh vao obj_train_data/ cua <out>.zip (dung cho "Import dataset").

Vi du:
  python tools/make_cvat_import.py --images frames --labels prelabels --out cvat_import_r1.zip --split-by-video 3
"""
import argparse
import csv
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["GreenSM"]


def video_of(stem):
    m = re.match(r"^(.*)_f\d+$", stem)
    return m.group(1) if m else stem


def write_part(zip_path, imgs, label_dir, with_images):
    n_box = n_missing = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(CLASSES) + "\n")
        z.writestr("obj.data", f"classes = {len(CLASSES)}\nnames = obj.names\ntrain = train.txt\nbackup = backup/\n")
        z.writestr("train.txt", "".join(f"obj_train_data/{p.name}\n" for p in imgs))
        for p in imgs:
            lab = label_dir / f"{p.stem}.txt" if label_dir else None
            if lab is not None and lab.is_file():
                text = lab.read_text(encoding="utf-8")
                n_box += sum(1 for ln in text.splitlines() if ln.strip())
            else:
                text, n_missing = "", n_missing + 1
            z.writestr(f"obj_train_data/{p.stem}.txt", text)
            if with_images:
                z.write(p, f"obj_train_data/{p.name}", compress_type=zipfile.ZIP_STORED)
    img_zip = zip_path.with_name(zip_path.stem + "_images.zip")
    with zipfile.ZipFile(img_zip, "w", zipfile.ZIP_STORED) as z:  # jpg da nen san
        for p in imgs:
            z.write(p, p.name)
    return n_box, n_missing, img_zip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="thu muc anh (quet de quy)")
    ap.add_argument("--labels", default=None, help="thu muc nhan nhap <stem>.txt (bo trong = task trong)")
    ap.add_argument("--out", required=True, help="vd cvat_import_r1.zip")
    ap.add_argument("--split-by-video", type=int, default=1, metavar="K")
    ap.add_argument("--with-images", action="store_true")
    a = ap.parse_args()

    imgs = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        sys.exit(f"Khong co anh trong {a.images}")
    stems = [p.stem for p in imgs]
    if len(set(stems)) != len(stems):
        sys.exit("Co anh trung ten (stem) -> CVAT se ghep nhan sai. Doi ten truoc.")
    label_dir = Path(a.labels) if a.labels else None
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # nhom theo video, chia K phan can bang so anh (greedy: video lon nhat -> phan it anh nhat)
    groups = defaultdict(list)
    for p in imgs:
        groups[video_of(p.stem)].append(p)
    k = max(1, min(a.split_by_video, len(groups)))
    parts = [[] for _ in range(k)]
    for vid in sorted(groups, key=lambda v: (-len(groups[v]), v)):
        min(parts, key=len).extend(groups[vid])

    rows = []
    for i, part in enumerate(parts, 1):
        part.sort()
        zp = out if k == 1 else out.with_name(f"{out.stem}_part{i}.zip")
        n_box, n_missing, img_zip = write_part(zp, part, label_dir, a.with_images)
        vids = sorted({video_of(p.stem) for p in part})
        print(f"[{i}/{k}] {zp.name}: {len(part)} anh, {n_box} hop nhap, {len(vids)} video"
              + (f", {n_missing} anh khong co file nhan (de trong)" if n_missing else "")
              + f" | anh: {img_zip.name}")
        rows += [[zp.name, p.name, video_of(p.stem)] for p in part]
    with out.with_name(out.stem + "_parts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["part_zip", "image", "video"])
        w.writerows(rows)
    print("\nCVAT: Create task (label GreenSM, upload *_images.zip) -> Actions > Upload annotations > 'YOLO 1.1' -> chon zip nhan.")
    print("Nhan trong zip la NHAN NHAP: moi anh phai duoc nguoi duyet/sua truoc khi export.")


if __name__ == "__main__":
    main()
