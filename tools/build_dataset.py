#!/usr/bin/env python
"""Gom anh + nhan DA DUYET thanh zip dataset cho notebook chinh thuc (train_and_export.ipynb).

Nguon (--sources, thu muc hoac .zip) chap nhan 3 kieu:
  - CVAT export "YOLO 1.1" (obj_train_data/ chua anh + txt; neu export khong kem anh -> dung --image-pool)
  - Ultralytics: images/.../x.jpg + labels/.../x.txt
  - Phang: x.jpg canh x.txt
Nguon sau GHI DE nguon truoc khi trung ten anh (dat ban da sua sau cung).

Kiem tra: giong het read_labels cua notebook (5 cot, lop 0, toa do [0,1]) + them:
w,h > 0, hop trung lap, hop qua nho (px goc), hop tran ra ngoai anh (tu cat), anh khong co file nhan.
Chia train/val THEO VIDEO GOC (stem truoc '_f<so>') de val khong bi ro ri khung gan giong.
Zip ra: train/images, train/labels, val/images, val/labels -> notebook nhan la "da chia san".
Sau khi ghi, zip duoc giai nen va kiem lai bang dung logic label_path_of/read_labels/is_in cua notebook.

Vi du:
  python tools/build_dataset.py --sources reviewed/cvat_part1.zip reviewed/cvat_part2.zip \
      --image-pool frames --val-ratio 0.2 --seed 3 --out datasets/du_lieu_v1.zip
"""
import argparse
import csv
import datetime as dt
import hashlib
import random
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
META_TXT = {"train.txt", "valid.txt", "val.txt", "test.txt", "classes.txt", "readme.txt"}
CLASSES = ["GreenSM"]


def video_of(stem):
    m = re.match(r"^(.*)_f\d+$", stem)
    return m.group(1) if m else stem


def label_path_of(img):  # y het notebook
    parts = list(img.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        cand = Path(*parts).with_suffix(".txt")
        if cand.is_file():
            return cand
    cand = img.with_suffix(".txt")
    return cand if cand.is_file() else None


def read_labels_official(label_file):  # y het notebook (dung de tu kiem zip dau ra)
    lines, problems = [], []
    if label_file is None:
        return lines, problems
    for n, line in enumerate(label_file.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        cols = line.split()
        if len(cols) != 5:
            problems.append(f"{label_file.name} dong {n}: can 5 cot")
            continue
        try:
            cls, vals = int(cols[0]), [float(x) for x in cols[1:]]
        except ValueError:
            problems.append(f"{label_file.name} dong {n}: khong phai so")
            continue
        if cls != 0:
            problems.append(f"{label_file.name} dong {n}: id lop {cls} != 0")
        elif not all(0.0 <= v <= 1.0 for v in vals):
            problems.append(f"{label_file.name} dong {n}: toa do ngoai [0,1]")
        else:
            lines.append(" ".join(cols))
    return lines, problems


def image_size(p):
    try:
        from PIL import Image

        with Image.open(p) as im:
            return im.size
    except Exception:
        import cv2

        img = cv2.imread(str(p))
        return (img.shape[1], img.shape[0]) if img is not None else (0, 0)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def collect(sources, pools, tmp_root):
    """Tra ve dict stem -> (img_path, label_path|None, source_name)."""
    pool = {}
    for d in pools:
        for p in Path(d).rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                pool.setdefault(p.stem, p)
    items = {}
    for s in sources:
        s = Path(s)
        if s.suffix.lower() == ".zip":
            d = Path(tempfile.mkdtemp(dir=tmp_root))
            with zipfile.ZipFile(s) as z:
                z.extractall(d)
            root = d
        else:
            root = s
        if not root.is_dir():
            sys.exit(f"Khong thay nguon {s}")
        imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and "__MACOSX" not in p.parts]
        paired, n_new = set(), 0
        for img in imgs:
            lab = label_path_of(img)
            if lab:
                paired.add(lab.resolve())
            if img.stem in items:
                print(f"  [GHI DE] {img.name}: {items[img.stem][2]} -> {s.name}")
            items[img.stem] = (img, lab, s.name)
            n_new += 1
        # nhan khong kem anh (CVAT export bo tick 'Save images') -> tim anh trong --image-pool
        n_pool = 0
        for t in root.rglob("*.txt"):
            if t.resolve() in paired or t.name.lower() in META_TXT:
                continue
            img = pool.get(t.stem)
            if img is None:
                print(f"  [CANH BAO] {s.name}: nhan {t.name} khong co anh (them --image-pool ?) -> bo qua")
                continue
            if t.stem in items:
                print(f"  [GHI DE] {t.stem}: {items[t.stem][2]} -> {s.name}")
            items[t.stem] = (img, t, s.name)
            n_pool += 1
        print(f"Nguon {s.name}: {n_new} anh kem theo" + (f", {n_pool} nhan ghep anh tu pool" if n_pool else ""))
    return items


def read_excludes(log_path):
    """Ten anh (stem) co action=exclude trong labeling_log.csv (guideline muc 9)."""
    p = Path(log_path)
    if not log_path or not p.exists():
        return set()
    with p.open(newline="", encoding="utf-8-sig") as f:
        return {Path(r["image"].strip()).stem for r in csv.DictReader(f)
                if (r.get("action") or "").strip().lower() == "exclude" and (r.get("image") or "").strip()}


def check_labels(stem, img, lab, a):
    """Kiem tra 1 anh. Tra ve (lines_sach, problems, warnings, boxes_px, (w,h))."""
    w, h = image_size(img)
    probs, warns, boxes = [], [], []
    if lab is None:
        return [], probs, [("no_label", "khong co file nhan -> coi la anh nen")], [], (w, h)
    raw = lab.read_text(encoding="utf-8", errors="replace").splitlines()
    for n, line in enumerate(raw, 1):
        if not line.strip():
            continue
        cols = line.split()
        tag = f"{lab.name} dong {n}"
        if len(cols) != 5:
            probs.append(f"{tag}: can dung 5 cot `0 cx cy w h` (dang co {len(cols)})")
            continue
        try:
            cls, vals = int(cols[0]), [float(x) for x in cols[1:]]
        except ValueError:
            probs.append(f"{tag}: gia tri khong phai so")
            continue
        if cls != 0:
            probs.append(f"{tag}: id lop {cls} khong hop le (chi co lop 0 {CLASSES[0]})")
            continue
        if not all(-a.clip_eps <= v <= 1.0 + a.clip_eps for v in vals):
            probs.append(f"{tag}: toa do phai nam trong [0, 1]: {vals}")
            continue
        cx, cy, bw, bh = vals
        if bw <= 0 or bh <= 0:
            probs.append(f"{tag}: w/h phai > 0")
            continue
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        if min(x1, y1) < -1e-6 or max(x2, y2) > 1 + 1e-6:
            warns.append(("clipped", f"{tag}: hop tran ra ngoai anh -> da cat vao trong"))
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            probs.append(f"{tag}: hop rong sau khi cat")
            continue
        box = (x1, y1, x2, y2)
        if any(iou(box, b) > 0.98 for b in boxes):
            warns.append(("duplicate_removed", f"{tag}: hop trung lap -> da bo"))
            continue
        for b in boxes:
            if iou(box, b) > a.dup_iou:
                warns.append(("overlap", f"{tag}: IoU>{a.dup_iou} voi hop khac (2 xe chong nhau hay ve 2 lan?)"))
        if w and min((x2 - x1) * w, (y2 - y1) * h) < a.min_px:
            warns.append(("small", f"{tag}: hop nho {((x2 - x1) * w):.0f}x{((y2 - y1) * h):.0f}px < {a.min_px}px"))
        boxes.append(box)
    lines = [f"0 {(b[0] + b[2]) / 2:.6f} {(b[1] + b[3]) / 2:.6f} {b[2] - b[0]:.6f} {b[3] - b[1]:.6f}" for b in boxes]
    return lines, probs, warns, [((b[2] - b[0]) * w, (b[3] - b[1]) * h) for b in boxes], (w, h)


def split_by_video(stems, a):
    by_vid = defaultdict(list)
    for s in stems:
        by_vid[video_of(s)].append(s)
    vids = sorted(by_vid)
    if a.val_videos:
        val_v = {v.strip() for v in a.val_videos.split(",") if v.strip()}
        unknown = val_v - set(vids)
        if unknown:
            sys.exit(f"--val-videos khong ton tai: {sorted(unknown)}. Co: {vids[:30]}...")
    else:
        if len(vids) < 2:
            sys.exit("Chi co 1 video -> khong chia theo video duoc. Dung --val-videos hoac them nguon.")
        order = vids[:]
        random.Random(a.seed).shuffle(order)
        target, val_v, n = a.val_ratio * len(stems), set(), 0
        for v in order:  # them video neu dua tong so anh val GAN muc tieu hon
            if len(val_v) + 1 == len(vids):  # chua lai it nhat 1 video cho train
                break
            k = len(by_vid[v])
            if not val_v or abs(n + k - target) < abs(n - target):
                val_v.add(v)
                n += k
    val = sorted(s for s in stems if video_of(s) in val_v)
    train = sorted(s for s in stems if video_of(s) not in val_v)
    return train, val, sorted(val_v)


def verify_like_notebook(zip_path, tmp_root):
    """Giai nen zip va chay dung logic prepare_dataset (khong train)."""
    raw = Path(tempfile.mkdtemp(dir=tmp_root))
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw)
    images = sorted(p for p in raw.rglob("*") if p.suffix.lower() in IMG_EXTS and "__MACOSX" not in p.parts)
    probs, n_boxes, items = [], 0, []
    for img in images:
        lines, pr = read_labels_official(label_path_of(img))
        probs += pr
        n_boxes += len(lines)
        items.append(img)

    def is_in(img, names):
        return any(part.lower() in names for part in img.relative_to(raw).parts[:-1])

    n_val = sum(is_in(i, {"val", "valid", "validation"}) for i in items)
    n_tr = sum(is_in(i, {"train"}) for i in items)
    n_lab = sum(label_path_of(i) is not None for i in items)
    ok = not probs and n_boxes > 0 and len(images) >= 2 and n_val > 0 and n_tr > 0 and n_lab == len(images)
    return ok, dict(images=len(images), boxes=n_boxes, train=n_tr, val=n_val, label_files_found=n_lab, problems=probs[:5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True, help="thu muc hoac .zip (CVAT export / ultralytics / phang)")
    ap.add_argument("--image-pool", nargs="*", default=[], help="thu muc anh goc de ghep voi nhan khong kem anh")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--val-videos", default="", help="danh sach video stem cho val, ngan cach dau phay")
    g.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", required=True, help="vd datasets/du_lieu_v1.zip")
    ap.add_argument("--min-px", type=float, default=8, help="canh bao hop co canh ngan < N px (anh goc)")
    ap.add_argument("--dup-iou", type=float, default=0.9)
    ap.add_argument("--clip-eps", type=float, default=0.002, help="toa do lech [0,1] it hon eps -> tu cat")
    ap.add_argument("--exclude-log", default="logs/labeling_log.csv",
                    help="bo cac anh co action=exclude trong file nay (vung xam, khong chac); '' = tat")
    ap.add_argument("--log", default="logs/datasets.csv")
    ap.add_argument("--note", default="", help="ghi chu cho dong log (vd 'vong 2: them 120 anh am tinh kho')")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="build_ds_"))
    try:
        items = collect(a.sources, a.image_pool, tmp_root)
        excl = read_excludes(a.exclude_log)
        hit = [st for st in items if st in excl]
        for st in hit:
            del items[st]
        if excl:
            print(f"Loai {len(hit)} anh theo {a.exclude_log} (action=exclude)"
                  + (f"; {len(excl) - len(hit)} ten trong log khong co trong nguon" if len(excl) > len(hit) else ""))
        elif a.exclude_log:
            print(f"[CANH BAO] Khong doc duoc anh exclude nao tu '{a.exclude_log}' -> kiem lai duong dan neu nhom co loai anh")
        if len(items) < 2:
            sys.exit(f"Can it nhat 2 anh (co {len(items)}).")
        clean, all_probs, warn_count, no_label, sizes = {}, [], defaultdict(int), [], {}
        warn_lines = []
        for stem in sorted(items):
            img, lab, src = items[stem]
            lines, probs, warns, boxes_px, wh = check_labels(stem, img, lab, a)
            all_probs += [f"[{src}] {p}" for p in probs]
            for key, wmsg in warns:
                warn_count[key] += 1
                warn_lines.append(f"[{src}] {img.name}: {wmsg}")
            if lab is None:
                no_label.append(img.name)
            clean[stem] = (img, lines)
            sizes[stem] = (boxes_px, wh)
        if all_probs:
            print(f"\n[LOI] {len(all_probs)} dong nhan sai -> KHONG tao zip. Sua trong CVAT roi export lai:")
            for p in all_probs[:30]:
                print("  - " + p)
            sys.exit(1)
        n_boxes = sum(len(v[1]) for v in clean.values())
        if n_boxes == 0:
            sys.exit("Khong co hop nao -> notebook se tu choi.")

        train, val, val_v = split_by_video(list(clean), a)
        if not train or not val:
            sys.exit(f"Chia hong: train {len(train)} / val {len(val)} anh.")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # anh jpg da nen
            for split, stems in (("train", train), ("val", val)):
                for s in stems:
                    img, lines = clean[s]
                    z.write(img, f"{split}/images/{s}{img.suffix.lower()}")
                    z.writestr(f"{split}/labels/{s}.txt", "\n".join(lines) + ("\n" if lines else ""))
        ok, info = verify_like_notebook(out, tmp_root)
        sha = hashlib.sha256(out.read_bytes()).hexdigest()

        # ---- thong ke ----
        def stats(stems):
            nb = sum(len(clean[s][1]) for s in stems)
            bg = sum(1 for s in stems if not clean[s][1])
            return len(stems), nb, bg

        def size_hist(stems, scale640=False):
            hst = {"small": 0, "medium": 0, "large": 0}
            for s in stems:
                boxes_px, (w, h) = sizes[s]
                k = 640.0 / max(w, h) if scale640 and max(w, h) else 1.0
                for bw, bh in boxes_px:
                    area = bw * bh * k * k
                    hst["small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"] += 1
            return hst

        print("\n=== THONG KE ===")
        for name, stems in (("TONG", list(clean)), ("train", train), ("val", val)):
            n, nb, bg = stats(stems)
            print(f"{name:5s}: {n} anh | {nb} hop | anh nen {bg} ({100 * bg / max(1, n):.0f}%) | "
                  f"{len({video_of(s) for s in stems})} video")
        print(f"Kich thuoc hop (px goc, nguong COCO 32/96): {size_hist(list(clean))}")
        print(f"Kich thuoc hop (quy ve 640 khi train):      {size_hist(list(clean), True)}")
        print(f"Video val: {', '.join(val_v)}")
        if stats(val)[1] == 0:
            print("[CANH BAO] val khong co hop nao -> mAP val vo nghia. Chon video val co GreenSM.")
        if warn_lines:
            print(f"Canh bao ({len(warn_lines)}): " + ", ".join(f"{k}={v}" for k, v in warn_count.items()))
            for wl in warn_lines[:10]:
                print("  - " + wl)
        rep = out.with_name(out.stem + "_report.txt")
        rep.write_text("\n".join(warn_lines) + "\n\n# Anh khong co file nhan (coi la nen):\n" + "\n".join(no_label) + "\n",
                       encoding="utf-8")
        with out.with_name(out.stem + "_split.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "image", "video", "source", "n_boxes"])
            for split, stems in (("train", train), ("val", val)):
                for s in stems:
                    w.writerow([split, clean[s][0].name, video_of(s), items[s][2], len(clean[s][1])])
        print(f"\nTu kiem theo logic notebook: {'OK' if ok else 'THAT BAI'} {info}")
        print(f"Zip: {out} | sha256 {sha[:16]}... | chi tiet: {rep.name}, {out.stem}_split.csv")

        log = Path(a.log)
        log.parent.mkdir(parents=True, exist_ok=True)
        new = not log.exists()
        n, nb, bg = stats(list(clean))
        with log.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["version", "datetime", "images", "boxes", "background", "train_images", "train_boxes",
                            "val_images", "val_boxes", "val_videos", "seed", "sha256", "zip", "note"])
            w.writerow([out.stem, dt.datetime.now().isoformat(timespec="seconds"), n, nb, bg, len(train),
                        stats(train)[1], len(val), stats(val)[1], ";".join(val_v), a.seed, sha, str(out), a.note])
        print(f"Da ghi log: {log}")
        if not ok:
            sys.exit(2)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
