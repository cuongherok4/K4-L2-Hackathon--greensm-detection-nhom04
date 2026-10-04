#!/usr/bin/env python
"""Gan nhan NHAP (draft) 2 tang cho GreenSM. Day KHONG phai nhan cuoi cung.

Tang 1: model COCO (mac dinh yolov8x.pt) tim xe (car=2, bus=5, truck=7).
Tang 2: voi moi hop, tinh ty le pixel xanh ngoc (teal) trong HSV tren 75% phia tren
        cua crop (bo 25% duoi: duong/bong/banh xe). ratio >= --teal-thresh => GreenSM.

Dau ra (--out):
  <stem>.txt         YOLO `0 cx cy w h` chi cho hop GreenSM duoc chap nhan (anh nen = file rong)
  review_queue.csv   moi anh 1 dong, sap xep theo priority giam dan -> duyet anh rui ro truoc
  boxes.csv          moi hop xe 1 dong (conf, teal ratio, accepted...) de chinh nguong
  viz/               (--viz) anh ve hop de xem nhanh

LUAT CUOC THI: moi nhan do model sinh ra PHAI duoc nguoi kiem tra lai trong CVAT.

Vi du:
  python tools/prelabel.py --images frames --out prelabels --viz
  python tools/prelabel.py --images frames --out prelabels --model yolov8n.pt --device cpu --imgsz 640
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REMINDER = ("NHAC NHO: day la NHAN NHAP do model sinh ra. Luat cuoc thi cam dung nhan model "
            "chua kiem tra -> moi anh (ke ca anh rong) phai duoc nguoi duyet/sua trong CVAT.")


def teal_ratio(img, box, a):
    """Ty le pixel teal trong phan tren (1 - bottom_ignore) cua crop."""
    x1, y1, x2, y2 = box
    y2c = y1 + max(1, int(round((y2 - y1) * (1.0 - a.bottom_ignore))))
    crop = img[y1:y2c, x1:x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)  # H: 0-180 (OpenCV)
    mask = cv2.inRange(hsv, (a.h_min, a.s_min, a.v_min), (a.h_max, 255, 255))
    return float(np.count_nonzero(mask)) / mask.size


def to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    vals = [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
    return [min(1.0, max(0.0, v)) for v in vals]


def draw(img, recs):
    vis = img.copy()
    for r in recs:
        x1, y1, x2, y2 = r["box"]
        if r["accepted"]:
            col = (0, 165, 255) if r["borderline"] else (200, 200, 0)  # cam = can xem ky, teal = GreenSM
        else:
            col = (0, 165, 255) if r["teal_near"] else (160, 160, 160)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        tag = f"{'GSM' if r['accepted'] else r['cls_name']} c{r['conf']:.2f} t{r['teal']:.2f}"
        cv2.putText(vis, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolov8x.pt", help="model COCO (yolov8x.pt / yolov8l.pt / rtdetr-x.pt...)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--classes", type=int, nargs="+", default=[2, 5, 7], help="COCO id: 2 car, 5 bus, 7 truck")
    ap.add_argument("--device", default=None, help="0 (GPU) hoac cpu; bo trong = tu chon")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--augment", action="store_true", help="TTA cua ultralytics (cham hon, hop on dinh hon)")
    # tang 2: mau teal (OpenCV H 0-180)
    ap.add_argument("--h-min", type=int, default=80)
    ap.add_argument("--h-max", type=int, default=100)
    ap.add_argument("--s-min", type=int, default=60)
    ap.add_argument("--v-min", type=int, default=50)
    ap.add_argument("--bottom-ignore", type=float, default=0.25, help="bo phan duoi crop (duong, bong)")
    ap.add_argument("--teal-thresh", type=float, default=0.28, help="ratio >= nguong => GreenSM")
    ap.add_argument("--teal-band", type=float, default=0.08, help="|ratio - nguong| < band => borderline")
    ap.add_argument("--lowconf", type=float, default=0.5, help="hop GreenSM conf < nguong nay => borderline")
    ap.add_argument("--tiny-px", type=int, default=20, help="hop co canh ngan < N px => tiny")
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args()

    img_dir, out = Path(a.images), Path(a.out)
    paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        sys.exit(f"Khong co anh trong {img_dir}")
    stems = [p.stem for p in paths]
    if len(set(stems)) != len(stems):
        sys.exit("Co anh trung ten (stem) trong cac thu muc con -> doi ten truoc khi chay.")
    out.mkdir(parents=True, exist_ok=True)
    if a.viz:
        (out / "viz").mkdir(exist_ok=True)

    if a.device is None:
        import torch
        a.device = "0" if torch.cuda.is_available() else "cpu"
    from ultralytics import YOLO

    model = YOLO(a.model)
    names = model.names
    print(f"Model {a.model} | device {a.device} | imgsz {a.imgsz} | conf {a.conf} | classes {a.classes} | {len(paths)} anh")
    print(REMINDER)

    queue, box_rows = [], []
    lo, hi = a.teal_thresh - a.teal_band, a.teal_thresh + a.teal_band
    results = model.predict(source=[str(p) for p in paths], imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                            classes=a.classes, device=a.device, augment=a.augment, batch=a.batch,
                            stream=True, verbose=False)
    for k, r in enumerate(results, 1):
        p = Path(r.path)  # lay ten tu ket qua (an toan hon zip theo thu tu)
        img = r.orig_img
        h, w = img.shape[:2]
        recs = []
        for xyxy, c, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            x1, y1 = int(max(0, np.floor(xyxy[0]))), int(max(0, np.floor(xyxy[1])))
            x2, y2 = int(min(w, np.ceil(xyxy[2]))), int(min(h, np.ceil(xyxy[3])))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            t = teal_ratio(img, (x1, y1, x2, y2), a)
            acc = t >= a.teal_thresh
            tiny = min(x2 - x1, y2 - y1) < a.tiny_px
            near = lo <= t < hi
            low = float(c) < a.lowconf
            recs.append(dict(box=(x1, y1, x2, y2), conf=float(c), cls_name=names[int(cl)], teal=t, accepted=acc,
                             teal_near=near, borderline=acc and (near or low or tiny), low=low, tiny=tiny))
        acc_recs = [x for x in recs if x["accepted"]]
        lines = ["0 " + " ".join(f"{v:.6f}" for v in to_yolo(x["box"], w, h)) for x in acc_recs]
        (out / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        n_near = sum(x["teal_near"] for x in recs)
        n_low = sum(x["low"] for x in acc_recs)
        n_tiny = sum(x["tiny"] for x in acc_recs)
        min_conf = min((x["conf"] for x in acc_recs), default=None)
        # priority: hop mau teal mo ho > GreenSM conf thap > hop nho > anh co GreenSM > anh co xe
        prio = 3 * n_near + 2 * n_low + 1 * n_tiny + (1 if acc_recs else 0) + (0.5 if recs else 0)
        queue.append([p.name, len(recs), len(acc_recs), "" if min_conf is None else f"{min_conf:.3f}",
                      n_near, n_low, n_tiny, f"{prio:.1f}"])
        for x in recs:
            box_rows.append([p.name, x["cls_name"], f"{x['conf']:.3f}", f"{x['teal']:.3f}", int(x["accepted"]),
                             int(x["teal_near"]), int(x["tiny"]), *x["box"]])
        if a.viz:
            cv2.imwrite(str(out / "viz" / f"{p.stem}.jpg"), draw(img, recs), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if k % 50 == 0:
            print(f"  ... {k}/{len(paths)}")

    queue.sort(key=lambda r: -float(r[-1]))
    with (out / "review_queue.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "n_vehicles", "n_greensm", "min_conf_greensm", "n_teal_borderline",
                     "n_greensm_lowconf", "n_greensm_tiny", "priority"])
        wr.writerows(queue)
    with (out / "boxes.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "coco_class", "conf", "teal_ratio", "accepted", "teal_borderline", "tiny",
                     "x1", "y1", "x2", "y2"])
        wr.writerows(box_rows)

    n_img_g = sum(1 for q in queue if q[2] > 0)
    print(f"\nXong: {len(paths)} anh | {len(box_rows)} hop xe | {sum(q[2] for q in queue)} hop GreenSM nhap "
          f"tren {n_img_g} anh | {sum(1 for q in queue if q[1] == 0)} anh khong co xe")
    print(f"Nhan nhap: {out}/*.txt | hang doi duyet: {out / 'review_queue.csv'}" + (f" | viz: {out / 'viz'}" if a.viz else ""))
    print(REMINDER)


if __name__ == "__main__":
    main()
