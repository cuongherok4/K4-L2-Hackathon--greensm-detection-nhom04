#!/usr/bin/env python
"""Trich khung hinh tu video (va copy+resize anh chup) cho bai GreenSM.

- Lay mau N khung/giay, resize canh dai ve --width (khong phong to), giu ty le.
- Bo khung gan trung lap lien tiep bang dHash (numpy, khong can thu vien ngoai).
- (Tuy chon) bo khung qua mo bang phuong sai Laplacian (--blur-thresh > 0).
- Ten file: <videostem>_f<000123>.jpg  -> luon truy nguoc duoc video goc
  (video stem = phan truoc '_f<so>' cuoi cung).
- Ghi <out>/manifest.csv: moi khung da xet (kept / dropped + ly do).

Vi du:
  python tools/extract_frames.py --videos raw/videos --out frames --fps 1.5 --width 1280
  python tools/extract_frames.py --videos raw/cuong --out frames --prefix cuong --blur-thresh 40
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".3gp", ".webm"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_stem(text):
    """Ten an toan cho file: chi giu chu/so/-/_ ; bo '_f<so>' o cuoi de khong nham video stem."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "src"
    return re.sub(r"_f(\d+)$", r"_F\1", s)


def dhash(img_bgr, size=8):
    """dHash 64 bit: so sanh pixel ke nhau tren anh xam 9x8."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b)) if a is not None and b is not None else 999


def blur_score(img_bgr):
    """Phuong sai Laplacian tren anh xam (tinh o ~640 px de on dinh). Cang thap cang mo."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    if max(h, w) > 640:
        s = 640.0 / max(h, w)
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def resize_long_side(img, target):
    """Resize canh dai = target (anh ngang => chieu rong = target). Khong phong to."""
    h, w = img.shape[:2]
    if target <= 0 or max(h, w) <= target:
        return img
    s = target / float(max(h, w))
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def rotate(img, deg):
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


class Filter:
    """Giu trang thai dedup trong 1 nguon (so voi khung DA GIU gan nhat)."""

    def __init__(self, hash_thresh, blur_thresh, hash_size=8):
        self.hash_thresh, self.blur_thresh, self.hash_size, self.last = hash_thresh, blur_thresh, hash_size, None

    def check(self, img):
        b = blur_score(img)
        if self.blur_thresh > 0 and b < self.blur_thresh:
            return "dropped_blur", b, ""
        h = dhash(img, self.hash_size)
        d = hamming(h, self.last)
        if self.hash_thresh >= 0 and d <= self.hash_thresh:
            return "dropped_dup", b, d
        self.last = h
        return "kept", b, ("" if d == 999 else d)


def process_video(path, stem, args, out_dir, rows):
    cap = cv2.VideoCapture(str(path))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):  # tu xoay theo metadata (iPhone quay doc)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    if not cap.isOpened():
        print(f"  [LOI] khong mo duoc video {path.name} (HEVC/codec?). Bo qua.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 1e-3 or fps > 1000:
        fps = 30.0
        print(f"  [CANH BAO] {path.name}: khong doc duoc FPS, gia dinh 30.")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    flt = Filter(args.hash_thresh, args.blur_thresh, args.hash_size)
    step_t, next_t, idx, kept = 1.0 / args.fps, 0.0, -1, 0
    while True:
        if not cap.grab():  # grab() nhanh hon read(); chi decode khung can lay
            break
        idx += 1
        t = idx / fps
        if t + 1e-6 < next_t:
            continue
        next_t += step_t
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue
        frame = resize_long_side(rotate(frame, args.rotate), args.width)
        name = f"{stem}_f{idx:06d}.jpg"
        status, b, d = flt.check(frame)
        if status == "kept":
            cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            kept += 1
        rows.append([name, path.name, idx, f"{t:.2f}", frame.shape[1], frame.shape[0], f"{b:.1f}", d, status])
        if args.max_per_video and kept >= args.max_per_video:
            break
    cap.release()
    h, w = (rows[-1][5], rows[-1][4]) if rows else (0, 0)
    warn = "  [CANH BAO: video doc -> khac mien test 16:9 ngang]" if h > w else ""
    print(f"  {path.name}: fps={fps:.1f}, ~{n_total} khung, giu {kept}{warn}")


def process_photo(path, stem, args, out_dir, rows):
    img = cv2.imread(str(path))  # imread tu ap dung EXIF orientation
    if img is None:
        print(f"  [LOI] khong doc duoc anh {path.name} (HEIC? hay doi sang JPG). Bo qua.")
        return
    img = resize_long_side(rotate(img, args.rotate), args.width)
    name = f"{stem}_f000000.jpg"
    flt = Filter(-1, args.blur_thresh)  # anh chup: khong dedup theo chuoi, chi loc mo
    status, b, _ = flt.check(img)
    if status == "kept":
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    rows.append([name, path.name, 0, "0.00", img.shape[1], img.shape[0], f"{b:.1f}", "", status])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="thu muc chua video va/hoac anh chup (quet de quy)")
    ap.add_argument("--out", required=True, help="thu muc ghi khung hinh")
    ap.add_argument("--fps", type=float, default=1.5, help="so khung lay moi giay")
    ap.add_argument("--width", type=int, default=1280, help="canh dai sau resize (test ~1280); 0 = giu nguyen")
    ap.add_argument("--hash-thresh", type=int, default=6, help="Hamming dHash <= nguong => trung lap (-1 = tat)")
    ap.add_argument("--hash-size", type=int, default=16,
                    help="8 => 64 bit. Camera dung yen (xe nho di chuyen): dung 16 hoac --hash-thresh -1")
    ap.add_argument("--blur-thresh", type=float, default=0, help="Laplacian var < nguong => bo (0 = tat, goi y 30-60)")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="xoay thu cong neu metadata sai")
    ap.add_argument("--prefix", default="", help="tien to ten file, vd ten nguoi quay (tranh trung IMG_0001 giua cac may)")
    ap.add_argument("--quality", type=int, default=95, help="chat luong JPEG")
    ap.add_argument("--max-per-video", type=int, default=0, help="gioi han so khung giu moi video (0 = khong)")
    args = ap.parse_args()

    src, out_dir = Path(args.videos), Path(args.out)
    if not src.is_dir():
        sys.exit(f"Khong thay thu muc {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS | IMG_EXTS)
    if not files:
        sys.exit(f"Khong co video/anh nao trong {src}")

    rows, seen = [], {}
    for p in files:
        stem = safe_stem((args.prefix + "_" if args.prefix else "") + p.stem)
        if stem in seen:  # 2 file cung stem (vd IMG_0001.MOV va IMG_0001.JPG)
            other = seen[stem]
            stem = f"{stem}_{p.suffix.lower().lstrip('.')}"
            print(f"  [CANH BAO] {p.name} trung ten nguon voi {other}, doi stem -> {stem}")
        seen[stem] = p.name
        if p.suffix.lower() in VIDEO_EXTS:
            process_video(p, stem, args, out_dir, rows)
        else:
            process_photo(p, stem, args, out_dir, rows)

    man = out_dir / "manifest.csv"
    new = not man.exists()
    with man.open("a", newline="", encoding="utf-8") as f:  # append: chay nhieu lan (nhieu nguoi) van giu lich su
        w = csv.writer(f)
        if new:
            w.writerow(["file", "source", "frame_idx", "time_s", "width", "height", "blur", "dhash_dist", "status"])
        w.writerows(rows)

    from collections import Counter

    c = Counter(r[8] for r in rows)
    print(f"\nTong: xet {len(rows)} khung | giu {c['kept']} | bo trung lap {c['dropped_dup']} | bo mo {c['dropped_blur']}")
    print(f"Anh o: {out_dir}  | manifest: {man}")


if __name__ == "__main__":
    main()
