"""Sinh hình minh hoạ cho GUIDELINE_v1 từ 11 ảnh mẫu của BTC (src/).
Toạ độ hộp lấy từ prelabel YOLOv8x rồi được người kiểm và chỉnh tay (xem ghi chú từng hộp)."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
SRC, OUT = ROOT / "src", ROOT / "img"
OUT.mkdir(exist_ok=True)
FONT = "C:/Windows/Fonts/arial.ttf"
FONT_B = "C:/Windows/Fonts/arialbd.ttf"
GREEN, RED, ORANGE, GREY = (0, 200, 70), (230, 40, 40), (255, 150, 0), (120, 120, 120)


def font(sz, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, sz)


def load(name):
    return Image.open(SRC / f"{name}.jpg").convert("RGB")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def n_thr(v):
    return sum(v >= t - 1e-9 for t in np.arange(0.5, 0.96, 0.05))


def crop(img, box, pad=0.25, size=360, boxes=(), min_pad=12):
    """Cắt quanh box, phóng về chiều dài cạnh lớn = size; vẽ các boxes=(box,color,width,dash)."""
    W, H = img.size
    x1, y1, x2, y2 = box
    p = max(min_pad, int(pad * max(x2 - x1, y2 - y1)))
    a, b, c, d = max(0, x1 - p), max(0, y1 - p), min(W, x2 + p), min(H, y2 + p)
    s = size / max(c - a, d - b)
    im = img.crop((a, b, c, d)).resize((int((c - a) * s), int((d - b) * s)), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    for bx, col, w, dash in boxes:
        r = [(bx[0] - a) * s, (bx[1] - b) * s, (bx[2] - a) * s, (bx[3] - b) * s]
        if dash:
            dashed_rect(dr, r, col, w)
        else:
            dr.rectangle(r, outline=col, width=w)
    return im


def dashed_rect(dr, r, col, w, seg=10):
    x1, y1, x2, y2 = r
    for (xa, ya, xb, yb) in [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]:
        L = max(abs(xb - xa), abs(yb - ya))
        n = int(L // seg)
        for i in range(0, n, 2):
            t0, t1 = i / max(n, 1), min((i + 1) / max(n, 1), 1)
            dr.line([(xa + (xb - xa) * t0, ya + (yb - ya) * t0), (xa + (xb - xa) * t1, ya + (yb - ya) * t1)], fill=col, width=w)


def full(img, boxes=(), width=1038):
    s = width / img.size[0]
    im = img.resize((width, int(img.size[1] * s)), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    for bx, col, w, dash in boxes:
        r = [v * s for v in bx]
        (dashed_rect(dr, r, col, w) if dash else dr.rectangle(r, outline=col, width=w))
    return im


def card(im, tag, tag_col, lines, w=None):
    """Ảnh + nhãn màu (ĐÚNG/SAI...) + 1-3 dòng chú thích bên dưới."""
    w = w or im.size[0]
    cap_h = 34 + 24 * len(lines)
    out = Image.new("RGB", (w, im.size[1] + cap_h), "white")
    out.paste(im, ((w - im.size[0]) // 2, 0))
    dr = ImageDraw.Draw(out)
    f = font(20, True)
    tw = dr.textlength(tag, font=f)
    dr.rounded_rectangle([8, im.size[1] + 6, 8 + tw + 16, im.size[1] + 32], 6, fill=tag_col)
    dr.text((16, im.size[1] + 8), tag, font=f, fill="white")
    for i, ln in enumerate(lines):
        dr.text((10, im.size[1] + 36 + 24 * i), ln, font=font(18), fill=(30, 30, 30))
    return out


def grid(cards, cols, title=None, gap=14, bg=(245, 247, 246)):
    cw = max(c.size[0] for c in cards)
    ch = max(c.size[1] for c in cards)
    rows = (len(cards) + cols - 1) // cols
    th = 50 if title else 0
    out = Image.new("RGB", (cols * cw + (cols + 1) * gap, th + rows * ch + (rows + 1) * gap), bg)
    dr = ImageDraw.Draw(out)
    if title:
        dr.text((gap, 12), title, font=font(26, True), fill=(20, 50, 50))
    for i, c in enumerate(cards):
        r, k = divmod(i, cols)
        out.paste(c, (gap + k * (cw + gap), th + gap + r * (ch + gap)))
    return out


def save(im, name):
    im.save(OUT / name, quality=88)
    print("->", name, im.size)


# ---------- hộp đã kiểm (toạ độ ảnh 1038 px) ----------
G = dict(
    smp005_main=(501, 192, 802, 394),
    smp006_main=(187, 204, 638, 423),
    smp009_main=(453, 188, 667, 306),
    smp009_far=(625, 187, 715, 227),
    smp001_far=(106, 131, 255, 198),
    smp001_tiny=(326, 127, 352, 143),       # xe GreenSM ~26 px, khuất sau xe khác
    smp003_a=(166, 108, 361, 206),          # chỉnh tay: prelabel gộp 2 xe (166..410)
    smp003_b=(362, 110, 418, 142),          # xe thứ 2 bị SUV đen che, chỉ lộ nóc + kính
    smp004_main=(684, 192, 826, 293),
    smp002_a=(154, 224, 259, 278),
    smp002_b=(280, 216, 337, 249),
    smp005_left=(71, 250, 159, 291),
    smp008_main=(224, 299, 437, 404),
    smp010_far=(25, 202, 113, 246),
)

# 1) Bộ sưu tập: GreenSM trông thế nào
pos = [
    ("SMP005", "smp005_main", ["Chính diện, gần: thân xanh ngọc,", "nóc bạc, logo V ở đầu xe"]),
    ("SMP006", "smp006_main", ["Nhìn ngang: logo 'V Xanh SM'", "trên cửa, viền bạc thân dưới"]),
    ("SMP009", "smp009_main", ["Nhìn từ phía sau", "(đuôi xe, đèn hậu)"]),
    ("SMP004", "smp004_main", ["Đi tới, khoảng cách vừa"]),
    ("SMP003", "smp003_a", ["Đang đỗ, bụi cây che chân xe:", "VẪN GÁN, hộp tới mép nhìn thấy"]),
    ("SMP001", "smp001_far", ["Ở xa, nhỏ (~150 px)", "vẫn gán, vẽ sát"]),
]
cards = [card(crop(load(n), G[k], boxes=[(G[k], GREEN, 3, False)]), "GÁN NHÃN", GREEN, cap) for n, k, cap in pos]
save(grid(cards, 3, "GreenSM: các góc nhìn cần GÁN (hộp xanh lá = hộp đúng)"), "01_greensm_gallery.jpg")

# 2) Không phải GreenSM
neg = [
    ("SMP007", (29, 250, 247, 414), ["VinFast màu vàng: cùng dáng xe,", "KHÁC màu -> không gán"]),
    ("SMP008", (464, 318, 1020, 521), ["Sedan xanh dương: màu gần giống,", "đậm hơn, không logo -> không gán"]),
    ("SMP001", (823, 117, 1038, 231), ["Xe xanh đậm / xanh rêu:", "không phải xanh ngọc -> không gán"]),
    ("SMP006", (0, 360, 310, 584), ["Xe máy điện màu xanh", "(kể cả Xanh SM Bike) -> không gán"]),
    ("SMP010", (173, 183, 436, 305), ["VinFast trắng (xe cá nhân)", "-> không gán"]),
    ("SMP007", (378, 250, 582, 323), ["Xe đỏ, xe hãng khác", "-> không gán"]),
]
cards = []
for n, bx, cap in neg:
    im = crop(load(n), bx, pad=0.12, boxes=[(bx, RED, 3, True)])
    cards.append(card(im, "KHÔNG GÁN", RED, cap))
save(grid(cards, 3, "Dễ nhầm nhưng KHÔNG phải GreenSM (để trống, coi là nền)"), "02_not_greensm.jpg")

# 3) Vùng xám: xe bạc có chữ SM
im = crop(load("SMP004"), (0, 223, 230, 572), pad=0.05, size=420, boxes=[((0, 223, 230, 572), ORANGE, 4, True)])
save(card(im, "VÙNG XÁM -> LOẠI CẢ ẢNH", ORANGE,
          ["Xe màu BẠC có số '369' và chữ '...SM' trên cửa.",
           "Không sơn xanh ngọc -> không chắc tính là GreenSM.",
           "Không gán cũng không để làm nền: loại ảnh, ghi log."], w=520), "03_grayzone_silver_sm.jpg")

# 4) Độ chặt của hộp (xe lớn)
b = G["smp005_main"]
loose = (b[0] - 30, b[1] - 20, b[2] + 30, b[3] + 20)
cut = (b[0] + 17, b[1] + 14, b[2] - 16, b[3] - 16)
img = load("SMP005")
win = (b[0] - 40, b[1] - 30, b[2] + 40, b[3] + 30)
p_ok = card(crop(img, win, pad=0, boxes=[(b, GREEN, 3, False)]), "ĐÚNG", GREEN,
            ["Sát mép trái/phải/trên/dưới,", "tính cả gương và lốp chạm đất"])
v = iou(loose, b)
p_lo = card(crop(img, win, pad=0, boxes=[(b, GREEN, 1, True), (loose, RED, 3, False)]), "SAI: HỘP LỎNG", RED,
            [f"Dư ~10% mỗi phía -> IoU {v:.2f}", f"chỉ đạt {n_thr(v)}/10 ngưỡng chấm"])
v2 = iou(cut, b)
p_cu = card(crop(img, win, pad=0, boxes=[(b, GREEN, 1, True), (cut, RED, 3, False)]), "SAI: CẮT MẤT", RED,
            [f"Hụt gương, cản, lốp -> IoU {v2:.2f}", f"chỉ đạt {n_thr(v2)}/10 ngưỡng chấm"])
save(grid([p_ok, p_lo, p_cu], 3, "Hộp phải ÔM SÁT: điểm chấm mAP@[.5:.95] phạt cả hộp lỏng lẫn hộp hụt"), "04_box_tightness.jpg")

# 5) Xe nhỏ: lệch 3 px đã mất điểm
t = G["smp001_tiny"]
t_sh = (t[0] + 3, t[1], t[2] + 3, t[3])
vt = iou(t_sh, t)
big_sh = (b[0] + 3, b[1], b[2] + 3, b[3])
vb = iou(big_sh, b)
p1 = card(crop(load("SMP001"), t, pad=1.2, size=420, boxes=[(t, GREEN, 3, False), (t_sh, RED, 2, True)]),
          f"XE NHỎ 26 px: LỆCH 3 px", RED, [f"IoU {vt:.2f} -> {n_thr(vt)}/10 ngưỡng", "Zoom 300-400% khi vẽ xe nhỏ!"], w=440)
p2 = card(crop(load("SMP005"), b, pad=0.15, size=420, boxes=[(b, GREEN, 3, False), (big_sh, RED, 2, True)]),
          f"XE LỚN 300 px: LỆCH 3 px", GREEN, [f"IoU {vb:.2f} -> {n_thr(vb)}/10 ngưỡng", "Xe lớn chịu sai số tốt hơn"], w=440)
save(grid([p1, p2], 2, "Cùng lệch 3 pixel: xe nhỏ mất điểm nặng, xe lớn gần như không sao"), "05_small_box_precision.jpg")

# 6) Xe nhỏ ở xa: toàn ảnh + phóng to
img = load("SMP001")
fi = full(img, [(G["smp001_far"], GREEN, 3, False), (G["smp001_tiny"], GREEN, 3, False)])
inset = crop(img, G["smp001_tiny"], pad=1.5, size=300, boxes=[(G["smp001_tiny"], GREEN, 3, False)])
dr = ImageDraw.Draw(fi)
fi.paste(inset, (fi.size[0] - inset.size[0] - 12, fi.size[1] - inset.size[1] - 70))
dr.rectangle([fi.size[0] - inset.size[0] - 12, fi.size[1] - inset.size[1] - 70, fi.size[0] - 12, fi.size[1] - 70], outline="white", width=3)
dr.text((fi.size[0] - inset.size[0] - 4, fi.size[1] - 66), "Phóng to: xe GreenSM ~26 px khuất sau xe khác", font=font(18, True), fill="white", stroke_width=3, stroke_fill="black")
save(card(fi, "GÁN CẢ 2 XE", GREEN, ["Ảnh có 2 GreenSM: 1 xe ở xa bên trái và 1 xe TÍ HON ở giữa (người dễ bỏ sót).",
                                      "Quét toàn ảnh trước khi bấm 'xong'. Xe bị sót nhãn = dạy model rằng xe đó là nền."]),
     "06_small_far_cars.jpg")

# 7) Hai xe dính nhau + bị che (SMP003)
img = load("SMP003")
win = (120, 80, 470, 230)
a = card(crop(img, win, pad=0, size=520, boxes=[((166, 108, 410, 208), ORANGE, 3, True)]), "NHÁP CỦA MÁY: SAI", RED,
         ["1 hộp gộp 2 xe GreenSM", "(xe sau bị SUV đen che)"], w=540)
c = card(crop(img, win, pad=0, size=520, boxes=[(G["smp003_a"], GREEN, 3, False), (G["smp003_b"], GREEN, 3, False)]),
         "SAU KHI SỬA: ĐÚNG", GREEN, ["2 hộp riêng; xe sau chỉ lộ nóc + kính", "-> hộp ôm đúng phần nhìn thấy"], w=540)
save(grid([a, c], 2, "Hai xe đứng sát / che nhau: mỗi xe MỘT hộp, chỉ ôm phần nhìn thấy"), "07_two_cars_occluded.jpg")

# 8) Bị che bởi người / cây / xe máy
o1 = card(crop(load("SMP002"), G["smp002_a"], pad=0.6, size=380, boxes=[(G["smp002_a"], GREEN, 3, False), (G["smp002_b"], GREEN, 3, False)]),
          "GÁN", GREEN, ["Bụi cây và xe máy che một phần", "-> vẫn gán, hộp tới mép nhìn thấy"])
o2 = card(crop(load("SMP005"), G["smp005_left"], pad=0.5, size=380, boxes=[(G["smp005_left"], GREEN, 3, False)]),
          "GÁN", GREEN, ["Người đứng trước xe", "-> hộp vẫn ôm cả xe từ trái sang phải"])
o3 = card(crop(load("SMP006"), (184, 233, 233, 263), pad=0.8, size=380, boxes=[((184, 233, 233, 263), GREEN, 3, False)]),
          "GÁN", GREEN, ["Xe phía sau, bị người lái xe máy", "che một phần -> vẫn gán"])
save(grid([o1, o2, o3], 3, "Bị che một phần nhưng còn nhận ra là GreenSM: VẪN GÁN"), "08_partial_occlusion.jpg")

# 9) Nhiều xe + viền đen (SMP009)
fi = full(load("SMP009"), [(G["smp009_main"], GREEN, 3, False), (G["smp009_far"], GREEN, 3, False)])
save(card(fi, "GÁN TẤT CẢ GreenSM TRONG ẢNH", GREEN,
          ["2 xe: 1 xe gần (nhìn từ sau) + 1 xe ở xa phía sau. Hai hộp được phép chồng lên nhau.",
           "Ảnh có viền đen hai bên: chỉ gán trong vùng ảnh thật, không kéo hộp vào viền đen."]), "09_multiple_cars_pillarbox.jpg")

# 10) Nhoè chuyển động: ảnh bìa BTC (đã có hộp mẫu của BTC)
save(card(full(load("COVER")), "MẪU CỦA BTC", GREEN,
          ["Hộp mẫu do BTC vẽ: ôm toàn bộ thân xe từ mũi đến đuôi, từ nóc xuống chân bánh.",
           "Xe bị nhoè do chạy nhanh vẫn gán; SUV bạc và xe trắng không gán."]), "10_btc_reference_blur.jpg")

# 11) Các lỗi hay gặp của nhãn nháp
e1 = card(crop(load("SMP001"), (823, 117, 1038, 231), pad=0.1, size=360, boxes=[((823, 117, 1038, 231), ORANGE, 3, True)]),
          "NHÁP SAI -> XOÁ", RED, ["Máy nhận xe xanh đậm", "là GreenSM"])
e2 = card(crop(load("SMP005"), (71, 240, 190, 295), pad=0.25, size=360, boxes=[((145, 247, 181, 286), ORANGE, 3, True), (G["smp005_left"], GREEN, 3, False)]),
          "XOÁ CAM, GIỮ XANH", RED, ["Hộp nhỏ dính sang xe xanh dương", "bên cạnh -> xoá"])
e3 = card(crop(load("SMP007"), (240, 240, 362, 300), pad=0.3, size=360, boxes=[((243, 250, 309, 298), ORANGE, 2, True), ((244, 247, 358, 298), ORANGE, 2, True), ((263, 245, 294, 265), ORANGE, 2, True)]),
          "NHÁP TRÙNG -> GỘP 1", RED, ["3 hộp chồng trên 1 xe bị che:", "giữ 1 hộp đúng, xoá phần trùng"])
save(grid([e1, e2, e3], 3, "Nhãn nháp của máy hay sai kiểu này (cam = nháp, xanh = đúng)"), "11_prelabel_errors.jpg")
print("done")
