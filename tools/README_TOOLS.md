# Bộ công cụ dữ liệu — GreenSM Data Centric Challenge (Nhóm 04, seed 3)

Chỉ cần Python + `opencv-python` + `numpy` (+ `ultralytics` cho `prelabel.py`). Không cần ffmpeg, không cần GPU
(GPU chỉ làm `prelabel.py` nhanh hơn). Chạy được trên Windows (Git Bash) và Google Colab.

> **Luật cuộc thi:** nhãn do model sinh ra mà chưa được người kiểm tra là bị cấm. Mọi thứ `prelabel.py` tạo ra
> chỉ là **nhãn nháp**. Phải duyệt TỪNG ảnh trong CVAT, kể cả ảnh không có hộp nào (có thể bị sót xe).

## 1. Từng công cụ làm gì

**`extract_frames.py`**: trích khung từ video (và copy + resize ảnh chụp) vào `frames/`. Lấy `--fps` khung mỗi giây,
resize cạnh dài về `--width` (mặc định 1280, giống ảnh test; không phóng to), tự xoay theo metadata của video.
Bỏ khung gần trùng khung đã giữ trước đó bằng dHash (`--hash-thresh`, khoảng cách Hamming). Nếu bật
`--blur-thresh > 0` thì bỏ cả khung mờ (phương sai Laplacian). Tên file là `<videostem>_f<000123>.jpg`, nên luôn biết
khung đến từ video nào. Ghi thêm `frames/manifest.csv` (giữ/bỏ + lý do) để làm bằng chứng quy trình.

**`prelabel.py`**: tạo nhãn nháp theo 2 tầng. Tầng 1: model COCO (mặc định `yolov8x.pt`) tìm xe (car/bus/truck).
Tầng 2: cắt từng hộp, tính tỉ lệ pixel xanh ngọc trong HSV (H 80–100 theo thang 0–180 của OpenCV, S≥60, V≥50, bỏ 25%
phía dưới của crop). Hộp có tỉ lệ ≥ `--teal-thresh` thì được tính là GreenSM. Kết quả: một `<stem>.txt` cho mỗi ảnh
(ảnh nền thì file rỗng), `review_queue.csv` (ảnh rủi ro xếp trước: tỉ lệ teal sát ngưỡng, conf thấp, hộp nhỏ),
`boxes.csv` (từng hộp xe với conf và tỉ lệ teal, dùng để chỉnh ngưỡng), và `viz/` nếu bật `--viz`.
Màu trong viz: teal = GreenSM, cam = cần xem kỹ, xám = xe khác.

**`make_cvat_import.py`**: đóng gói để đưa vào CVAT. Mỗi phần tạo 2 file: `<tên>_images.zip` (ảnh, dùng khi
**Create task**) và `<tên>.zip` (nhãn định dạng **"YOLO 1.1"**: `obj.names`=GreenSM, `obj.data`, `train.txt`,
`obj_train_data/<stem>.txt`). `--split-by-video K` chia ảnh thành K phần theo video gốc, số ảnh mỗi phần gần bằng
nhau, để K người duyệt song song. Ngoài ra còn ghi `<tên>_parts.csv`.

**`build_dataset.py`**: gom ảnh và nhãn **đã duyệt** thành zip cho notebook chính thức. Nhận 3 kiểu nguồn (thư mục
hoặc .zip): bản export "YOLO 1.1" từ CVAT, bố cục Ultralytics `images/`+`labels/`, hoặc thư mục phẳng có ảnh nằm
cạnh txt. Nếu export không kèm ảnh thì thêm `--image-pool frames`. Nguồn đứng sau ghi đè nguồn đứng trước khi trùng
tên ảnh. Phần kiểm tra giống `read_labels` của notebook, cộng thêm: w,h > 0, hộp trùng (tự bỏ), hộp tràn mép (tự
cắt), hộp nhỏ hơn `--min-px`, ảnh thiếu file nhãn. **Nếu có lỗi thì dừng lại, không tạo zip.**
Train/val được chia **theo video gốc**. Zip có dạng `train/images, train/labels, val/images, val/labels`; notebook
sẽ nhận đây là "đã chia sẵn". Sau khi ghi zip, công cụ tự giải nén và kiểm tra lại bằng đúng logic
`label_path_of/read_labels/is_in` của notebook. Cuối cùng in thống kê và ghi thêm một dòng vào `logs/datasets.csv`
(version, giờ, số lượng, video val, sha256).

## 2. Quy ước thư mục / tên file

```
greensm/                       # thư mục làm việc của đội (local hoặc Drive)
  tools/                       # 4 script này
  raw/<nguoi>/                 # video/ảnh GỐC, KHÔNG xóa (cần cho reproducibility)
  frames/                      # khung đã trích + manifest.csv
  prelabels/                   # nhãn nháp + review_queue.csv + boxes.csv + viz/
  cvat/                        # zip import CVAT (cvat_import_r1_part1.zip ...)
  reviewed/                    # zip export từ CVAT SAU KHI DUYỆT: r1_cuong.zip, r1_long.zip ...
  datasets/                    # du_lieu_v1.zip + _split.csv + _report.txt
  logs/datasets.csv            # lịch sử các phiên bản dữ liệu (đưa vào slide)
```
- Video stem là phần đứng trước `_f<số>`. Dùng `--prefix <tên người>` để không trùng `IMG_0001` giữa các máy.
- Không đổi tên ảnh sau khi trích, vì CVAT và build_dataset ghép nhãn theo **tên file**.
- iPhone: nên để chế độ "Most Compatible" (JPG/H.264). OpenCV không đọc được HEIC, và HEVC có thể lỗi.

## 3. Lệnh theo thứ tự — Windows Git Bash (máy local)

```bash
cd /d/AI20K/MySpace/AI20K-hoc-tap/DAY15        # hoặc thư mục làm việc của đội
# 1) Trích khung (mỗi người một lần, có prefix)
python tools/extract_frames.py --videos raw/cuong --out frames --prefix cuong --fps 1.5 --width 1280
# 2) Nhãn nháp: CPU thì dùng model nhỏ (chậm). Nên chạy bước này trên Colab GPU.
python tools/prelabel.py --images frames --out prelabels --model yolov8n.pt --device cpu --imgsz 640 --viz
# 3) Đóng gói cho 3 người duyệt
python tools/make_cvat_import.py --images frames --labels prelabels --out cvat/cvat_import_r1.zip --split-by-video 3
#    CVAT: Create task (label "GreenSM", Rectangle) -> Select files: cvat_import_r1_partK_images.zip
#          -> Actions > Upload annotations > format "YOLO 1.1" -> cvat_import_r1_partK.zip
#    Duyệt theo prelabels/review_queue.csv -> Export task dataset, format "YOLO 1.1" -> reviewed/r1_<nguoi>.zip
# 4) Build dataset (chia val theo video)
python tools/build_dataset.py --sources reviewed/r1_cuong.zip reviewed/r1_long.zip reviewed/r1_trung.zip \
    --image-pool frames --val-ratio 0.2 --seed 3 --out datasets/du_lieu_v1.zip --note "vong 1"
#    hoặc chỉ định video val:  --val-videos cuong_IMG_0012,long_IMG_0003
```

## 4. Lệnh trên Google Colab (GPU T4)

```python
# Ô 1: mount Drive, vào thư mục đội (đã upload tools/ và raw/ lên Drive)
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/greensm
!pip -q install ultralytics   # Colab da co san opencv + numpy
```
```python
# Ô 2: trích khung ra đĩa local của Colab (nhanh hơn Drive), xong copy về
!python tools/extract_frames.py --videos raw --out /content/frames --fps 1.5 --width 1280
# Ô 3: nhãn nháp bằng model lớn trên GPU
!python tools/prelabel.py --images /content/frames --out /content/prelabels --model yolov8x.pt --imgsz 1280 --conf 0.05 --device 0 --viz
# Ô 4: đóng gói CVAT, copy về Drive
!python tools/make_cvat_import.py --images /content/frames --labels /content/prelabels --out /content/cvat/cvat_import_r1.zip --split-by-video 3
!mkdir -p frames prelabels cvat && cp -r /content/frames/. frames/ && cp -r /content/prelabels/. prelabels/ && cp /content/cvat/* cvat/
```
```python
# Ô 5 (sau khi duyệt xong trên CVAT): build zip cho notebook chính thức
!python tools/build_dataset.py --sources reviewed/r1_cuong.zip reviewed/r1_long.zip reviewed/r1_trung.zip --image-pool frames --val-ratio 0.2 --seed 3 --out datasets/du_lieu_v1.zip
# Trong notebook chính thức: DATASET_ZIP_PATH = "/content/drive/MyDrive/greensm/datasets/du_lieu_v1.zip"
```

## 5. Mẹo chỉnh tham số

- **Ngưỡng teal phải được hiệu chỉnh trên ảnh thật** trước khi chạy toàn bộ. Chạy `prelabel.py --viz` trên khoảng
  30 ảnh, mở `boxes.csv` và so cột `teal_ratio` của xe Xanh SM thật với xe khác (VinFast xanh rêu, sedan xanh dương),
  rồi chỉnh `--teal-thresh`, `--h-min/--h-max`, `--s-min`.
- Quay cố định một chỗ (máy không di chuyển, chỉ xe chạy): dHash 64 bit có thể coi nhầm các khung là trùng. Khi đó
  dùng `--hash-size 16` hoặc `--hash-thresh -1`.
- `--blur-thresh` mặc định tắt. Ảnh đường phố thật thường có giá trị blur vài trăm (xem cột `blur` trong
  manifest). Bắt đầu thử khoảng 30–60.
- Nếu `build_dataset.py` báo lỗi nhãn thì sửa trong CVAT rồi export lại; đừng sửa tay file txt.

## 6. Hiệu chỉnh ngưỡng teal trên ảnh thật (04/10, 11 ảnh mẫu BTC, YOLOv8x @1280)

- GreenSM thật (~15 xe): `teal_ratio` 0,32–0,72. Xe xanh đậm/xanh rêu, xe cạnh xe teal: 0,17–0,22.
- → mặc định `--teal-thresh 0.28`, `--teal-band 0.08` (0,20–0,36 = borderline, duyệt trước).
- Lỗi nháp còn gặp: hộp gộp 2 xe đứng sát (SMP003), hộp nhỏ dính sang xe xanh dương cạnh xe teal (SMP005),
  nhiều hộp chồng trên 1 xe bị che (SMP007). Xem `guideline/GUIDELINE.html` mục 7.

## 7. Sau khi duyệt: loại ảnh và đếm số hộp đã sửa (thêm 04/10, guideline v1.1)

- `build_dataset.py` tự bỏ mọi ảnh có `action=exclude` trong `logs/labeling_log.csv` (tham số `--exclude-log`,
  mặc định đúng đường dẫn này; truyền `--exclude-log ""` để tắt). Người gán không cần xoá ảnh trong CVAT.
- `review_stats.py` so nhãn nháp với file export sau duyệt, ghi `logs/review_stats.csv` (mỗi ảnh: số hộp nháp,
  số hộp cuối, thêm/xoá/chỉnh) và in tổng. Số liệu này dùng cho slide ("người duyệt 100%, sửa X% ảnh").

```bash
python tools/review_stats.py --prelabels prelabels --reviewed reviewed/r1_cuong.zip reviewed/r1_long.zip reviewed/r1_trung.zip     --image-pool frames --out logs/review_stats.csv
```
Ghép hộp nháp với hộp duyệt theo IoU ≥ 0,5; cặp có IoU < 0,98 tính là "chỉnh". Đã thử trên dữ liệu giả, chưa chạy trên dữ liệu thật.
