# Chiến lược GreenSM — L2B Nhóm 04 (TEAM_ID = 3)

> Bản gộp từ 4 góc nhìn: kiểm luật, kỹ thuật YOLO (đọc source ultralytics), điều phối, pipeline.
> Soạn 10:20 ngày 04/10/2026. Nguồn luật: slide BTC (S3–S12), `GreenSM_Challenge_context.pdf`, notebook `train_and_export.ipynb`.

## Chiến lược trong 1 câu

**Dữ liệu giống ảnh test + nhãn đủ, không sót và ôm sát + mỗi vòng sửa đúng chỗ model sai.** Máy gán nháp để làm nhanh, nhưng **người duyệt 100% ảnh**.

Vì sao chỉ cần chừng đó: mọi đội dùng cùng model YOLOv8n, nên chỉ có dữ liệu tạo ra khác biệt. Điểm là mAP@[.5:.95], tức thưởng cho hai thứ: **tìm đủ xe** (không sót) và **hộp sát mép** (IoU cao ở các ngưỡng 0.75–0.95).

---

## 1. So với plan cũ: giữ, sửa, bỏ

| Plan cũ | Quyết định | Vì sao (có kiểm chứng) |
|---|---|---|
| Viết guideline trước khi gán | **Giữ** | Nhãn lệch quy ước giữa 3 người thì model học trung bình của 3 kiểu hộp, IoU tụt |
| Quay video giống ảnh mẫu, trích 1–2 khung/giây, lọc trùng | **Giữ** | Ảnh test là ảnh điện thoại, ngang tầm mắt, khu đô thị, nhiều xe xa hoặc bị che |
| Gán nháp 2 tầng: COCO tìm xe, lọc màu xanh ngọc | **Giữ, có chỉnh** | COCO cho hộp xe khá sát. Lọc màu giúp bỏ bớt xe hãng khác |
| Chia train/val theo video | **Giữ** | Khung liên tiếp gần giống nhau. Chia ngẫu nhiên thì val mAP bị đẩy lên ảo |
| Conf ≥ 0.6 tự chấp nhận; kiểm 10% ngẫu nhiên | **SỬA → duyệt 100%** | **Phạm luật.** Slide 7 cấm "nhãn do model sinh ra mà không kiểm tra lại". Conf chỉ dùng để **sắp thứ tự** ảnh cần duyệt |
| COCO `classes=[2]` (car) | **SỬA → `[2,5,7]` + `agnostic_nms=True`** | VF e34, VF5, Limo Green đôi khi bị COCO gọi là truck (7) hoặc bus (5) |
| TTA `augment=True` | **SỬA → predict ở `imgsz=1280`** | TTA của ultralytics thu ảnh nhỏ lại (×0.83, ×0.67), **hại xe nhỏ ở xa**, và ghép bằng NMS chứ không phải WBF |
| Lọc HSV quyết định nhãn | **SỬA → chỉ để xếp hạng duyệt** | Dễ sai với sedan xanh sáng, xe trong bóng râm, crop nhỏ hơn 20 px |
| Tỉ lệ 70% có xe / 20% âm tính khó / 10% nền | **SỬA → 85–90% có xe / 10–15% âm tính khó** | Hướng dẫn của Ultralytics là 0–10% ảnh nền. Ảnh có GreenSM vốn đã chứa sẵn xe khác làm mẫu âm |
| `scale=0.5–0.9` | **SỬA → giữ 0.5** | `scale=0.9` có thể thu xe 20 px còn 2 px, rồi hộp đó bị lọc bỏ |
| Chọn 2 bài final theo val | **SỬA → 1 bài public cao nhất + 1 bài val cao nhất** | Chia rủi ro: public chỉ 40 ảnh nên nhiễu, val là của mình nên cũng có thể lệch |
| Dùng "biển vàng" để nhận diện | **BỎ** | Xe kinh doanh vận tải nào cũng biển vàng. Dựa vào màu xanh ngọc và logo "V Xanh SM" |
| Teacher YOLOv8l/x @ 960–1280 tự train | **BỎ** | Trên T4 mất 1–3 giờ, không kịp. **Thay bằng best.pt của chính đội** từ vòng 2 |
| WBF | **BỎ** | Cần nhiều model và thêm thư viện. Đằng nào người cũng phải chỉnh từng hộp |
| Train student 2 giai đoạn, GĐ2 `lr0=0.002` | **BỎ** | Khi mọi nhãn đã được người duyệt thì không còn khác biệt giữa "nhãn teacher" và "nhãn người". Hơn nữa `optimizer=auto` **bỏ qua lr0** và tự chọn đúng 0.002, nên GĐ2 không thấp hơn GĐ1 |
| `hsv_h ≤ 0.005` | **BỎ (để mặc định 0.015)** | Mặc định chỉ dịch hue ±2,7 đơn vị. Xanh ngọc H≈90, sedan xanh dương H≈110, không thể lẫn sang nhau. Nên giảm `hsv_s` xuống 0.5 thì đúng hơn |
| `copy_paste=0.1` | **BỎ** | Chỉ chạy khi có nhãn polygon. Nhãn chỉ có bbox thì tham số này bị bỏ qua hoàn toàn |

**Thay cho teacher riêng:** từ vòng 2, dùng `best.pt` của đội làm máy gán nháp cho khung hình mới. Ảnh nào model kém tự tin thì cho duyệt trước, và xem đó là chỗ cần quay thêm. Đây chính là **active learning**, và cũng là câu chuyện "vòng lặp dữ liệu tự cải thiện" đáng kể trên slide.

---

## 2. Pipeline 7 bước: làm gì, đem lại gì, vì sao

### Bước 1 — Guideline v1 (Trung soạn, cả nhóm chốt trong 5 phút lúc 11:00)
| Trường hợp | Quy tắc |
|---|---|
| **Gán** | Xe Xanh SM sơn xanh ngọc, kể cả loại phối xanh ngọc + bạc (như ảnh bìa). Nhận biết bằng màu xanh ngọc và logo "V Xanh SM" |
| **Không gán** | VinFast cùng dáng nhưng màu khác, sedan xanh dương, xe máy hoặc tài xế Xanh SM Bike, xe trên poster hoặc biển quảng cáo, hình phản chiếu trên kính |
| Xe bạc có chữ "SM" (SMP004) | ❓ Hỏi BTC. **Trong lúc chờ: loại cả ảnh khỏi tập**, để không dạy model sai theo hướng nào |
| Bị che một phần | Gán nếu còn nhận ra. Hộp ôm **phần nhìn thấy** (❓ hỏi TA vẽ kiểu nào) |
| Bị cắt ở mép ảnh | Gán phần nhìn thấy |
| Xe nhỏ, ở xa | Gán nếu còn thấy màu xanh ngọc và dáng ô tô. **Không chắc có phải GreenSM không → loại cả ảnh**, không được để sót |
| Mép hộp | Sát từng pixel, **zoom vào khi vẽ xe nhỏ**, tính cả gương chiếu hậu (❓). Nhất quán quan trọng hơn chọn quy ước nào |
| Ảnh nền | Vẫn phải có người xem qua. File `.txt` rỗng cũng là một nhãn |

- **Đem lại:** 3 người vẽ cùng một kiểu hộp.
- **Vì sao:** với hộp 20 px, lệch 1 px mỗi mép đã mất khoảng 10–17% IoU. Xe sót nhãn thì model học rằng "đây là nền".

### Bước 2 — Quay video (Cường và Long, 10:20–11:00)
- **Làm gì:**
  - Quay ngang, ngang tầm mắt. iPhone chỉnh sang "Most Compatible" (H.264) để dễ đọc file.
  - Mỗi clip 30–60 giây, đứng yên rồi lia chậm. Quay **≥ 5 điểm khác nhau**, ghi tên điểm vào tên file: `{nguoi}_{diadiem}_{hhmm}`.
  - Lấy đủ loại cảnh: xa (xe chỉ 30–80 px), vừa, gần; nhìn trước, sau, ngang, chéo; bị xe máy hoặc cây che; nhiều GreenSM trong một khung.
  - Quay thêm clip **âm tính khó**: VinFast trắng, vàng, đỏ, xanh rêu; sedan xanh dương; xe máy xanh.
  - Cứ khoảng 15 phút upload một lần lên Drive chung `GSM_T3/raw/`.
- **Đem lại:** dữ liệu cùng miền với ảnh test, đa dạng góc nhìn.
- **Vì sao:** 200 ảnh đúng kiểu ảnh test đáng giá hơn 1000 ảnh lệch kiểu. Ghi tên điểm quay để chia val theo **địa điểm**, khi đó val mới đo được khả năng tổng quát.

### Bước 3 — Cắt khung hình và lọc trùng (Colab, khoảng 11:00)
- **Làm gì:** lấy 1,5 khung mỗi giây, resize về chiều rộng 1280, bỏ khung gần giống nhau (dHash) và khung quá mờ. Tên file giữ nguồn: `{video}_f000123.jpg`.
- **Đem lại:** vài trăm đến khoảng 1000 ảnh không trùng lặp, luôn truy được về video gốc.
- **Vì sao:**
  - Ảnh test rộng khoảng 1280. Model resize cạnh dài về 640, nên lưu ảnh lớn hơn cũng không thêm chi tiết mà chỉ làm zip nặng.
  - **Không cắt nhỏ hay tile ảnh**: tile làm xe to gấp đôi so với lúc test, lệch phân bố kích thước.

### Bước 4 — Gán nháp 2 tầng (Colab T4)
- **Làm gì:**
  - YOLOv8x COCO: `imgsz=1280, conf=0.05, classes=[2,5,7], agnostic_nms=True`.
  - Cắt từng hộp, tính tỉ lệ pixel xanh ngọc: H 80–100, S ≥ 70, V ≥ 50, bỏ 10–15% mỗi mép crop. Tỉ lệ ≥ 0.10 là ứng viên, ≥ 0.25 là chắc.
  - Xuất nhãn YOLO kèm hàng đợi duyệt: ảnh nào nhiều rủi ro (conf thấp, tỉ lệ màu sát ngưỡng, hộp tí hon) thì xếp lên đầu.
- **Đem lại:** khoảng 1000 khung xử lý trong vài phút. Người chỉ phải sửa, không phải vẽ từ đầu, nhanh gấp 3–5 lần.
- **Vì sao:** slide 6 của BTC gợi ý "Model có sẵn gán nháp, người kiểm tra lại từng hộp". Dùng COCO làm công cụ là hợp lệ, chỉ cấm dùng **bộ dữ liệu** đã gán sẵn.

### Bước 5 — Người duyệt 100% (cả 3 người, mỗi người duyệt video của mình trên CVAT riêng)
- **Làm gì:**
  1. **Hiệu chuẩn (5 phút):** cả 3 cùng vẽ chung 10 ảnh rồi so IoU giữa các bản. Lưu kết quả làm bằng chứng QA.
  2. Duyệt từng ảnh theo thứ tự:
     - **quét cả ảnh tìm GreenSM bị sót**, nhất là xe xa hoặc bị che (lỗi đắt nhất);
     - xoá hộp sai (sedan xanh, VinFast);
     - **kéo mép hộp cho sát**;
     - đánh dấu đã xong.
  3. Long kiểm chéo khoảng 15% ảnh của 2 người còn lại và ghi lỗi vào `qa/`. Người sở hữu task tự sửa.
  4. Mỗi ảnh ghi log: ai duyệt, nhãn nháp hay tay, đã thêm, sửa hay xoá gì.
- **Đem lại:** nhãn sạch, hợp lệ, có bằng chứng để tái lập.
- **Vì sao:** đây là chỗ thắng hay thua. Tốc độ mục tiêu ≥ 3 ảnh/phút/người. Mỗi video chỉ một người sở hữu, nên gộp nhãn chỉ là chép thư mục, không xung đột.

### Bước 6 — Đóng zip, train, nộp (Trung)
- **Làm gì:**
  - Zip cấu trúc `train/images, train/labels, val/images, val/labels`. Notebook nhận ra là đã chia sẵn khi thấy thư mục tên `train` và `val`.
  - **Tập val gồm 2–3 video cố định** quay ở địa điểm khác với train. Giữ nguyên qua mọi vòng, và cho 2 người duyệt.
  - Train bằng notebook chính thức (cấu hình ở mục 6), chạy Bước 3 nguyên bản, rồi nộp.
- **Đem lại:** một thước đo nội bộ ổn định, và số liệu trước/sau so được với nhau giữa các vòng.
- **Vì sao:** bảng public chỉ 40 ảnh, sai 1 xe là điểm nhảy mạnh. Train trong notebook chính thức thì BTC chạy lại được.

### Bước 7 — Phân tích lỗi rồi quay lại vòng sau
- **Làm gì:**
  - Chạy `model.val(plots=True)` trên val, chia lỗi thành:
    - FN (sót): xe xa, bị che, góc lạ;
    - FP (báo nhầm): sedan xanh, VinFast, xe máy xanh.
  - Ứng với từng loại lỗi: lập shot list quay vòng 2, thêm ảnh âm tính khó, hoặc sửa nhãn.
  - Chạy `best.pt` lên khung mới, duyệt trước những ảnh conf 0.1–0.4 (active learning).
- **Đem lại:** mỗi vòng sửa đúng chỗ yếu, không gom thêm dữ liệu một cách mù quáng.
- **Vì sao:** đây là nội dung chính của slide ("số liệu trước và sau mỗi vòng"), và cũng là tiêu chí của giải Sáng tạo.

---

## 3. Lịch song song

| Giờ | Cường | Long | Trung | Colab |
|---|---|---|---|---|
| 10:20–11:00 | ★Quay điểm A, B, C | ★Quay điểm D, E, F | Hỏi BTC; soạn guideline v1; bật CVAT; Colab Bước 0 (TEAM_ID=3); thử công cụ | A: tải yolov8x |
| 11:00–11:15 | Upload; **chốt guideline (5')**; **hiệu chuẩn 10 ảnh** | ← | ← | A: cắt khung + gán nháp |
| 11:15–12:00 | ★Duyệt video của mình | ★Duyệt | ★Duyệt; ghi log | |
| 12:00–12:20 | Duyệt tiếp | Kiểm chéo | ★Đóng `du_lieu_v1.zip` → train → **nộp #1** | B: train v1 |
| 12:20–13:00 | Ăn trưa luân phiên, duyệt nốt lô 1 | Phân tích lỗi v1 → shot list vòng 2 | Ghi metrics; dựng khung slide | |
| 13:00–13:40 | Sửa nhãn theo lỗi | [nếu qua G3] **Quay vòng 2** nhắm đúng lỗi | Train v2 → nộp #2 | A: best.pt gán nháp khung vòng 2 |
| 13:40–14:30 | ★Duyệt vòng 2 | ★Duyệt vòng 2 | ★Duyệt; đóng v3 | B/C: train v3 và một biến thể |
| **14:40** | **Bắt đầu lượt train cuối** | | | |
| 14:40–15:15 | Chạy lại notebook trên runtime sạch (kiểm tái lập) | Bảng và biểu đồ trước/sau | Nộp các lượt cuối; làm slide | |
| **15:00** | Đóng băng dữ liệu | | | |
| **15:15 / 15:20** | Lượt nộp cuối / **chốt 2 bài final** | | ★ | |
| 15:30–16:00 | Slide | Slide | Slide | |

**Mốc quyết định:**
- **G1 (11:00):** có ≥ 8 clip chưa? Chưa đủ thì quay thêm 15 phút ngay gần chỗ đang đứng.
- **G2 (12:00):** đủ ảnh đã duyệt thì train v1 ngay, kể cả chỉ có 100 ảnh. Nộp sớm để bắt lỗi định dạng.
- **G3 (12:45):** v1 đã có điểm và tốc độ duyệt đạt ≥ 3 ảnh/phút → quay vòng 2. Không đạt thì bỏ quay, dồn sức duyệt nốt và sửa nhãn.
- **Teacher riêng:** chỉ làm nếu đã có ≥ 500 ảnh duyệt **và** có một Colab rảnh lúc 13:00. Khi đó dùng YOLOv8s/m, `imgsz=960`, train tối đa 20 phút.

**Lịch 10 lượt nộp:** #1 khoảng 12:15 (v1) · #2 khoảng 13:00 (v2) · #3 (biến thể của v2) · #4–#5 khoảng 14:20–14:40 (v3) · #6–#7 đến 15:10 · **giữ 3 lượt dự phòng**. Không nộp sau 15:15, vì mỗi lúc chỉ được một bài chờ chấm.

**Chọn final:** **F1** = bài có điểm public cao nhất. **F2** = bài có val mAP50-95 cao nhất trên tập val cố định.

---

## 4. Câu hỏi cho BTC (xếp theo mức ảnh hưởng)
1. "Model gán nháp, người mở từng ảnh để duyệt hoặc sửa, có ghi log. Như vậy đã tính là 'kiểm tra từng hộp' chưa ạ?"
2. "Xe bị che thì TA vẽ hộp ôm phần nhìn thấy hay cả xe? Gương chiếu hậu có tính vào hộp không ạ?"
3. "Xe Xanh SM màu bạc như ảnh SMP004 có tính là GreenSM không ạ?"
4. "Có được dùng ảnh internet không ạ? Có cần ghi nguồn không?"
5. "10 ảnh mẫu SMP có được đưa vào train không ạ?"
6. "'Điểm phạt' trên bảng public là gì ạ?"

## 5. Bằng chứng (phục vụ 50% điểm thuyết trình và tái lập), lưu trong Drive `GSM_T3/`
- `raw/`: video gốc, **không bao giờ sửa**
- `00_guideline/`: guideline có số phiên bản, kèm ảnh ví dụ
- `logs/labeling_log.csv`, `logs/metrics.csv` (mỗi lượt: giờ, zip, số ảnh/số hộp, TRAIN_ARGS, val mAP50-95, điểm public)
- `qa/`: biên bản hiệu chuẩn 10 ảnh, kiểm chéo, 5–10 ảnh chụp trước/sau khi sửa
- `datasets/du_lieu_vN.zip` kèm sha256; `runs/vN/` gồm best.pt, results.csv, submission.zip, run_report.json, notebook còn output
- 2 bài final lưu ở hai nơi

## 6. Cấu hình

**Train (ô Bước 2 của notebook chính thức)**, đã kiểm với source ultralytics:
```python
TRAIN_ARGS = dict(
    data=DATASET["data_yaml"], imgsz=IMGSZ, seed=SEED, deterministic=True,
    project=str(WORK_DIR / "runs"), name="train", exist_ok=True,
    epochs=80, patience=30, batch=32, workers=2, cache="disk",   # cache="ram" có thể làm mất tính tái lập
    optimizer="AdamW", lr0=0.002,
    mosaic=1.0, close_mosaic=10, scale=0.5, translate=0.1, fliplr=0.5,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=0.0, mixup=0.0, copy_paste=0.0,
)
```
❓ Thời gian train chưa đo thực tế. Ước tính 800 ảnh khoảng 12–20 giây mỗi epoch. Hãy đo ở v1 rồi chỉnh `epochs` theo.

**Gán nháp (Colab T4)**:
```python
from ultralytics import YOLO
m = YOLO("yolov8x.pt")
m.predict(source="frames/", imgsz=1280, conf=0.05, iou=0.6, classes=[2, 5, 7],
          agnostic_nms=True, max_det=100, half=True, device=0, stream=False,
          save_txt=True, save_conf=True, project="prelabel")
# sau đó: lọc HSV → đổi class về 0 → import CVAT → người duyệt TỪNG hộp
```
Công cụ đầy đủ: thư mục `DAY15/tools/` (xem `README_TOOLS.md`).
