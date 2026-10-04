# Kế hoạch thực thi — GreenSM Data Centric Challenge · L2B Nhóm 04 (TEAM_ID = 3)

> **Nguồn căn cứ:** S = số slide (đánh số ở chân slide) trong `GreenSM Data Centric Challenge.pdf`; CTX = mục trong `GreenSM_Challenge_context.pdf` (tổng hợp từ website cuộc thi); NB = notebook `train_and_export.ipynb`.
> Lý do kỹ thuật chi tiết: `CHIEN-LUOC-GREENSM.md`. Công cụ: `tools/README_TOOLS.md`.
> Giờ trong kế hoạch là mốc mục tiêu. Nếu trễ thì lùi theo, nhưng **không lùi 3 mốc cứng**: 14:40 bắt đầu lượt train cuối, 15:20 chốt final, 15:30 đóng nộp.

## Mục tiêu và cách thắng

- **Xếp hạng cuối = 50% hạng điểm private + 50% đánh giá giải pháp khi thuyết trình** (S12, CTX §1). Phải thắng ở **cả hai**: model tốt, **và** quy trình có số liệu cùng bằng chứng.
- **Điểm = 100 × mAP@[.5:.95]** (S9). Thưởng cho việc **tìm đủ xe** và **vẽ hộp sát mép**.
- **Model là như nhau cho mọi đội** (S2, S4), chỉ có dữ liệu tạo ra khác biệt. Ba đòn bẩy:
  1. Ảnh **giống ảnh test** (điện thoại, ban ngày, ngang tầm mắt, khu đô thị, nhiều xe xa hoặc bị che: CTX §3b).
  2. Nhãn **đủ, không sót, ôm sát**.
  3. **Vòng lặp:** phân tích lỗi → sửa đúng chỗ → đo lại (S5).

## Đầu ra phải có lúc 15:30

| Đầu ra | Phục vụ |
|---|---|
| 2 bài nộp đã chọn final trên website | 50% điểm private (S10) |
| `du_lieu_final.zip` + notebook đã train (còn output) + `run_report.json` | Tái lập kết quả (S12) |
| Thư mục bằng chứng: video gốc, guideline, labeling log, QA notes, ảnh chụp màn hình CVAT | Chứng minh khi thuyết trình (S7, S12) |
| Bảng số liệu các vòng (val mAP, điểm public, số ảnh/số hộp) | Nội dung chính của slide (S5, CTX §8) |

---

## GIAI ĐOẠN A — Chuẩn bị và thu thập (10:30 → 11:05)

### A1. Chốt các điểm luật còn mở ✅ XONG 10:50 (đổi cách làm: KHÔNG hỏi BTC)
- **Quyết định của nhóm:** không hỏi BTC. Các quy ước gán nhãn là một phần của cuộc thi; hỏi hết thì chẳng khác gì biết trước cách gán bộ test.
- **Đã làm:** tự chốt 11 quyết định (QĐ-01 đến QĐ-11), mỗi quyết định có căn cứ từ thể lệ và các guideline 2D của khoá học (DAY03, DAY08, G01). Xem Phụ lục A của guideline.
- **Giữ nguyên mặc định an toàn:** KHÔNG dùng ảnh internet, KHÔNG đưa 10 ảnh SMP vào train (QĐ-06). Ảnh SMP chỉ dùng làm hình minh hoạ và để chạy thử pipeline.
- **Thể lệ:** S7 ("Ảnh tải từ internet: BTC sẽ xác nhận"), S3 (định nghĩa GreenSM).

### A2. Viết guideline ✅ v1.1 XONG 11:00, chờ nhóm duyệt
- **Sản phẩm:** `guideline/GUIDELINE.html` trong repo nhóm (bản duy nhất, 11 hình dựng từ ảnh thật, cây quyết định, 8 quy tắc vẽ hộp, quy trình CVAT, checklist, bảng tra nhanh, lịch sử thay đổi).
- **Còn lại:** cả nhóm đọc trong 5 phút, sau đó hiệu chuẩn 10 ảnh (B4). Chỗ nào lệch thì Trung sửa thành v1.2 và ghi vào lịch sử thay đổi.
- **Vì sao:**
  - 3 người vẽ 3 kiểu thì model học kiểu trung bình, hộp không sát.
  - Xe GreenSM bị sót nhãn thì model học rằng "chỗ này là nền".
- **Thể lệ:** S5 (bước 02 "Viết label guideline"), S6 ("Guideline và duyệt nhãn: quy ước thống nhất"), S3 ("Xe khác không gán nhãn"), CTX §8 (slide phải có "label guideline").

### A3. Dựng hạ tầng và nơi lưu bằng chứng
- **Ai / khi nào:** Trung, 10:30–11:00.
- **Làm gì:**
  1. Tạo Drive chung `GSM_T3/` với các thư mục: `raw/`, `00_guideline/`, `frames/`, `prelabels/`, `cvat/`, `reviewed/`, `datasets/`, `runs/`, `logs/`, `qa/`, `evidence/`. Upload `tools/` lên.
  2. Bật CVAT: mở Ubuntu (WSL) trước, rồi mới mở Docker Desktop.
  3. Mở notebook chính thức trên Colab, bật GPU T4, điền `TEAM_ID = 3`, chạy Bước 0.
  4. Tạo `logs/metrics.csv` với các cột: giờ, phiên bản dữ liệu, số ảnh, số hộp, TRAIN_ARGS, val mAP50-95, val mAP50, điểm public, ghi chú.
- **Kết quả:** chỗ để mọi bằng chứng có ngay từ phút đầu. Không ai phải chờ ai.
- **Vì sao:** BTC yêu cầu xuất trình bằng chứng khi tái lập. Gom lại vào phút cuối sẽ thiếu.
- **Thể lệ:** S7 ("Lưu lại ngay từ đầu: thư mục ảnh gốc, nhật ký gán nhãn, ảnh chụp màn hình công cụ, ghi chú QA"), S14 (việc số 2), CTX §0 (TEAM_ID 3), S12 ("Seed = mã đội").

### A4. Thu thập vòng 1 (đường găng)
- **Ai / khi nào:** Cường và Long, 10:30–11:05.
- **Làm gì:**
  - iPhone để chế độ "Most Compatible". Quay **ngang**, ngang tầm mắt.
  - Quay ở **≥ 5 địa điểm** (vòng xuyến, ngã tư, vỉa hè, trục đường lớn). Mỗi clip 30–60 giây, đứng yên rồi lia chậm.
  - Đặt tên file `{nguoi}_{diadiem}_{hhmm}`.
  - Shot list:
    - xa (xe chỉ 30–80 px), vừa, gần;
    - nhìn trước, sau, ngang, chéo;
    - bị xe máy, cây hoặc xe khác che;
    - nhiều GreenSM trong một khung;
    - **ít nhất 2 clip âm tính khó**: VinFast trắng, vàng, xanh rêu; sedan xanh dương; xe máy xanh.
  - Cứ khoảng 15 phút upload lên `raw/` một lần.
- **Kết quả:** khoảng 10–15 clip, ước tính 400–900 khung sau khi cắt.
- **Vì sao:**
  - Dữ liệu giống ảnh test là đòn bẩy lớn nhất.
  - Nhiều địa điểm thì tách được val **theo địa điểm**, đo đúng khả năng tổng quát.
  - Ảnh âm tính khó giúp giảm báo nhầm.
- **Thể lệ:** S7 (được phép: "Ảnh thô do chính đội thu thập: tự chụp, quay video rồi cắt khung hình"), S6 ("Thu thập đa dạng… Thêm cả ảnh không có GreenSM"), S3 ("Nên có cả ảnh chứa các xe này làm dữ liệu nền").

**Mốc G1 (11:05):** có ≥ 8 clip, ≥ 5 địa điểm? Chưa đủ thì quay thêm 15 phút ngay gần chỗ đang đứng.

---

## GIAI ĐOẠN B — Tạo nhãn vòng 1 (11:05 → 12:05)

### B1. Cắt khung hình
- **Ai / khi nào:** Trung, trên Colab, khoảng 11:05.
- **Làm gì:** chạy `extract_frames.py --videos raw --out frames --fps 1.5 --width 1280`.
- **Kết quả:** `frames/` (tên file giữ nguồn video, ví dụ `cuong_vongxuyen_1040_f000123.jpg`) và `manifest.csv` ghi khung nào được giữ, khung nào bị bỏ, vì sao.
- **Vì sao:**
  - Khung liên tiếp gần giống nhau thì chỉ tốn công duyệt mà không thêm thông tin.
  - Ảnh test rộng khoảng 1280. Model tự thu về 640, nên lưu lớn hơn chỉ làm zip nặng.
  - **Không tile ảnh**, vì tile làm xe to bất thường so với lúc test.
  - `manifest.csv` là bằng chứng truy ngược từ khung về video gốc.
- **Thể lệ:** S7 ("quay video rồi cắt khung hình"), S12 ("Chứng minh: cho xem ảnh thô").

### B2. Hiệu chỉnh ngưỡng màu trên ảnh thật
- **Ai / khi nào:** Trung, 5 phút.
- **Làm gì:**
  1. Chạy `prelabel.py --viz` trên khoảng 30 khung.
  2. Mở `boxes.csv`, so cột `teal_ratio` của xe GreenSM thật với sedan xanh và VinFast xanh rêu.
  3. Chốt `--teal-thresh` và ghi vào `logs/`.
- **Kết quả:** ngưỡng hợp với ánh sáng thật hôm nay.
- **Vì sao:** ngưỡng mặc định mới chỉ được thử trên dữ liệu giả. Ngưỡng sai thì nhãn nháp sai nhiều, người duyệt phải sửa lâu hơn.
- **Thể lệ:** không ràng buộc. Đây là thao tác chất lượng nội bộ.

### B3. Gán nhãn nháp
- **Ai / khi nào:** Trung, Colab GPU, khoảng 11:10.
- **Làm gì:**
  1. Chạy `prelabel.py --model yolov8x.pt --imgsz 1280 --device 0 --viz`.
  2. Chạy `make_cvat_import.py --split-by-video 3` để chia thành 3 phần theo video, mỗi người một phần.
- **Kết quả:** nhãn nháp cho toàn bộ khung, kèm `review_queue.csv` xếp ảnh rủi ro lên đầu.
- **Vì sao:** người chỉ cần sửa thay vì vẽ từ đầu, nhanh gấp 3–5 lần. Chia theo video thì mỗi video chỉ một người sở hữu, gộp lại không xung đột.
- **Thể lệ:**
  - S6 ("Gán nhãn nháp: Model có sẵn gán nháp, người kiểm tra lại từng hộp"), S7 (được phép: "Công cụ hỗ trợ gán nhãn, đội chịu trách nhiệm kiểm tra từng hộp").
  - ⚠️ Chỉ dùng **trọng số** model COCO làm công cụ. **Không** dùng ảnh hoặc nhãn của COCO hay bất kỳ bộ dữ liệu gán sẵn nào (S7 cấm).
  - Đây là **nhãn nháp**, chưa được đưa vào train.

### B4. Hiệu chuẩn giữa 3 người
- **Ai / khi nào:** cả 3, 5 phút, khoảng 11:15.
- **Làm gì:**
  1. Cả 3 vẽ cùng 10 ảnh khó (xe nhỏ, bị che, có xe dễ nhầm).
  2. So số hộp và IoU giữa các bản.
  3. Chỗ nào lệch thì bổ sung quy tắc vào guideline thành v1.1.
- **Kết quả:** bằng chứng QA định lượng, ví dụ "đồng thuận 9/10 ảnh, IoU trung bình 0,88", lưu ở `qa/calibration.md`.
- **Vì sao:** phát hiện cách hiểu lệch **trước khi** gán hàng trăm ảnh, rẻ hơn nhiều so với sửa sau. Kết quả này cũng là một con số đẹp cho slide.
- **Thể lệ:** S5 (bước 03 "Kiểm tra chéo giữa các thành viên"), S6 ("kiểm tra chéo").

### B5. Duyệt 100% (đường găng)
- **Ai / khi nào:** cả 3, mỗi người duyệt phần của mình trên CVAT riêng, 11:20–12:00.
- **Làm gì:**
  1. Tạo task từ `_images.zip`, rồi Upload annotations dạng "YOLO 1.1". **Thử với 5 ảnh trước.**
  2. Duyệt theo thứ tự của `review_queue.csv`. Với **mỗi ảnh**:
     - quét cả ảnh tìm GreenSM bị sót;
     - xoá hộp sai;
     - kéo mép hộp cho sát;
     - đánh dấu xong.
  3. Ảnh cần loại (vùng xám, không chắc): ghi một dòng `exclude` vào `logs/labeling_log.csv` (guideline mục 9). Số hộp thêm/sửa/xoá do `tools/review_stats.py` tự đếm, không đếm tay.
  4. Chụp 5–10 ảnh màn hình trước/sau khi sửa, lưu vào `evidence/`.
  5. Export dạng "YOLO 1.1" ra `reviewed/r1_<tên>.zip`.
- **Kết quả:** khoảng 150–300 ảnh có nhãn đã được người kiểm (tốc độ mục tiêu ≥ 3 ảnh/phút/người).
- **Vì sao:** chất lượng nhãn quyết định điểm. Log và ảnh chụp chứng minh rằng mọi nhãn máy sinh đều đã được người kiểm.
- **Thể lệ:**
  - S7 cấm "nhãn do model sinh ra mà không kiểm tra lại", nên **không có ngoại lệ "conf cao thì tự chấp nhận"**.
  - Ảnh chưa duyệt **không được đưa vào zip**.

### B6. Kiểm chéo
- **Ai / khi nào:** Long, xen kẽ trong B5, khoảng 15% số ảnh của hai bạn còn lại.
- **Làm gì:** ghi lỗi vào `qa/crosscheck_v1.md`. Người sở hữu task tự sửa.
- **Kết quả:** tỉ lệ lỗi đo được, ví dụ "6/45 ảnh có lỗi: 4 sót xe xa, 2 hộp lỏng". Biết guideline còn hở ở đâu.
- **Vì sao:** người tự duyệt bài mình thường không thấy lỗi của mình.
- **Thể lệ:** S6 ("kiểm tra chéo, sửa nhãn nhiễu"), S12 ("Chứng minh… cách kiểm tra QA").

---

## GIAI ĐOẠN C — Train và nộp vòng 1 (12:05 → 12:30)

### C1. Dựng `du_lieu_v1.zip`
- **Ai:** Trung.
- **Làm gì:** chạy `build_dataset.py --sources reviewed/*.zip --image-pool frames --val-videos <2–3 video ở địa điểm khác> --out datasets/du_lieu_v1.zip`.
- **Kết quả:**
  - Zip cấu trúc `train/…`, `val/…`, đã tự kiểm bằng đúng logic của notebook.
  - Thống kê kích thước hộp.
  - Một dòng trong `logs/datasets.csv`, có sha256 để khóa phiên bản.
- **Vì sao:**
  - **Val cố định theo địa điểm và giữ nguyên qua mọi vòng**, nhờ đó số liệu trước/sau so được với nhau.
  - Chia ngẫu nhiên theo khung sẽ làm val mAP cao ảo.
  - Công cụ chặn nhãn sai định dạng trước khi notebook từ chối.
- **Thể lệ:** CTX §3 (định dạng `0 cx cy w h`, chuẩn hoá [0,1], chỉ lớp 0, ảnh nền là `.txt` rỗng), NB `prepare_dataset` (nhận dữ liệu chia sẵn khi có thư mục `train` và `val`).

### C2. Train trong notebook chính thức
- **Ai:** Trung, Colab B.
- **Làm gì:** điền `DATASET_ZIP_PATH`. Ở Bước 2, dùng TRAIN_ARGS khuyến nghị: `epochs=80, patience=30, batch=32, cache="disk", scale=0.5, hsv_s=0.5, copy_paste=0` (đầy đủ ở `CHIEN-LUOC-GREENSM.md` §6). **Không đụng** `imgsz`, `seed`, `MODEL_NAME`.
- **Kết quả:** `best.pt`, `results.csv` có val mAP50-95, và **thời gian train thực tế** để chỉnh lịch các vòng sau.
- **Vì sao:** train trong notebook chính thức với seed 3 thì BTC chạy lại ra cùng kết quả.
- **Thể lệ:** S4 (YOLOv8n fine-tune từ yolov8n.pt, không đổi kiến trúc, imgsz 640), CTX §2 (seed = 3, ultralytics==8.4.170), NB (chỉ được sửa ô Bước 1 và Bước 2).

### C3. Đóng gói và nộp lượt #1
- **Ai:** Trung.
- **Làm gì:** chạy **Bước 3 nguyên bản**, tải `submission.zip` và `run_report.json`, nộp ở tab "Nộp bài". Lưu cả hai vào `runs/v1/`.
- **Kết quả:** điểm public đầu tiên. Xác nhận toàn bộ đường đi từ dữ liệu đến ONNX đến nộp bài là thông suốt.
- **Vì sao:** lỗi định dạng phát hiện sớm thì còn thời gian sửa. Bài bị từ chối không mất lượt.
- **Thể lệ:** S8 (zip đúng 1 `model.onnx`, ≤ 25 MB), S14 ("Nộp bài đầu tiên sớm để phát hiện lỗi định dạng"), CTX §5 (self-check là chính validator của hệ thống chấm, Bước 3 không được sửa).

### C4. Ghi số liệu
- **Ai:** Trung.
- **Làm gì:** thêm một dòng vào `metrics.csv`, chụp màn hình bảng điểm và đường cong train.
- **Kết quả:** điểm xuất phát (baseline) cho slide.
- **Thể lệ:** S5 ("Ghi lại số liệu trước và sau mỗi vòng cải thiện. Chúng là nội dung chính của slide").

---

## GIAI ĐOẠN D — Các vòng cải thiện (12:30 → 14:40)

### D1. Phân tích lỗi
- **Ai / khi nào:** Long, 12:30–12:50.
- **Làm gì:**
  1. Chạy `model.val(plots=True)` trên val, xem ảnh dự đoán.
  2. Đếm và chia lỗi thành:
     - **FN (sót):** xe xa, bị che, góc lạ, ngược sáng;
     - **FP (báo nhầm):** sedan xanh, VinFast, xe máy xanh, biển hiệu;
     - **hộp lỏng** (đúng xe nhưng IoU thấp).
  3. Ghi vào `qa/error_analysis_v1.md`, kèm 4–6 ảnh minh hoạ.
- **Kết quả:** danh sách lỗi có số đếm, ví dụ "12 FN thì 8 là xe < 40 px". Từ đó ra shot list vòng 2.
- **Vì sao:** sửa đúng chỗ yếu thay vì gom thêm dữ liệu mù quáng. Đây là phần "giải pháp" giám khảo chấm.
- **Thể lệ:** S5 (bước 05 "Phân tích lỗi: xem model sai ở đâu, rồi quay lại bước 01 đến 03"), S6 ("Phân tích lỗi… thu thêm đúng chỗ sai").

### D2. Mốc G3: chọn hành động cho vòng 2 (12:50)

| Lỗi chính | Hành động |
|---|---|
| Sót xe xa hoặc bị che | Long đi quay vòng 2 (30 phút), tập trung xe xa và cảnh đông xe |
| Báo nhầm xe giống màu | Quay thêm clip âm tính khó (giữ ≤ 15% tổng ảnh) |
| Hộp lỏng | Rà lại độ sát của hộp trên các ảnh train, bổ sung quy tắc mép hộp vào guideline v2 |
| Tốc độ duyệt < 3 ảnh/phút | Không quay thêm. Dồn sức duyệt nốt số khung còn lại |

- **Vì sao:** thời gian có hạn, nên chỉ làm thứ đem lại nhiều điểm nhất.

### D3. Active learning cho dữ liệu mới
- **Ai / khi nào:** Trung, Colab A, khoảng 13:30.
- **Làm gì:** dùng `best.pt` của đội (thay cho COCO + lọc màu) gán nháp khung vòng 2. Ảnh có conf 0.1–0.4, hoặc có màu xanh ngọc mà model không bắt được, thì **duyệt trước**.
- **Kết quả:** nhãn nháp vòng 2 chính xác hơn, vì model đã biết GreenSM. Người duyệt tập trung vào đúng ca model còn yếu.
- **Vì sao:** đây là vòng lặp dữ liệu tự cải thiện, câu chuyện đáng kể nhất trên slide. Nó thay cho việc tự train một teacher lớn, vốn mất 1–3 giờ trên T4 và không kịp.
- **Thể lệ:** S6 ("Active learning: ưu tiên gán những ảnh model kém tự tin"). Vẫn phải duyệt 100% (S7).

### D4. Duyệt, dựng v2/v3, train, nộp
- **Ai / khi nào:** cả 3 duyệt 13:40–14:20. Trung dựng zip và train. Colab B và C chạy song song 2 biến thể.
- **Làm gì:**
  1. Duyệt như B5, kiểm chéo như B6.
  2. Dựng `du_lieu_v2.zip`, sau đó v3, **giữ nguyên các video val**.
  3. Train bản chính và một biến thể (ví dụ khác epochs hoặc augmentation), nộp theo lịch, ghi `metrics.csv` sau mỗi lượt.
- **Kết quả:** số liệu trước/sau cho từng thay đổi. Ví dụ ✱ (số minh hoạ): "v1 → v2: thêm 180 ảnh xe xa, val mAP 0,41 → 0,52".
- **Vì sao:** **mỗi vòng chỉ đổi một thứ**, nhờ đó biết chắc điểm tăng là do thay đổi nào.
- **Thể lệ:** S6 ("Augmentation… phù hợp với điều kiện ảnh kiểm thử"), CTX §6 ("Chia val trước khi tăng cường").

**Lịch 10 lượt nộp:** #1 khoảng 12:15 · #2 khoảng 13:00 · #3 biến thể · #4–#5 khoảng 14:20–14:40 · #6–#7 trước 15:10 · **giữ 3 lượt dự phòng**. Mỗi lúc chỉ được một bài chờ chấm, nên không nộp dồn (S10).

---

## GIAI ĐOẠN E — Chốt bài (14:40 → 15:30)

### E1. Hai mốc đóng băng
- **14:40:** bắt đầu lượt train cuối cùng.
- **15:00:** **đóng băng dữ liệu**. Sau giờ này không sửa nhãn nữa.
- **Vì sao:** train cộng đóng gói cộng chấm mất khoảng 30 phút trở lên. Trễ mốc này thì có nguy cơ không kịp nộp trước 15:30.

### E2. Kiểm tái lập
- **Ai / khi nào:** Cường, 14:40–15:15.
- **Làm gì:**
  1. Mở notebook trên **runtime sạch** (tài khoản khác).
  2. Chạy lại với zip của bài dự kiến chọn final, seed 3, cho tới khi ra `submission.zip`.
  3. So val mAP với lần train gốc.
- **Kết quả:** biết chắc BTC chạy lại sẽ ra kết quả tương đương. Nếu lệch thì biết trước để giải thích (GPU không hoàn toàn deterministic).
- **Thể lệ:** S12 ("Chạy lại: trên Colab, runtime sạch, đến khi có submission.zip"; "Giữ lại: notebook đã huấn luyện và zip dữ liệu của bài chọn chung cuộc").

### E3. Chọn 2 bài final
- **Ai / khi nào:** Trung, **15:20** (đặt báo thức).
- **Làm gì:** trong tab "Nộp bài", mục "Bài đã nộp", chọn **F1** = bài có điểm public cao nhất và **F2** = bài có val mAP50-95 cao nhất. Chụp màn hình xác nhận.
- **Vì sao:**
  - Public chỉ có 40 ảnh, nên điểm nhiễu (CTX §4: "chỉ là tín hiệu tham khảo, đừng tối ưu riêng cho nó").
  - Val của mình có thể lệch theo kiểu khác.
  - Chọn mỗi thước đo một bài để chia rủi ro.
- **Thể lệ:** S10 ("chọn tối đa 2 bài… Không chọn thì hệ thống lấy 2 bài chấm gần nhất"), S14 (việc số 4).

### E4. Đóng gói bằng chứng
- **Ai:** Long.
- **Làm gì:** kiểm Drive `GSM_T3/` đủ mọi thứ: `raw/`, guideline các phiên bản, `labeling_log.csv`, `qa/`, `evidence/`, `datasets/*.zip` (có sha256), `runs/vN/` (notebook còn output, `run_report.json`, `submission.zip`). Lưu 2 bài final ở 2 nơi.
- **Thể lệ:** S7, S12 ("Chứng minh: cho xem ảnh thô, quá trình gán nhãn và cách kiểm tra QA").

---

## GIAI ĐOẠN F — Slide và thuyết trình (15:30 → 16:00, 50% điểm)

Khung slide đã dựng dần từ 12:20. Thứ tự slide theo yêu cầu của CTX §8:

1. **Bài toán và chiến lược:** "Data-centric: dữ liệu giống ảnh test, nhãn đủ và sát, vòng lặp sửa lỗi".
2. **Thu thập:** số địa điểm, số clip, số khung trước/sau lọc trùng (lấy từ `manifest.csv`), ảnh ví dụ.
3. **Label guideline:** bảng quy tắc kèm ảnh ca khó (xe bạc, bị che, xe nhỏ, sedan xanh).
4. **Gán nhãn:** pipeline 2 tầng (COCO + lọc màu) làm nhãn nháp, **người duyệt 100%**, số hộp đã thêm/sửa/xoá (từ `logs/review_stats.csv`).
5. **QA:** hiệu chuẩn 10 ảnh (IoU giữa các người), kiểm chéo 15% (tỉ lệ lỗi), ảnh trước/sau khi sửa.
6. **Huấn luyện:** cấu hình, lý do các lựa chọn (scale 0.5, không copy_paste…), val cố định theo địa điểm.
7. **Các vòng cải thiện:** bảng và biểu đồ v1 → v2 → v3: thay đổi gì, số ảnh, val mAP, điểm public. **Đây là slide quan trọng nhất.**
8. **Bài học và tái lập:** seed 3, zip và notebook đã lưu, kết quả chạy lại.

Hướng tới **giải Sáng tạo** (S13): kể câu chuyện "model tự tìm ảnh khó cho người duyệt" (active learning ở D3).

---

## Bảng đối chiếu thể lệ

| # | Điều thể lệ | Nguồn | Cách nhóm tuân thủ | Bằng chứng |
|---|---|---|---|---|
| 1 | YOLOv8n fine-tune từ yolov8n.pt, không đổi kiến trúc | S4, CTX §2 | Train trong Bước 2 của notebook chính thức, giữ `MODEL_NAME` | Notebook còn output |
| 2 | imgsz = 640 (đổi là bị từ chối) | S4, CTX §2 | Không sửa `IMGSZ`. Gán nháp ở 1280 chỉ là công cụ, không phải model nộp | TRAIN_ARGS trong notebook |
| 3 | 1 lớp id 0, nhãn YOLO chuẩn hoá [0,1] | S4, CTX §3 | `build_dataset.py` kiểm giống notebook, sai thì dừng | Báo cáo build |
| 4 | Seed = TEAM_ID = 3 | S12, CTX §0 | Điền `TEAM_ID = 3` | `run_report.json` |
| 5 | Nộp `submission.zip` có đúng 1 `model.onnx`, ≤ 25 MB, bằng ô cuối | S4, S8, CTX §5 | Chạy Bước 3 nguyên bản, không sửa | `runs/vN/` |
| 6 | Chỉ dùng ảnh thô đội tự thu | S7 | Chỉ dùng video tự quay | `raw/` + `manifest.csv` |
| 7 | Cấm bộ dữ liệu gán sẵn (COCO, Open Images, Roboflow, Kaggle) | S7 | Chỉ dùng **trọng số** COCO làm công cụ gán nháp, không dùng ảnh hay nhãn của chúng | Giải trình trên slide |
| 8 | Cấm nhãn do model sinh mà không kiểm tra lại | S7, S6 | Duyệt 100% ảnh, kể cả ảnh nền, không tự chấp nhận theo conf | `review_stats.csv`, `labeling_log.csv`, ảnh chụp CVAT |
| 9 | Cấm dùng nhãn của đội khác, cấm tìm ảnh hoặc nhãn của bộ test | S7 | Không trao đổi dữ liệu với đội khác | — |
| 10 | Ảnh internet: chờ BTC xác nhận | S7 | Không dùng (QĐ-06) | Guideline, Phụ lục A |
| 11 | Tối đa 10 lượt, mỗi lúc chỉ 1 bài chờ chấm | S10, CTX §5 | Lịch nộp ở D4, giữ 3 lượt dự phòng | `metrics.csv` |
| 12 | Chọn tối đa 2 bài final trước giờ đóng nộp | S10, S14 | Chốt lúc 15:20 | Ảnh chụp màn hình |
| 13 | Lưu ảnh gốc, nhật ký gán nhãn, ảnh chụp màn hình, ghi chú QA ngay từ đầu | S7, S14 | Drive `GSM_T3/` dựng ngay ở A3 | Toàn bộ thư mục |
| 14 | Giữ notebook và zip dữ liệu, tái lập được trên runtime sạch | S12, CTX §8 | Kiểm tái lập ở E2 | Kết quả chạy lại |
| 15 | Slide có thu thập, guideline, duyệt nhãn, huấn luyện, các vòng kèm số liệu trước/sau | CTX §8, S5 | Khung slide ở giai đoạn F | Slide |

✱ Các con số có dấu ✱ chỉ là ví dụ minh hoạ cách ghi, không phải kết quả thật.
