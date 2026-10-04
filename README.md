# GreenSM Data Centric Challenge — L2B Nhóm 04 (TEAM_ID = 3)

Thành viên: Hoàng Mạnh Cường · Hoàng Văn Long · Trịnh Nam Trung

## Đọc gì trước
1. `guideline/GUIDELINE.html`: **guideline gán nhãn** (có hình), bản duy nhất. Clone repo rồi mở file bằng trình duyệt (ảnh nằm ở `guideline/img/`, đừng tách file ra riêng).
2. [docs/KE-HOACH-THUC-THI.md](docs/KE-HOACH-THUC-THI.md) — các bước thực thi, ai làm gì, mốc giờ, căn cứ thể lệ.
3. [docs/CHIEN-LUOC-GREENSM.md](docs/CHIEN-LUOC-GREENSM.md) — lý do kỹ thuật của từng lựa chọn.
4. [tools/README_TOOLS.md](tools/README_TOOLS.md) — công cụ: cắt khung → gán nháp → CVAT → dựng zip.

## Thư mục
| Thư mục | Nội dung |
|---|---|
| `guideline/` | Guideline + hình minh hoạ + script dựng hình |
| `docs/` | Chiến lược, kế hoạch thực thi |
| `tools/` | `extract_frames.py`, `prelabel.py`, `make_cvat_import.py`, `build_dataset.py`, `review_stats.py` |
| `notebooks/` | Notebook chính thức của BTC (`train_and_export.ipynb`, **không sửa ô Bước 3**) |
| `logs/` | `labeling_log.csv` (chỉ ảnh `exclude`), `review_stats.csv`, `metrics.csv`, `datasets.csv` — bằng chứng cho slide và tái lập |
| `qa/` | Biên bản hiệu chuẩn, kiểm chéo, phân tích lỗi |
| `evidence/` | Ảnh chụp màn hình CVAT trước/sau sửa, bảng điểm |

File lớn (video gốc, khung hình, zip dữ liệu, `.pt`, `submission.zip`) để trên **Google Drive `GSM_T3/`**, không đưa lên git. Ghi sha256 của zip dữ liệu trong `logs/datasets.csv`.

## Mốc cứng
14:40 bắt đầu lượt train cuối · 15:00 đóng băng dữ liệu · 15:20 chốt 2 bài final · 15:30 đóng nộp.
