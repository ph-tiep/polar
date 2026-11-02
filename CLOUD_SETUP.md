# Cloud Phase 1: chạy và đánh giá

Checkout dùng trực tiếp tại `/workspace/polar`, không cần Git worktree. Runtime là Python 3.12.

## Cài đặt

```bash
cd /workspace/polar
bash scripts/cloud/install.sh
export MPLCONFIGDIR=/workspace/.polar-setup/matplotlib XDG_CACHE_HOME=/workspace/.polar-setup/cache
export MPLBACKEND=Agg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2
```

Hai virtualenv: `.polar-venv` cho PyTorch 2.12.1 CPU/Phase 2, `.polar-phase1-venv` cho TensorFlow 2.21.0 và Sionna RT 2.0.1. Dr.Jit cần cache `/home/agent/.drjit` có thể ghi.

## Chạy bản đánh giá riêng

```bash
/workspace/.polar-phase1-venv/bin/python scripts/cloud/run-phase1.py --layout outdoor-review --mode full
```

Launcher tạo thư mục mới dưới `results/` trong repository, ghi `RUN_CONFIG.json` và từ chối ghi đè một thư mục đã tồn tại. Có thể chỉ định `--output-dir /workspace/polar-results/my-new-run`.

**outdoor-review thay vị trí BS từ [0,0,25] thành [20,20,25] m**, giữ nguyên Bob=[45,10,1.5] và Eve=[42,12,1.5]. Tọa độ BS cũ nằm dưới mái z=26 m. Đây là kịch bản đánh giá mới, không phải bản tái lập hình học gốc.

`--layout original` giữ nguyên hình học gốc; kiểm thử thực tế vẫn không có tia và giờ báo ValueError rõ ràng trước khi xuất dữ liệu. Không tạo CSI giả.

Đầu ra nằm trong từng thư mục lần chạy: `dataset/140GHz/`, `figures/`, `logs/`, `RUN_CONFIG.json`. Không ghi đè CIR hay các hình Phase 2 đã có trong Git.

Để xác minh và tạo thêm hình đối chiếu:

```bash
/workspace/.polar-venv/bin/python scripts/cloud/review-phase1.py /workspace/polar-results/TEN-THU-MUC-LAN-CHAY
```

## Kết quả cloud đã xác minh

Bản full thành công: `/workspace/polar-results/phase1-outdoor-review-003`. Một triệu mẫu/nguồn, seed 2026, 140 GHz, 2 GHz, 512 subcarrier, BS 16×16, UE 4×4, reflection bật và max depth=3. Có 23 đường tới Bob và 24 đường tới Eve. CSI `(16,256,512)` hữu hạn và khác 0. MATLAB khớp NumPy; tổng Fourier trực tiếp từ CIR khớp CSI (sai số tương đối cực đại <0,02% trên cặp anten kiểm tra).

PDP và thống kê loại padding delay=-1. CFR dùng offset baseband vì CIR đã chứa pha sóng mang, giữ nguyên các trục anten và không nhân bản anten. Tỷ lệ công suất tia tới sớm nhất không được coi là nhãn LOS. Sửa khởi tạo thư viện native, phiên bản RT và tên trường MATLAB dài.

Bốn kiểm thử hồi quy đạt:

```bash
/workspace/.polar-phase1-venv/bin/python -m unittest discover -s tests -p test_phase1.py -v
```

Bộ Crypto-Polar 5 decoder đạt từ trước. Phase 2 chưa được chạy với dataset mới; cần đánh giá hình học và kết quả Phase 1 trước. Công suất kênh trong báo cáo là dB của gain, không phải dBm hoặc phép đo thực nghiệm.

Cấu hình cloud được lưu dưới dạng nháp; Git và việc lưu cấu hình không tự publish môi trường.

## Chạy bản sửa theo góp ý

Xem [RUN_VSCODE.md](RUN_VSCODE.md) và [ADVISOR_CHANGES.md](ADVISOR_CHANGES.md). Các kết quả cloud nêu trên là lịch sử Phase 1; không phải kết quả của thí nghiệm Phase 2 mới.
