# Thiết lập cloud và kết quả chạy

Repo được dùng trực tiếp tại `/workspace/polar`; không cần tạo Git worktree.
Các đường dẫn trong script dành cho môi trường cloud này, không phải Git Bash trên Windows.

## Các file đã bổ sung

- `munich/munich.xml`: XML người dùng cung cấp, giống bản trong `phase1_digital_twin/`.
- `munich/meshes/`: 2.367 file PLY giải nén từ `meshes.rar`, khớp đầy đủ tham chiếu của XML.
- `scripts/cloud/install.sh`: cài dependency và kiểm tra môi trường.
- `scripts/cloud/run-phase1.py`: launcher khởi tạo Sionna RT trước TensorFlow, cung cấp metadata phiên bản RT và đặt thư mục kết quả ngoài repo.

## Cài đặt và chạy

```bash
cd /workspace/polar
bash scripts/cloud/install.sh
export MPLCONFIGDIR=/workspace/.polar-setup/matplotlib
export XDG_CACHE_HOME=/workspace/.polar-setup/cache
export MPLBACKEND=Agg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2
/workspace/.polar-phase1-venv/bin/python scripts/cloud/run-phase1.py
```

Python 3.12 được dùng với hai virtualenv riêng:

- Crypto-Polar/Phase 2: PyTorch 2.12.1 CPU, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8.
- Phase 1: TensorFlow 2.21.0 và Sionna RT 2.0.1.

Dr.Jit cần thư mục cache `/home/agent/.drjit` có thể ghi. Nếu sandbox chặn cache này, chạy qua quyền thực thi cloud phù hợp; không tắt kiểm tra TLS hay checksum. Không chạy trực tiếp `main_phase1.py` trong runtime này vì thứ tự nạp thư viện native có thể gây crash.

Kết quả Phase 1 được đặt tại `/workspace/polar-results/phase1/`, tránh ghi đè CIR và hình đã có trong Git.

## Kết quả đã xác minh

- Bộ kiểm thử `fec/polar/test_crypto.py` đạt với cả 5 bộ giải mã; Bob BER bằng 0 trong kiểm thử không nhiễu.
- Import Phase 1 và Phase 2 thành công.
- Nạp cảnh Munich, ánh xạ vật liệu, bố trí anten và ray tracing CPU chạy được.
- Phase 1 ở cả debug (200.000 mẫu) và full (1.000.000 mẫu) trả về 0 multipath hợp lệ. Sau đó thất bại tại `phase1_digital_twin/validation_metrics.py`, hàm `los_ratio()`, do truy cập `power[0]` khi mảng rỗng.
- Kiểm tra hình học: đường LOS từ BS đến Bob và Eve bị che; giao điểm đầu tiên cách BS lần lượt khoảng 1,217 m và 1,086 m. Điều này chưa giải thích đầy đủ việc không có tia phản xạ.
- Chưa xuất dataset CSI mới; Phase 2 đầy đủ chưa chạy vì thiếu CSI hợp lệ `H_Bob.npy` và `H_Eve.npy`.

Cần kiểm tra vị trí node/cấu hình đường truyền và xử lý trường hợp không có tia trong một thay đổi mã nguồn riêng. Không thay cảnh hoặc tạo CSI giả để bỏ qua lỗi.

## Log trong phiên cloud đã chạy

- `/workspace/.polar-setup/phase1-meshes-run.log`
- `/workspace/.polar-setup/phase1-full-run.log`
- `/workspace/.polar-setup/scene-diagnosis.log`

Log và các virtualenv nằm ngoài repo. Cấu hình cloud đã được lưu ở dạng nháp; thao tác Git không publish môi trường cloud.
