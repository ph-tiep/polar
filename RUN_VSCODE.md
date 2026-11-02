# Chạy bản sửa theo góp ý của Thầy

Chạy các lệnh dưới đây từ thư mục gốc `polar` trong terminal PowerShell của VS Code.
Mã đã kiểm tra trên Linux/Python 3.12 CPU. Chưa kiểm chứng trực tiếp trên Windows;
nếu Sionna RT/Dr.Jit không tải được trên Windows, dùng VS Code với WSL2 (Ubuntu).
Các môi trường Phase 1 và Phase 2 được tách để tránh xung đột TensorFlow/PyTorch.

## 1. Lấy mã và tạo môi trường

Nếu đang có file sửa chưa commit, lưu lại trước khi cập nhật. Không dùng reset/force.

```powershell
git pull --ff-only origin main
py -3.12 -m venv .venv-phase2
.\.venv-phase2\Scripts\python.exe -m pip install --upgrade pip
.\.venv-phase2\Scripts\python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-phase2\Scripts\python.exe -m pip install -r requirements-phase2.txt
py -3.12 -m venv .venv-phase1
.\.venv-phase1\Scripts\python.exe -m pip install --upgrade pip
.\.venv-phase1\Scripts\python.exe -m pip install -r requirements-phase1.txt
```

Trên Linux/WSL, dùng `python3.12 -m venv ...` thay `py -3.12`, và
`.venv-phase1/bin/python` / `.venv-phase2/bin/python` thay đường dẫn `Scripts`.
Trong VS Code, chọn đúng interpreter cho từng phase. Không cần activate môi trường
nếu gọi đầy đủ đường dẫn Python như các lệnh này.

## 2. Phase 1: tạo CSI thật, lưu riêng từng cấu hình anten

Ví dụ **outdoor-review** dùng BS=[20,20,25], Bob=[45,10,1.5], Eve=[42,12,1.5].
Đây là lựa chọn hình học tường minh, khác BS gốc [0,0,25]. Đổi thành
`--layout original` nếu cần kiểm tra BS gốc; trường hợp không có tia sẽ báo lỗi,
không tự đổi BS hoặc tạo dữ liệu thay thế.

```powershell
.\.venv-phase1\Scripts\python.exe scripts/cloud/run-phase1.py --layout outdoor-review --mode full --tx-array 16 16 --rx-array 4 4 --output-dir results/phase1-tx256-rx16
.\.venv-phase1\Scripts\python.exe scripts/cloud/run-phase1.py --layout outdoor-review --mode full --tx-array 8 16 --rx-array 2 2 --output-dir results/phase1-tx128-rx4
```

Hai lệnh là hai thí nghiệm riêng, có thể chạy cấu hình đầu trước để kiểm tra.
Mỗi thư mục có `RUN_CONFIG.json` (phải có `status: completed`) và:

- `dataset/140GHz/H_Bob.npy`, `H_Eve.npy`: lần lượt kích thước `[16,256,512]` hoặc `[4,128,512]`.
- `Metadata.npz`, CIR và bản MATLAB.
- `figures/` và `logs/validation.json`.

Không cắt hoặc lặp anten trong CSI để giả lập cấu hình mới. Không ghi đè thư mục
đã tồn tại: dùng tên mới cho lần chạy lại.

Kiểm tra Fourier/MATLAB và vẽ hình CSI trực tiếp, không chạy ray tracing lại:

```powershell
.\.venv-phase2\Scripts\python.exe scripts/cloud/review-phase1.py results/phase1-tx256-rx16
```

## 3. Xem kế hoạch, kiểm tra nhanh, rồi chạy Phase 2

```powershell
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --dataset-dir results/phase1-tx256-rx16/dataset/140GHz --expected-tx 256 --expected-rx 16 --plan
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --dataset-dir results/phase1-tx256-rx16/dataset/140GHz --expected-tx 256 --expected-rx 16 --smoke --output-dir results/phase2-smoke-256x16
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --dataset-dir results/phase1-tx256-rx16/dataset/140GHz --expected-tx 256 --expected-rx 16 --output-dir results/phase2-review-256x16
```

`--plan` chỉ in cấu hình; `--smoke` dùng 2 khung/SNR để kiểm tra nối các bước,
**không dùng kết quả smoke cho báo cáo**. Lệnh thứ ba mới chạy cấu hình đánh giá.
Nếu đã có CSI từ lần chạy sửa trước, có thể chỉ đường dẫn tới bộ CSI đó, nhưng
phải có `Metadata.npz` phù hợp. Hình cũ hoặc CIR riêng lẻ không thay thế được H.

Cấu hình mặc định `configs/phase2_review.json`:

- N=256, K=128 **bao gồm CRC11**, payload=117; R_code=0.5, R_payload=117/256.
- CA-SCL L=4,8,16 cho cả Standard và Crypto. Không có SC/BP trong thí nghiệm mới.
- Decoder Modified: L ban đầu 8, mở rộng tối đa 16, tối đa 4 lần lật LLR.
- 2.000 khung mỗi điểm Eb/N0 từ -1 đến 6 dB; batch 32, seed 2026, CPU 2 luồng.
- Eve dùng CA-SCL L=8; có khóa đúng, sai 1%,25%,50%,100% bit của keystream dẫn xuất.
- Hai phạm vi sai khóa tách riêng: `full` và `frozen`. Trong `frozen`, phần khóa
  che thông tin được giả định biết đúng. Không gọi đây là tỷ lệ biết master key.
- `eve_snr: physical` giữ chênh lệch gain Bob/Eve. Có thể tạo bản config riêng với
  `equal-average` để kiểm tra cùng SNR trung bình; chạy vào thư mục khác.

2.000 khung là ngân sách ban đầu, không đảm bảo phân giải BER rất thấp. Xem số lỗi
và khoảng bất định trước khi tăng `frames_per_snr`. Không suy ra lợi ích nếu
chênh lệch còn nhỏ so với nhiễu Monte Carlo.

## 4. Mở rộng N, K, R và Tx/Rx

`configs/phase2_expanded.json` có 12 trường hợp: N=128,256,512,1024; mỗi N có
K/N=1/4,1/2,3/4. R được tính từ N,K, không đặt một giá trị R mâu thuẫn với chúng.
Payload thực luôn là K trừ số bit CRC. Có thể giữ vài mục trong `cases` để chạy từng đợt.

```powershell
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --config configs/phase2_expanded.json --dataset-dir results/phase1-tx128-rx4/dataset/140GHz --expected-tx 128 --expected-rx 4 --output-dir results/phase2-expanded-128x4
```

Chạy tương tự với CSI 256x16 để so anten. Đây là tập thí nghiệm lớn, đặc biệt
N=1024 và decoder có retries; không cần chạy toàn bộ ngay lần đầu.

Để khảo sát từng thành phần decoder, sao chép config rồi lần lượt đặt
`damping=1`, `retry_damping=1`, `expand_list=false`, hoặc `max_flip_trials=0`.
Giữ các mục còn lại và seed, chạy vào thư mục khác. `modified.enabled=false`
chỉ chạy đối chiếu Standard/Crypto và Eve. Nếu bật mở rộng list, config bắt buộc
có baseline cùng L tối đa. Không mặc định decoder chỉnh sửa sẽ tốt hơn.

## 5. Xem kết quả

Trong từng thư mục Phase 2:

- `RUN_CONFIG.json`: trạng thái, cấu hình thực tế, phiên bản, commit, dấu hiệu mã
  chưa commit, metadata Phase 1 và SHA256 các file đầu vào.
- `Fig1_RayTraced_Channel.png/pdf`: gain và tương quan tính từ H thật. Fig1 được
  tạo ở Phase 2; Phase 1 có hình kiểm tra CSI/PDP riêng.
- `N..._K.../Fig2_Bob_BER.*`, `Fig3_Bob_BLER.*`: mỗi hình con một L, chỉ so Bob.
- `Decoder_BER/BLER.*`: CA-SCL cơ sở so Modified, tách khỏi so mã hóa.
- `Fig5_Eve_full_*`, `Fig5_Eve_frozen_*`: các mức sai khóa theo Eb/N0.
- `Eve_Controls_*`: Standard tại Eve, Crypto khóa đúng, và đối chứng frozen-only
  bằng hard decision + biến đổi Polar ngược. Đây là các luật giải mã khác nhau.
- `metrics.csv`: BER/BLER, số lỗi/khung/bit, Wilson CI cho BLER, sai số chuẩn BER
  tính theo khung, thời gian, số lượt retry, L tối đa và payload goodput.
- `diagnostics.json`, `Key_Perturbation_and_Weight.*`: thay một bit master key
  với cùng plaintext/CSI/frame ID; phổ trọng số dùng cùng payload và CRC.
- `Decoding_Cost_and_Payload_Goodput.*`: thời gian giải mã trung bình theo batch
  và payload goodput; không phải độ trễ đơn khung hoặc secret capacity.

BER/BLER bằng 0 vẫn ghi 0 trong CSV. Trên trục log, tam giác biểu diễn cận trên
95% khi không quan sát lỗi; với BER dùng cận bảo thủ BER <= BLER, không giả định
các bit trong một khung độc lập. Không nối cận này như một BER đo được.

`figures/` cũ trong Git là lịch sử, không phải đầu ra bản mới. Không dùng lẫn hai
bộ hình. NIST đã bỏ khỏi luồng chạy; phần system model chưa được soạn/sửa.

## 6. Kiểm tra kỹ thuật khi cần

```powershell
.\.venv-phase2\Scripts\python.exe -m unittest discover -s tests -p test_phase2.py -v
.\.venv-phase1\Scripts\python.exe -m unittest discover -s tests -p test_phase1.py -v
```

Bộ kiểm tra dùng dữ liệu nhỏ có nhãn fixture, không phải kết quả khoa học.
