# Hướng dẫn Phase 1 và các thí nghiệm legacy

**Bản mới: đọc [RUN_DT_VSCODE.md](RUN_DT_VSCODE.md) để chạy thiết kế Polar theo Digital Twin.** Mặc định CLI hiện là `phase2_dt_review.json`. Các lệnh Phase 2 bên dưới được ghi rõ config legacy để tái hiện nhóm so sánh cũ; Phase 1 giữ nguyên.

Chạy các lệnh dưới đây từ thư mục gốc `polar` trong terminal PowerShell của VS Code.
Mã đã kiểm tra trên Linux/Python 3.12 CPU. Chưa kiểm chứng trực tiếp trên Windows;
nếu Sionna RT/Dr.Jit không tải được trên Windows, dùng VS Code với WSL2 (Ubuntu).
Các môi trường Phase 1 và Phase 2 được tách để tránh xung đột TensorFlow/PyTorch.

## 1. Lấy mã và tạo môi trường

Nếu đang có file sửa chưa commit, lưu lại trước khi cập nhật. Không dùng reset/force.
Nếu bạn lấy mã bằng Download ZIP thay vì git clone, tải ZIP mới từ nhánh main; giữ riêng thư mục results và các môi trường Python trước khi thay mã. Lệnh git pull bên dưới áp dụng cho bản clone có .git.

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
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --config configs/phase2_review.json --dataset-dir results/phase1-tx256-rx16/dataset/140GHz --expected-tx 256 --expected-rx 16 --plan
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --config configs/phase2_review.json --dataset-dir results/phase1-tx256-rx16/dataset/140GHz --expected-tx 256 --expected-rx 16 --smoke --output-dir results/phase2-smoke-256x16
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --config configs/phase2_review.json --dataset-dir results/phase1-tx256-rx16/dataset/140GHz --expected-tx 256 --expected-rx 16 --output-dir results/phase2-review-256x16
```

`--plan` chỉ in cấu hình; `--smoke` dùng 2 khung tại 0 dB để kiểm tra nối các bước,
chỉ xuất CSV/JSON và `SMOKE_ONLY.txt`, **không xuất hình hiệu năng**. Không dùng smoke cho báo cáo. Lệnh thứ ba mới chạy cấu hình đánh giá.
Nếu đã có CSI từ lần chạy sửa trước, có thể chỉ đường dẫn tới bộ CSI đó, nhưng
phải có `Metadata.npz` phù hợp. Hình cũ hoặc CIR riêng lẻ không thay thế được H.

Cấu hình legacy `configs/phase2_review.json` (chọn tường minh bằng --config):

- N=256, K=128 **bao gồm CRC11**, payload=117; R_code=0.5, R_payload=117/256.
- CA-SCL L=4,8,16 cho cả Standard và Crypto. Không có SC/BP trong thí nghiệm mới.
- Decoder Modified: L ban đầu 8, mở rộng tối đa 16, tối đa 4 lần lật LLR.
- `decoder_ablation: true`: thêm ba đối chứng chỉ tăng list, chỉ damping, chỉ flip để phân biệt đóng góp.
- 2.000 khung mỗi điểm Eb/N0 từ -1 đến 6 dB; batch 32, seed 2026, CPU 2 luồng.
- Eve dùng CA-SCL L=8; có khóa đúng, sai 1%,25%,50%,100% bit của keystream dẫn xuất.
- Hình 5 chính dùng `full`: K dài N bit gồm cả khóa che thông tin và khóa frozen, theo định nghĩa của bài tham chiếu. Bob dùng K giống Alice; Eve lật 1%,25%,50%,100% bit K. Có thể đặt `mismatch_scopes: ["frozen"]` cho đối chứng phụ, với phần khóa thông tin giả định biết đúng; không dùng đối chứng này thay Hình 5 chính.
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
- `Fig5_Eve_full_*`: đúng 5 đường Bob khóa đúng + Eve p=1%,25%,50%,100%. Eve khóa đúng được đưa sang `Eve_Controls_*`. `Fig5_Eve_frozen_*` chỉ xuất khi chọn đối chứng frozen.
- `Eve_Controls_*`: Standard tại Eve, Crypto khóa đúng, và đối chứng frozen-only
  bằng hard decision + biến đổi Polar ngược. Đây là các luật giải mã khác nhau.
- `metrics.csv`: BER/BLER, số lỗi/khung/bit, Wilson CI cho BLER, sai số chuẩn BER
  tính theo khung, thời gian, số lượt retry, L tối đa và payload goodput.
- `frame_errors.npz`: lỗi bit theo từng khung, trục `[SNR, curve, frame]`, cho phân tích ghép cặp.
- `paired_metrics.csv`: chênh lệch BER/BLER và đếm khung lỗi riêng/chung; chênh lệch = candidate - reference, âm là candidate ít lỗi hơn. Sai số chuẩn bằng 0 không chứng minh tương đương khi chưa quan sát lỗi.
- `diagnostics.json`, `Key_Perturbation_and_Weight.*`: thay một bit master key
  với cùng plaintext/CSI/frame ID; phổ trọng số dùng cùng payload và CRC.
- `Decoding_Cost_and_Payload_Goodput.*`: thời gian giải mã trung bình theo batch
  và payload goodput; không phải độ trễ đơn khung hoặc secret capacity.

BER/BLER bằng 0 vẫn ghi 0 trong CSV. Trên hình log, ghi rõ các SNR có 0 lỗi ở chú thích dưới trục; không vẽ cận trên thành điểm BER. `zero_error_observations.csv` lưu số khung/bit và cận trên 95% để tham khảo. Các đường nối các điểm đo khác 0, không làm trơn hoặc ngoại suy phần chưa đo.

`figures/` cũ trong Git là lịch sử, không phải đầu ra bản mới. Không dùng lẫn hai
bộ hình. NIST đã bỏ khỏi luồng chạy; phần system model chưa được soạn/sửa.

## 6. Kiểm tra kỹ thuật khi cần

```powershell
.\.venv-phase2\Scripts\python.exe -m unittest discover -s tests -p test_phase2.py -v
.\.venv-phase1\Scripts\python.exe -m unittest discover -s tests -p test_phase1.py -v
```

Bộ kiểm tra dùng dữ liệu nhỏ có nhãn fixture, không phải kết quả khoa học.


## 7. Vẽ lại kết quả đã chạy, không cần chạy lại Phase 1/2

Dùng thư mục **review**, không chọn thư mục **smoke**. Công cụ sẽ từ chối smoke,
lần chạy chưa hoàn tất, hoặc CSV chỉ có một SNR.

```powershell
.\.venv-phase2\Scripts\python.exe replot_phase2.py --run-dir results/phase2-review-256x16 --output-dir results/phase2-review-256x16-redraw
```

Chỉ cần RUN_CONFIG.json, metrics.csv và diagnostics.json từ lần chạy cũ; không
cần H/CIR để vẽ lại các hình hiệu năng. Không ghi đè file gốc. CSV được chép nguyên
byte, kèm SHA256 trong REPLOT_INFO.json. Fig1 kênh giữ ở thư mục lần chạy gốc.

## 8. Chạy Hình 5 theo bố cục bài tham chiếu

```powershell
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --config configs/phase2_fig5.json --dataset-dir results/phase1-tx256-rx16/dataset/140GHz --expected-tx 256 --expected-rx 16 --output-dir results/phase2-fig5-256x16
```

Config này có N=128,K=64 và N=256,K=64; Eb/N0=0,2,...,18 dB; 10.000 khung/SNR;
Bob/Eve cùng CA-SCL L=8, CRC11; không chạy Modified/ablation trong bộ Fig5 riêng.
Đây là cấu hình kiểm chứng mới trên CSI Phase 1; bài cũ dùng SC/AWGN và không
nêu CRC, nên không gọi là tái lập nguyên bản số liệu cũ. K/N là rate gồm CRC;
payload tương ứng 53 bit trong cả hai trường hợp. Số bit lật làm tròn gần nhất,
ví dụ 1% của N=128 là 1/128 bit, của N=256 là 3/256 bit; lưu tỷ lệ thực tế trong CSV.

Nội dung khóa K gồm cả information key và frozen key. p=100% là phần bù K, khác
khóa ngẫu nhiên độc lập. Các đường phải theo kết quả thực, không ép BER Eve về
0,5 hoặc 1. Ghi nhận hạn chế này khi phân tích báo cáo.

Muốn khảo sát dày vùng chuyển tiếp Bob/decoder, dùng `--config configs/phase2_dense.json`:
SNR -1 đến 6 dB, bước 0,5 dB và 10.000 khung/SNR. Tổng khung trên mỗi đường tăng
9,375 lần so review 2.000 khung tại 8 SNR; chọn ngân sách phù hợp máy. Config dày
là tùy chọn để bạn tự chạy, không phải kết quả đã có. `decoder_ablation: false`
có thể dùng khi chỉ muốn chạy phép so sánh chính. Không tăng N/K/TxRx chỉ để làm
đường đẹp; dùng mỗi thay đổi để trả lời một câu hỏi đánh giá rõ ràng.
