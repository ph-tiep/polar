# Chạy bản thiết kế Polar theo Digital Twin — 04/11/2026

Tải ZIP mới từ nhánh main của https://github.com/ph-tiep/polar (Code → Download ZIP), giải nén vào thư mục mới. Giữ lại thư mục results chứa CSI Phase 1 và môi trường Python của bạn. Nếu dùng bản clone: `git pull --ff-only origin main`; không reset/force các file đang sửa.

Chạy từ thư mục gốc dự án trong terminal VS Code. Các ví dụ dưới dùng PowerShell. Trên Linux/WSL, thay `.\.venv-phase2\Scripts\python.exe` bằng `.venv-phase2/bin/python`.

## 1. Môi trường

Nếu môi trường Phase 2 đã chạy được bản trước, dùng lại; bản sửa không thêm thư viện.

```powershell
py -3.12 -m venv .venv-phase2
.\.venv-phase2\Scripts\python.exe -m pip install --upgrade pip
.\.venv-phase2\Scripts\python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-phase2\Scripts\python.exe -m pip install -r requirements-phase2.txt
```

## 2. Dữ liệu Phase 1

Không cần chạy lại Phase 1 nếu đã có bộ CSI hợp lệ với `H_Bob.npy`, `H_Eve.npy`, `Metadata.npz`. Ví dụ đường dẫn:

`results/phase1-tx256-rx16/dataset/140GHz`

Nếu dữ liệu nằm chỗ khác, thay `--dataset-dir` bằng đường dẫn thật. H có dạng [Rx, Tx, subcarrier]. Không cắt/lặp anten để tạo cấu hình mới. Cách chạy Phase 1 giữ nguyên trong RUN_VSCODE.md.

## 3. Kiểm tra rồi chạy bộ chính

```powershell
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --dataset-dir results/phase1-tx256-rx16/dataset/140GHz --expected-tx 256 --expected-rx 16 --plan
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --dataset-dir results/phase1-tx256-rx16/dataset/140GHz --expected-tx 256 --expected-rx 16 --smoke --output-dir results/phase2-dt-smoke
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --dataset-dir results/phase1-tx256-rx16/dataset/140GHz --expected-tx 256 --expected-rx 16 --output-dir results/phase2-dt-review
```

Lệnh thứ ba mới tạo hình hiệu năng. Smoke chỉ dùng hai khung tại 0 dB, không dùng làm kết quả nghiên cứu. Thư mục đầu ra phải chưa tồn tại; đặt tên mới khi chạy lại.

Mặc định `configs/phase2_dt_review.json`: N256/K128, CRC11, 117 bit payload, CA-SCL L4/8, Eb/N0 từ -0,5 đến 4,5 dB bước 0,5; 10.000 khung/điểm. So sánh Fixed Standard, Fixed Crypto, DT Standard, DT Crypto. Modified decoder, SC, BP và NIST không chạy trong bộ chính. Thời gian phụ thuộc máy; không có cam kết hoàn tất trong một thời lượng cố định. CSV và lỗi từng khung được lưu sau mỗi SNR.

Các nhóm ánh xạ dùng chung cho mọi phương pháp và phủ toàn bộ subcarrier. Vị trí information/frozen được tính trước tại design Eb/N0=2 dB, không chọn theo kết quả BER. Cấu trúc DT dùng cận Bhattacharyya của kênh BPSK song song, không gọi là thuật toán giải mã mới hay mặc định tốt hơn đối chứng.

## 4. Mở kết quả

Trong `results/phase2-dt-review/N256_K128/`:

- `Fig2_Bob_L4_BER.png`, `Fig2_Bob_L8_BER.png`: hình riêng dễ đưa vào báo cáo; mỗi hình so cùng L.
- `Fig2_Bob_BER.png`, `Fig3_Bob_BLER.png`: các panel so mã hóa và cấu trúc.
- `Construction_BER.png`, `Construction_BLER.png`: Crypto cố định so Crypto theo DT, giữ nguyên decoder.
- `Fig5_Eve_full_BER.png`: cấu trúc DT, Bob đúng khóa + Eve sai 1/25/50/100%; cấu trúc cố định có file riêng `Fig5_Eve_fixed_full_BER.png`.
- `Eve_Controls_*.png`: đối chứng khóa đúng, frozen-only và đảo bit đầu ra Eve. Không gộp vào các hình Bob.
- `metrics.csv`, `paired_metrics.csv`, `frame_errors.npz`: số lỗi, độ bất định và dữ liệu ghép cặp.
- `construction.json`, `construction_arrays.npz`: vị trí mã, ánh xạ, cận thiết kế, thời gian thiết kế và thiết lập.
- `key_mismatch_audit.csv`: số bit khóa thực sự sai, kiểm tra riêng cả phần information và frozen.

Fig1 nằm ở thư mục run phía trên. Đường nối các điểm đo thật; không nội suy kết quả, không ép BER giảm hoặc đặt BER Eve=1. Điểm 0 lỗi được ghi dưới hình và trong CSV, không đổi thành sàn lỗi giả. 10.000 khung vẫn có thể chưa đủ ở vùng BER rất thấp; cần tăng frames_per_snr nếu muốn đo đuôi đường cong. Giữ đúng nhãn payload rate và code rate.

## 5. Khóa và chạy lặp lại

Alice/Bob được giả định có chung khóa bí mật. Runner tạo 32 byte ngẫu nhiên tại `private/shared_key.bin` trong thư mục run và dùng HMAC-SHA256 theo khung/ngữ cảnh để sinh chuỗi N bit. Đây không phải triển khai trích khóa CIR hay giao thức trao đổi khóa. Manifest chỉ ghi fingerprint và nonce; thư mục private được Git bỏ qua, không đưa khóa vào báo cáo hoặc ZIP công khai.

Muốn lặp lại số liệu: dùng cùng code, CSI, toàn bộ config (kể cả batch_size), file khóa cũ và nonce từ `RUN_CONFIG.json`:

```powershell
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --dataset-dir results/phase1-tx256-rx16/dataset/140GHz --expected-tx 256 --expected-rx 16 --output-dir results/phase2-dt-replay --key-file results/phase2-dt-review/private/shared_key.bin --key-nonce NONCE_TRONG_RUN_CONFIG
```

`NONCE_TRONG_RUN_CONFIG` phải được thay bằng chuỗi hex 32 ký tự, không phải khóa. Nếu không truyền file/nonce, chương trình tạo bộ mới. Các luồng khóa giữa các cấu trúc trong cùng run được giữ giống nhau để so sánh ghép cặp.

## 6. Hình 5 dải rộng và nhiều N/K

```powershell
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --config configs/phase2_dt_fig5.json --dataset-dir results/phase1-tx256-rx16/dataset/140GHz --expected-tx 256 --expected-rx 16 --output-dir results/phase2-dt-fig5
```

Bộ Fig5 dùng N128/K64 và N256/K64, 0–18 dB bước 2, L8 và 10.000 khung/SNR. K gồm CRC11 nên payload là 53 bit. P=100% lật toàn bộ chuỗi khóa N bit, không phải lật master key; BER phải lấy từ dữ liệu giải mã. Đây không phải tái lập SC/AWGN của bài cũ.

`configs/phase2_dt_expanded.json` có 12 cặp N/K, L4/8, 2.000 khung tại 8 SNR từ -1 đến 6 dB. Chạy bộ chính trước; bộ mở rộng có ngân sách lớn hơn tổng thể nhưng ít mẫu hơn mỗi điểm. Với Tx/Rx khác phải dùng CSI Phase 1 tương ứng và đổi expected-tx/rx.

## 7. Sai số dự đoán / Robust DT (tùy chọn)

Mặc định DT dùng chính gain Bob của kênh đánh giá để thiết kế: đây là giả thiết biết chính xác profile, chưa phải kiểm chứng khả năng dự đoán của Digital Twin. Không tự sinh sai số kênh để tạo kết quả đẹp.

Nếu có H_Bob dự đoán và các H_Bob tình huống độc lập có cơ sở, xuất profile:

```powershell
.\.venv-phase2\Scripts\python.exe prepare_design_profiles.py --evaluation-bob results/phase1-tx256-rx16/dataset/140GHz/H_Bob.npy --nominal-bob DUONG_DAN_H_BOB_DU_DOAN.npy --scenario-bob DUONG_DAN_H_BOB_TINH_HUONG.npy --output results/design_profiles.npz
.\.venv-phase2\Scripts\python.exe main_phase2_crypto_q1.py --config configs/phase2_dt_robust.json --design-profiles results/design_profiles.npz --dataset-dir results/phase1-tx256-rx16/dataset/140GHz --expected-tx 256 --expected-rx 16 --output-dir results/phase2-dt-robust
```

Thay các đường dẫn viết hoa bằng file thật; có thể lặp `--scenario-bob` cho nhiều tình huống. Các CSI phải cùng lưới tần số, thứ tự và kích thước anten. Công cụ chiếu các kênh lên beam phát đang dùng của kênh đánh giá, giữ giả thiết MRC/coherent hiện tại. Nó chỉ khảo sát sai lệch profile dùng để thiết kế mã, không mô phỏng sai số beamforming hoặc LLR của bộ thu.

Robust chọn K vị trí có cận xấu nhất nhỏ nhất qua nominal và các scenario đã cung cấp. Không có scenario thì báo lỗi. Đây là đề xuất để đo/so sánh, chưa xác nhận tính mới hay cải thiện. Tập thiết kế cần tách hợp lý với tập đánh giá khi nghiên cứu khả năng tổng quát hóa.

## 8. Vẽ lại và kiểm thử

```powershell
.\.venv-phase2\Scripts\python.exe replot_phase2.py --run-dir results/phase2-dt-review --output-dir results/phase2-dt-redraw
.\.venv-phase2\Scripts\python.exe -m unittest discover -s tests -p "test_phase2.py" -v
.\.venv-phase2\Scripts\python.exe -m unittest discover -s tests -p "test_construction.py" -v
```

Replot chỉ đọc run đã hoàn tất, giữ CSV nguyên byte và không sao chép khóa riêng. Bộ kiểm thử dùng fixture nhỏ, không phải kết quả nghiên cứu. Các cấu hình legacy phase2_review/dense/expanded/fig5 vẫn dùng được khi truyền rõ --config; chúng đánh giá cấu trúc cố định và decoder cũ, không chứa đóng góp DT construction. Phần system model vẫn để sau theo yêu cầu.
