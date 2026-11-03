# Rà soát Hình 5 và luồng xuất kết quả

## Nguyên nhân hình chỉ có chấm

`--smoke` chủ đích dùng `ebno_db=[0]` và hai khung. Bản trước vẫn gọi hàm xuất
các hình nghiên cứu nên người dùng có thể mở nhầm. Với một điểm x duy nhất,
Matplotlib chỉ vẽ marker và tự giãn trục quanh 0. Không có đoạn dữ liệu thứ hai
để nối thành đường; không được nội suy một đường từ một điểm.

Đã sửa: smoke chỉ xuất CSV/JSON và SMOKE_ONLY.txt; tách tên thư mục mặc định
smoke/review. Hàm vẽ lại từ chối smoke, run chưa hoàn tất và dữ liệu thiếu/đúp SNR.

## Zero-error và kiểu vẽ

Trước đây cận trên 95% ở các điểm 0 lỗi được đặt trên cùng trục như tam giác.
Cận bảo thủ có thể cao hơn BER đo tại SNR thấp hơn, gây hiểu nhầm có sàn lỗi.
Giờ số liệu 0 giữ nguyên trong CSV và được ghi dưới hình; cận ở bảng riêng.
Các điểm đo khác 0 nối bằng đường có màu/nét/marker phân biệt. Các subplot cùng
phép so sánh dùng cùng thang Y. Không làm trơn, ép xu hướng đơn điệu hay gán BER giả.

`replot_phase2.py` tạo bộ hình mới từ CSV/JSON đã chạy, không sửa nguồn và không
mô phỏng lại. Manifest của việc vẽ lại lưu hash CSV để kiểm tra.

## Định nghĩa Hình 5 theo bài tham chiếu và comment

- K dài N bit, gồm K1 XOR information và K2 gán frozen. Chương trình đóng gói
  K1 trước K2 rồi ánh xạ vào information/frozen positions; khác cách lưu chỉ số
  không làm đổi phạm vi lật toàn bộ N bit.
- Alice và Bob dùng cùng K. Eve dùng K XOR mask có đúng round(p*N) bit 1,
  với p=1%,25%,50%,100%. Vị trí lật ngẫu nhiên trong từng khung, số lượng cố định.
- Hình chính có 5 đường. Eve khóa đúng là đối chứng riêng. Frozen-only chỉ là
  đối chứng khi chủ động chọn, không thay thế phép thử toàn bộ K.
- Cùng CA-SCL và cùng L cho Bob/Eve ở Fig5. Standard/Crypto cùng L và CRC ở Fig2/3.
- `configs/phase2_fig5.json`: hai N/K theo ví dụ tham chiếu, quét 0..18 dB;
  vẫn dùng CSI mới và CA-SCL theo phạm vi đã chốt. Không đưa SC/NIST trở lại.

## Những giới hạn không thể xử lý bằng thay kiểu đường

1. Theo phép biến đổi Polar hiện dùng, vector toàn 1 sau nhân G_N có trọng số 1.
   Do đó toàn bộ keystream bị bù có thể vẫn dẫn tới giải mã tốt. Không được sửa
   khóa chỉ ở frozen để ép đường 100% lên BER gần 1 rồi gọi đó là toàn bộ K.
2. BER gần 1 có thể là plaintext bị bù; không phải bằng chứng perfect secrecy.
   Phép thử BER không đủ chứng minh an toàn mật mã tổng quát.
3. Bài tham chiếu dùng SC/AWGN, khóa lượng tử hóa CIR; pipeline hiện tại dùng
   CA-SCL/CRC, CSI ray tracing và keystream SHAKE gắn digest CSI. Kết quả và giả
   thiết không đồng nhất. Phạm vi generator không được đổi ngầm trong lần sửa này.
4. Hai nghìn khung không đủ kết luận ở vùng rất ít lỗi. Có config nhiều điểm/mẫu
   hơn và lỗi từng khung cho người dùng tự chạy/phân tích.
5. Modified có retries nên phải so cả chất lượng và chi phí. Thêm các đối chứng
   chỉ tăng list, chỉ damping, chỉ flip; chưa kết luận thành phần nào tốt trước
   khi có kết quả. `paired_metrics.csv` giữ đếm khung lỗi riêng/chung, không tự
   tuyên bố tương đương hay thắng thua thống kê.

System model tiếp tục để sau theo yêu cầu người dùng. Các kiểm tra kỹ thuật không
thay thế thí nghiệm đầy đủ và không được đưa vào báo cáo như kết quả nghiên cứu.
