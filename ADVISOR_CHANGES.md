# Đối chiếu góp ý và thay đổi mã

Tài liệu này mô tả cách kiểm chứng và giới hạn của chương trình, chưa kết luận
Crypto-Polar/decoder cải thiện hiệu quả. Phần system model để sau khi người dùng cho phép.

| Góp ý | Cách xử lý | Kết quả cần người dùng chạy và phân tích |
|---|---|---|
| Phân biệt Crypto-Polar và decoder | Mã hóa có khóa; CA-SCL cơ sở; Modified CA-SCL là biến thể thử nghiệm riêng | A: Standard+CA-SCL, B: Crypto+CA-SCL, C: Crypto+Modified |
| Có đề xuất decoder mới không? | Không khẳng định tính mới; tên biểu đồ là Modified, không Proposed Improved | Mô tả heuristic, đối chiếu BER/BLER và chi phí, rà tài liệu trước tuyên bố novelty |
| SC/SCL | Theo quyết định người dùng: bỏ SC khỏi thí nghiệm mới, tập trung CA-SCL; BP cũng không thuộc bộ so sánh mới | Giải thích thu hẹp phạm vi; không kết luận so SC–SCL |
| Hình 2/3 cùng L, cùng decoder | Bob riêng, mỗi L một subplot; Standard/Crypto cùng CRC, payload, channel map và noise realization | So A–B ở từng L và từng N,K,Tx/Rx |
| Eve lẫn vào chất lượng Bob | Tách Fig5 và nhóm đối chứng Eve | Không diễn giải chênh lệch kênh/khóa là coding gain của decoder |
| Tx128/Rx4, nhiều N,K,R | Phase 1 nhận kích thước array; Phase 2 kiểm tra H và metadata; 12 cặp N,K mẫu | Chạy CSI mới cho mỗi mảng anten; không cắt/lặp CSI |
| Vì sao NIST/65536 bit/8 test? | Bỏ battery NIST và hình NIST khỏi mã chạy theo yêu cầu người dùng | Xóa kết luận NIST khỏi bản báo cáo mới; không cần thay bằng số liệu giả |
| Fig5 sai khóa 1,25,50,100% theo Eb/N0 | Lật đúng số bit sau làm tròn; hai phạm vi keystream tách riêng, Bob khóa đúng và Eve khóa đúng làm mốc | Đọc số bit và tỷ lệ thực tế; không coi 100% là khóa ngẫu nhiên |
| Fig1 mục đích và cách tính | Vẽ gain beamforming từ H, tương quan vector TX chủ đạo, kích thước thật | Mô tả đáp ứng kênh và quan hệ không gian; không suy ra độ an toàn từ tương quan |
| 2 GHz/512 subcarrier | Ghi rõ đây là tham số mô phỏng đang dùng, chưa có cơ sở gọi là chuẩn 6G | Cần nguồn/tài liệu hoặc khảo sát bổ sung trước khi biện luận lựa chọn trong bài |

## Công bằng và các chỉnh sửa ảnh hưởng kết quả

- Bỏ ripple cos/sin thêm vào CSI và fading Rician thêm ngoài ray tracing. Kết quả
  mới có thể khác cũ vì mô hình thực thi đã đổi; không vẽ lại cho giống hình cũ.
- Bộ thu dùng coherent BPSK logits chính xác từ gain. Tất cả decoder nhận cùng
  logits, không cấp CSI bổ sung riêng cho Modified. Cấu hình damping của Modified
  là heuristic công bố tường minh, cần kiểm chứng chứ không được mặc định tốt hơn.
- BPSK được gọi đúng tên. Mã không thực hiện IFFT/FFT, CP hay tích chập OFDM theo
  thời gian. 512 là số mẫu đáp ứng kênh, N là chiều dài mã, hai tham số khác nhau.
  Với B=2 GHz và 512 carrier: khoảng cách tần số 3.90625 MHz, thời gian hữu ích
  tương ứng 256 ns. CP cũ 64 mẫu tương ứng 32 ns nhưng KHÔNG được mô phỏng ở đây;
  không dùng kết quả này để kết luận CP đủ che delay spread/ISI. Đây là giới hạn
  thực thi cần nhớ khi viết system model sau này, không phải system model đã hoàn tất.
- Mỗi khung được ánh xạ vào hoán vị carrier; đủ một vòng mới lặp. Các phương pháp
  dùng cùng ánh xạ, payload và nhiễu. Không chỉ lấy N carrier đầu để so các N khác nhau.
- Eb/N0 dùng payload rate (K-CRC)/N; CRC và frozen positions giống nhau giữa mã cơ sở
  và Crypto. K ghi trong config đã gồm CRC. Mã dùng ranking Polar có sẵn, không
  phải tối ưu construction riêng cho mỗi kênh.
- Budget cố định cho tất cả đường tại mỗi SNR; xử lý cả batch cuối không đủ kích thước.
  Ghi error counts và độ bất định. Không ép BER về sàn 1e-6 để làm đẹp đường.
- L ban đầu 8 của Modified có thể mở rộng 16 và giải mã lại nhiều lần. So với cả
  baseline 8 và 16, ghi `expanded_frames`, `flip_frames`, `core_calls`,
  `max_list_size`. L giống nhau không đồng nghĩa tổng chi phí giống nhau.
- Các vị trí lật là LLR của channel/codeword; không gọi là critical set bit thông tin.
  CRC có thể chấp nhận sai; BER/BLER so với payload thật, không dựa vào CRC pass.
- Thời gian ghi bằng CPU wall clock sau warmup, chia theo số khung của batch.
  Không diễn giải là độ trễ gói thực tế hoặc kết quả benchmark phần cứng phổ quát.
- Goodput=(K-CRC)/N*(1-BLER), đơn vị bit dữ liệu/channel use. Bỏ phép nhân entropy
  BER để gọi secret throughput; bỏ việc lấy BER Bob gán làm BER Eve Standard.

## Các đối chứng về khóa

`full`: làm sai vị trí ngẫu nhiên trong N bit khóa dẫn xuất gồm cả information và
frozen. `frozen`: chỉ phần N-K bit frozen bị sai, phần information được biết đúng.
Cả hai dùng mã Crypto đầy đủ. Đây KHÔNG phải thay đổi p% bit master key, cũng
không phải xác suất Eve biết master key. 100% là phần bù của dãy bit trong phạm vi
đã chọn, có cấu trúc, không được kết luận tự động là trường hợp khó nhất.

Frozen-only là biến thể riêng không che information. Biến đổi Polar khả nghịch:
không nhiễu có thể đảo codeword và đọc information positions mà không cần khóa
frozen. Có kiểm thử và đối chứng hard-decision qua kênh; đối chứng này không phải
bộ tấn công tối ưu. Không lấy BER của riêng CA-SCL đoán frozen sai làm bằng chứng
frozen-only bảo mật.

Chẩn đoán avalanche sửa lỗi cũ thay cả seed lẫn frame ID: mỗi cặp giờ có cùng
plaintext, CSI digest và frame ID, chỉ lật một bit master key. Phổ trọng số dùng
cùng độ dài payload và CRC. Các chẩn đoán không chứng minh IND-CPA, secrecy capacity
hoặc entropy khóa. CSI digest hiện hash magnitudes lượng tử hóa, không phải trích
entropy đo được từ kênh đối ứng. Master key trong runner là fixture công khai.

## Trình tự đọc và viết kết luận sau khi chạy

1. Kiểm tra manifest hoàn tất, đúng Tx/Rx, tọa độ BS và metadata.
2. Đọc Fig1/raw gain trước, ghi rõ `physical` hay `equal-average` cho Eve.
3. Đọc Fig2/3 cùng L: Crypto có làm giảm chất lượng Bob không? Dùng sai số thống kê
   để phân biệt khác biệt thật với dao động mẫu; không mặc định tương đương.
4. Đọc decoder comparison và bảng chi phí: có lợi ích nào sau khi tính retries?
5. Đọc Fig5 riêng từng phạm vi sai khóa; ghi giả thiết Eve biết gì, không suy rộng
   BER cao thành chứng minh an toàn mật mã.
6. Lặp lại N,K,Tx/Rx theo config. Các kết luận chỉ áp dụng những trường hợp đã chạy.

Chưa chạy full Phase 1/2 cho đợt sửa này. Mọi kết quả nghiên cứu mới sẽ do người dùng
chạy trên VS Code; kiểm thử kỹ thuật không thay thế các thí nghiệm đó.
