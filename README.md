# 🛡️ Ứng dụng Web Phân tích và Dự báo Trạng thái Rủi ro (PD)

Ứng dụng web này được phát triển bằng framework **Streamlit** để chuyển đổi quy trình xử lý và huấn luyện mô hình học máy dạng mã nguồn tĩnh từ tập tin Jupyter Notebook (`mohinh.ipynb`) sang một giao diện web trực quan, tương tác và dễ sử dụng cho người vận hành.

## 🛠️ Tính năng chính của hệ thống
1. **Sidebar quản lý tập trung**: Cho phép tải dữ liệu và tinh chỉnh động các siêu tham số của thuật toán rừng ngẫu nhiên (*Random Forest Classifier*) như số cây, độ sâu và mã hạt giống ngẫu nhiên.
2. **Tab Tổng quan dữ liệu**: Trực quan hóa cấu trúc dữ liệu thô và tính toán nhanh các chỉ số thống kê mô tả phục vụ phân tích ban đầu.
3. **Tab Trực quan hóa dữ liệu**: Tự động nhận diện cấu trúc phân phối của biến đích `PD` và các thuộc tính đặc trưng độc lập thông qua đồ thị động `Plotly`.
4. **Tab Kết quả & Kiểm định**: Trực quan hoá các chỉ số đánh giá chuyên sâu gồm Báo cáo phân loại (*Classification Report*), Ma trận nhầm lẫn (*Confusion Matrix*), và Đồ thị đường cong phản hồi *ROC-AUC*.
5. **Tab Sử dụng mô hình**: Tích hợp hai cơ chế chấm điểm linh hoạt: Nhập liệu kiểm tra nhanh một đối tượng đơn lẻ qua Form biểu mẫu hoặc Tải file chứa danh sách hồ sơ để phân loại rủi ro hàng loạt và xuất file kết quả đầu ra.

## 📊 Mô tả cấu trúc tệp dữ liệu mẫu
Hệ thống được thiết lập cấu hình tương thích hoàn toàn với tệp cấu trúc mẫu giống `5c.csv`. Các cột thông tin bắt buộc bao gồm:
- **Hệ thống 25 biến độc lập (X)**: Gồm các nhóm thuộc tính hành vi và chỉ số `TC1`-`TC5`, `NL1`-`NL4`, `DK1`-`DK5`, `V1`-`V6`, `TS1`-`TS4`, và biến định danh phân nhóm `NN`.
- **Biến mục tiêu phụ thuộc (y)**: Cột `PD` mang định dạng giá trị nhị phân (`0` ứng với hồ sơ An toàn / Bình thường; `1` ứng với hồ sơ Xuất hiện rủi ro).

## 🚀 Hướng dẫn cài đặt và chạy ứng dụng

1. **Cài đặt các thư viện phụ thuộc:**
   Đảm bảo bạn đã cài đặt Python (phiên bản khuyến nghị `>=3.9`). Chạy lệnh sau tại Terminal/Command Prompt:
   ```bash
   pip install -r requirements.txt

