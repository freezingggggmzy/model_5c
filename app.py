import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# 1) LỆNH STREAMLIT ĐẦU TIÊN: Cấu hình trang giao diện rộng (Wide layout)
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Dự báo và Phân tích Rủi ro",
    page_icon="🛡️"
)

# 2) Hàm nạp dữ liệu dùng chung có cache dữ liệu thông minh
@st.cache_data
def load_data(file_bytes, file_name):
    try:
        if file_name.endswith('.csv'):
            return pd.read_csv(file_bytes)
        else:
            return pd.read_excel(file_bytes)
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# Trích xuất danh sách biến độc lập X và biến mục tiêu y từ notebook
X_COLS = [
    'TC1', 'TC2', 'TC3', 'TC4', 'TC5', 
    'NL1', 'NL2', 'NL3', 'NL4', 
    'DK1', 'DK2', 'DK3', 'DK4', 'DK5', 
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 
    'TS1', 'TS2', 'TS3', 'TS4', 'NN'
]
Y_COL = 'PD'

# 3) SIDEBAR — VÙNG CẤU HÌNH & ĐIỀU KHIỂN
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Widget tải file dữ liệu huấn luyện mẫu
    uploaded_file = st.file_uploader(
        "Tải dữ liệu mẫu của ứng dụng (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn file dữ liệu mẫu chứa các cột chỉ số (TC, NL, DK, V, TS, NN) và biến mục tiêu rủi ro PD."
    )
    
    st.divider()
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: **Random Forest Classifier**")
    
    # Các siêu tham số lấy giá trị mặc định tương ứng với notebook gốc
    n_estimators = st.slider(
        "Số lượng cây quyết định (n_estimators)", 
        min_value=10, max_value=500, value=100, step=10,
        help="Số lượng cây sẽ được xây dựng trong mô hình rừng ngẫu nhiên."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa của cây (max_depth)", 
        min_value=1, max_value=50, value=10, step=1,
        help="Giới hạn chiều sâu tối đa của mỗi cây để tránh quá khớp (overfitting)."
    )
    
    random_state = st.number_input(
        "Mã ngẫu nhiên (random_state)", 
        value=42, step=1,
        help="Đảm bảo tính nhất quán và khả năng tái lập kết quả huấn luyện qua mỗi lần chạy."
    )
    
    st.divider()
    
    # Nút bấm hành động duy nhất để kích hoạt huấn luyện mô hình ở dưới cùng sidebar
    btn_train = st.button(
        "🚀 Huấn luyện mô hình", 
        type="primary", 
        use_container_width=True,
        help="Bấm để thực hiện phân chia dữ liệu Train/Test và huấn luyện thuật toán học máy."
    )

# 4) HEADER — VÙNG ĐỊNH HƯỚNG NẰM Ở ĐẦU TRANG CHÍNH
st.title("🛡️ Hệ thống Dự báo và Phân tích Rủi ro Tự động")
st.caption("Ứng dụng hỗ trợ phân loại và chấm điểm rủi ro hồ sơ tự động dựa trên mô hình học máy Random Forest.")

# Xử lý trạng thái rỗng nếu người dùng chưa tải file lên ứng dụng
if uploaded_file is None:
    st.info("💡 Vui lòng tải lên tệp dữ liệu mẫu (.csv hoặc .xlsx) ở thanh Sidebar bên trái để bắt đầu kích hoạt hệ thống.")
    st.stop()
else:
    # Gọi hàm nạp dữ liệu dùng chung đã qua cache
    df_raw = load_data(uploaded_file, uploaded_file.name)
    if df_raw is None:
        st.stop()
    st.caption(f"📁 Đang dùng tệp dữ liệu: **{uploaded_file.name}** | Quy mô: {df_raw.shape[0]:,} dòng, {df_raw.shape[1]} cột.")

st.divider()

# 5) Khối huấn luyện mô hình (Chỉ chạy MỘT LẦN duy nhất khi bấm nút, lưu kết quả trực tiếp vào st.session_state)
if btn_train:
    # Kiểm tra tính toàn vẹn và hợp lệ của các cột dữ liệu yêu cầu
    missing_cols = [col for col in X_COLS + [Y_COL] if col not in df_raw.columns]
    if missing_cols:
        st.error(f"❌ File dữ liệu thiếu các cột thuộc tính bắt buộc sau để huấn luyện: {missing_cols}")
    else:
        with st.spinner("⏳ Hệ thống đang xử lý và huấn luyện mô hình Random Forest..."):
            # Tiền xử lý: Loại bỏ các dòng chứa giá trị null ở các biến mô hình
            df_clean = df_raw.dropna(subset=X_COLS + [Y_COL])
            
            X = df_clean[X_COLS]
            y = df_clean[Y_COL]
            
            # Thực hiện phân chia tập dữ liệu huấn luyện (80%) và kiểm định (20%) theo phân tầng nhãn y
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state, stratify=y
            )
            
            # Khởi tạo mô hình học máy với tham số người dùng nhập từ sidebar
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            model.fit(X_train, y_train)
            
            # Dự đoán kết quả trên tập kiểm thử nhằm trích xuất các chỉ tiêu đo lường
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Lưu trữ đồng thời 3 thành phần cốt lõi vào session_state để tái sử dụng xuyên suốt không cần train lại
            st.session_state['trained_model'] = model
            st.session_state['feature_columns'] = X_COLS
            st.session_state['evaluation_metrics'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'report': classification_report(y_test, y_pred, output_dict=True),
                'cm': confusion_matrix(y_test, y_pred),
                'y_true': y_test.values,
                'y_prob': y_prob
            }
            st.success("🎉 Huấn luyện thành công! Kết quả đã được cập nhật vào các thẻ chức năng phía dưới.")

# 6) KHỐI TABS PHÂN PHỐI CHỨC NĂNG CHÍNH
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🔬 Kết quả & Kiểm định", 
    "🔮 Sử dụng mô hình"
])

# ---- TAB 1: TỔNG QUAN DỮ LIỆU ----
with tab1:
    st.subheader("Cấu trúc và Thống kê tệp dữ liệu")
    
    # Hiển thị tóm tắt kích thước qua hàng cột metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Tổng số dòng hồ sơ", f"{df_raw.shape[0]:,}")
    col_m2.metric("Số lượng cột thuộc tính", f"{df_raw.shape[1]}")
    file_size_mb = uploaded_file.size / (1024 * 1024)
    col_m3.metric("Dung lượng tệp dữ liệu", f"{file_size_mb:.2f} MB")
    
    st.write("#### 🔍 Hiển thị 5 dòng dữ liệu đầu tiên (Xem thô)")
    st.dataframe(df_raw.head(5), use_container_width=True)
    
    st.write("#### 📉 Bảng số liệu thống kê mô tả (Chỉ các biến đưa vào mô hình)")
    available_model_vars = [c for c in X_COLS + [Y_COL] if c in df_raw.columns]
    if available_model_vars:
        st.dataframe(df_raw[available_model_vars].describe(), use_container_width=True)
    else:
        st.warning("Không tìm thấy các biến đặc trưng tương ứng trong tệp dữ liệu này.")

# ---- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ----
with tab2:
    st.subheader("Biểu đồ phân phối các biến số độc lập và biến mục tiêu")
    
    # Ưu tiên hiển thị biến mục tiêu rủi ro (y = PD) lên đầu tiên
    if Y_COL in df_raw.columns:
        fig_y = px.histogram(
            df_raw, x=Y_COL, 
            title="Đồ thị phân phối nhãn rủi ro (Biến mục tiêu PD)",
            labels={Y_COL: "Trạng thái PD (0: Bình thường, 1: Rủi ro)"},
            color_discrete_sequence=['#4A90E2'],
            text_auto=True
        )
        fig_y.update_layout(height=350, bargap=0.4)
        st.plotly_chart(fig_y, use_container_width=True)
    
    st.write("#### 🎛️ Lựa chọn thuộc tính tính năng để hiển thị chi tiết (Tối đa hiển thị mạng lưới 2x2)")
    selected_features = st.multiselect(
        "Chọn các biến độc lập cần vẽ biểu đồ:",
        options=X_COLS,
        default=X_COLS[:4],
        max_selections=4
    )
    
    # Thiết lập phân bố biểu đồ dạng lưới 2 hàng x 2 cột (Cân đối tối đa 4 đồ thị)
    if selected_features:
        grid_cols = st.columns(2)
        for idx, feat in enumerate(selected_features):
            if feat in df_raw.columns:
                target_box = grid_cols[idx % 2]
                with target_box:
                    # Kiểm tra kiểu giá trị để lựa chọn đồ thị bar/histogram tương ứng với phân bố thực tế
                    if df_raw[feat].nunique() <= 10:
                        fig_f = px.bar(
                            df_raw[feat].value_counts().reset_index(),
                            x='index', y=feat,
                            title=f"Phân phối tần suất của biến số {feat}",
                            labels={'index': f'Giá trị {feat}', feat: 'Số lượng'},
                            color_discrete_sequence=['#50E3C2']
                        )
                    else:
                        fig_f = px.histogram(
                            df_raw, x=feat,
                            title=f"Biểu đồ phân phối liên tục của biến số {feat}",
                            labels={feat: f'Khoảng giá trị {feat}'},
                            color_discrete_sequence=['#B8E986']
                        )
                    fig_f.update_layout(height=300)
                    st.plotly_chart(fig_f, use_container_width=True)

# ---- TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH ----
with tab3:
    st.subheader("Kết quả đánh giá chất lượng mô hình phân loại")
    
    # Kiểm tra điều phối nếu chưa thực hiện train mô hình
    if 'evaluation_metrics' not in st.session_state:
        st.info("💡 Chưa tìm thấy mô hình đã huấn luyện. Vui lòng thiết lập thông số và bấm nút **'Huấn luyện mô hình'** tại thanh Sidebar.")
    else:
        metrics = st.session_state['evaluation_metrics']
        
        # Hiển thị độ chính xác toàn cục
        st.metric("Độ chính xác trên tập kiểm định (Accuracy Score)", f"{metrics['accuracy']*100:.2f} %")
        
        st.write("#### 📋 Báo cáo phân loại chi tiết (Classification Report)")
        report_df = pd.DataFrame(metrics['report']).transpose()
        st.dataframe(report_df.style.format(precision=4), use_container_width=True)
        
        # Hiển thị Ma trận nhầm lẫn và đường cong ROC tương phản song song
        plot_col1, plot_col2 = st.columns(2)
        
        with plot_col1:
            st.write("#### 🧮 Ma trận nhầm lẫn (Confusion Matrix)")
            fig_cm = px.imshow(
                metrics['cm'], text_auto=True,
                labels=dict(x="Giá trị dự đoán", y="Giá trị thực tế"),
                x=['An toàn (0)', 'Rủi ro (1)'],
                y=['An toàn (0)', 'Rủi ro (1)'],
                color_continuous_scale="Purples"
            )
            fig_cm.update_layout(height=350)
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with plot_col2:
            st.write("#### 📉 Đường cong đặc trưng hoạt động thụ nhận (ROC Curve)")
            y_true = metrics['y_true']
            y_prob = metrics['y_prob']
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.4f})', line=dict(color='crimson', width=2)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Dự đoán ngẫu nhiên', line=dict(color='gray', width=1, dash='dash')))
            fig_roc.update_layout(
                xaxis_title="Tỷ lệ Dương tính giả (FPR)",
                yaxis_title="Tỷ lệ Dương tính thật (TPR)",
                height=350,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

# ---- TAB 4: SỬ DỤNG MÔ HÌNH DỰ BÁO ----
with tab4:
    st.subheader("Vận hành mô hình phục vụ chấm điểm và phân loại rủi ro")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Vui lòng hoàn thành bước huấn luyện mô hình ở mục Sidebar trước khi sử dụng tính năng dự báo.")
    else:
        model = st.session_state['trained_model']
        
        # Lựa chọn chế độ nhập dữ liệu dự báo thông qua st.radio
        predict_mode = st.radio(
            "Chọn hình thức kiểm tra dự báo thông tin:",
            options=["Nhập thủ công thông số trực tiếp", "Tải file danh sách hồ sơ kiểm tra hàng loạt"],
            horizontal=True
        )
        
        if predict_mode == "Nhập thủ công thông số trực tiếp":
            st.write("#### 📝 Thiết lập thông số các chỉ số của đối tượng (X)")
            
            with st.form("form_predict_single"):
                # Tạo lưới 5 cột để gom gọn hiển thị nhập liệu cho toàn bộ 25 biến số đầu vào
                input_fields = st.columns(5)
                dict_single_input = {}
                
                for index, col_name in enumerate(X_COLS):
                    target_field = input_fields[index % 5]
                    with target_field:
                        # Gán giá trị mặc định dựa trên đặc trưng phân bổ của tệp dữ liệu mẫu 5c.csv
                        default_val = 1 if col_name == 'NN' else 3
                        dict_single_input[col_name] = st.number_input(
                            f"Chỉ số {col_name}", 
                            min_value=0, max_value=10, 
                            value=default_val, step=1
                        )
                        
                submit_single = st.form_submit_button("🔍 Tiến hành phân tích rủi ro", type="primary")
                
            if submit_single:
                df_single_test = pd.DataFrame([dict_single_input])
                
                single_pred = model.predict(df_single_test)[0]
                single_proba = model.predict_proba(df_single_test)[0]
                
                st.markdown("---")
                st.write("### 📊 Kết quả đánh giá đối tượng")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    if single_pred == 1:
                        st.error("🔴 KẾT LUẬN: ĐỐI TƯỢNG CÓ NGUY CƠ RỦI RO (PD = 1)")
                    else:
                        st.success("🟢 KẾT LUẬN: ĐỐI TƯỢNG AN TOÀN / KHÔNG NGUY HIỂM (PD = 0)")
                        
                with res_col2:
                    st.metric("Xác suất không có rủi ro (PD=0)", f"{single_proba[0]*100:.2f} %")
                    st.metric("Xác suất phát sinh rủi ro (PD=1)", f"{single_proba[1]*100:.2f} %")
                    
        elif predict_mode == "Tải file danh sách hồ sơ kiểm tra hàng loạt":
            st.write("#### 📂 Tải lên tệp dữ liệu cần chấm điểm (Yêu cầu chứa đầy đủ cấu trúc 25 biến độc lập)")
            bulk_uploader_file = st.file_uploader(
                "Tải lên tệp CSV hoặc Excel mới của bạn tại đây:", 
                type=["csv", "xlsx"], key="bulk_scoring"
            )
            
            if bulk_uploader_file is not None:
                df_bulk_raw = load_data(bulk_uploader_file, bulk_uploader_file.name)
                if df_bulk_raw is not None:
                    # Rà soát cấu trúc cột đầu vào của file tải lên
                    missing_features = [f_col for f_col in X_COLS if f_col not in df_bulk_raw.columns]
                    if missing_features:
                        st.error(f"❌ Tệp tin chấm điểm bị lỗi. Thiếu các thuộc tính cột bắt buộc sau: {missing_features}")
                    else:
                        # Lọc dòng trống và tiến hành dự đoán hàng loạt
                        df_bulk_clean = df_bulk_raw[X_COLS].dropna()
                        
                        bulk_predictions = model.predict(df_bulk_clean)
                        bulk_probabilities = model.predict_proba(df_bulk_clean)[:, 1]
                        
                        # Tạo bảng báo cáo đầu ra tổng hợp dữ liệu kết quả dự báo
                        df_bulk_output = df_bulk_clean.copy()
                        df_bulk_output['DuBao_TrangThai_PD'] = bulk_predictions
                        df_bulk_output['XacSuat_RuiRo_PD_1'] = bulk_probabilities
                        
                        st.success(f"🎉 Hệ thống đã hoàn tất chấm điểm tự động cho {df_bulk_output.shape[0]} dòng hồ sơ dữ liệu hợp lệ!")
                        st.write("##### Xem trước bảng kết quả phân tích:")
                        st.dataframe(df_bulk_output, use_container_width=True)
                        
                        # Xuất bảng dữ liệu ra file CSV để người dùng lưu trữ về máy
                        csv_bytes = df_bulk_output.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Tải xuống tệp kết quả dự báo tổng hợp (.csv)",
                            data=csv_bytes,
                            file_name="KetQua_PhanTich_RuiRo_HangLoat.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

