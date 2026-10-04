# 📈 Hệ Thống Kiểm Định Chiến Lược Giao Dịch Định Lượng (Quantitative Trading Backtesting Web App)

Ứng dụng web tương tác xây dựng trên nền tảng **Streamlit**, phục vụ việc backtest (kiểm định lịch sử) và đánh giá tính hiệu quả của chiến lược giao dịch kết hợp giữa **Đường Trung Bình Động (SMA)**, **Chỉ Số Dòng Tiền (OBV)** và **Tối Ưu Hóa Danh Mục Hiện Đại (Modern Portfolio Theory - Markowitz MPT)** trên dữ liệu cổ phiếu niêm yết tại Sở Giao dịch Chứng khoán TP.HCM (HOSE).

---

## 📌 1. Giới Thiệu Dự Án

Dự án được chuẩn hóa từ mô hình phân tích định lượng trong Jupyter Notebook sang một ứng dụng Web hoàn chỉnh, cho phép:
* Phân tích độc lập và kết hợp các chỉ báo xu hướng (**SMA**) và chỉ báo khối lượng dòng tiền (**OBV**).
* Tự động tối ưu hóa siêu tham số (Hyperparameter Optimization) bằng thuật toán Bayesian Optimization (**Hyperopt - TPE**) trên tập huấn luyện (**Train**).
* Xây dựng và so sánh danh mục đầu tư theo 2 phương pháp: **Equal Weight (Tỷ trọng đều 1/N)** và **Markowitz MPT (Tối đa hóa Sharpe Ratio)**.
* Kiểm định ngoài mẫu (**Out-of-sample Testing**) trên dữ liệu năm 2022 để kiểm tra mức độ sống sót và bảo toàn vốn trong giai đoạn thị trường Downtrend khốc liệt mà không bị thiên kiến quá khớp (**Overfitting**).
* Triệt tiêu hoàn toàn thiên kiến nhìn trước tương lai (**Look-ahead Bias**) thông qua cơ chế khớp lệnh trễ 1 phiên (`shift(1)`).

---

## 🚀 2. Hướng Dẫn Triển Khai Lên Streamlit Cloud (Deploy Guide)

### Bước 1: Khởi tạo và đẩy mã nguồn lên GitHub
1. Mở terminal hoặc Git Bash tại thư mục dự án và khởi tạo Git:
   ```bash
   git init
   git add app.py requirements.txt README.md HOSE_2020_2023_in.csv
   git commit -m "Initial commit: Quantitative backtesting web app"
   ```
2. Tạo một Repository mới trên [GitHub](https://github.com/new) (ví dụ đặt tên: `hose-trading-strategy-backtest`).
3. Liên kết và đẩy code lên GitHub:
   ```bash
   git remote add origin https://github.com/<tai-khoan-cua-ban>/hose-trading-strategy-backtest.git
   git branch -M main
   git push -u origin main
   ```

> 💡 **Lưu ý về file dữ liệu**: File `HOSE_2020_2023_in.csv` dung lượng ~5.5MB (nhỏ hơn nhiều so với giới hạn 100MB của GitHub), do đó bạn hoàn toàn có thể push trực tiếp lên GitHub để Web App tự động nhận diện dữ liệu khi deploy!

### Bước 2: Triển khai trên Streamlit Community Cloud
1. Truy cập [share.streamlit.io](https://share.streamlit.io) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"New app"** (hoặc **"Create app"**).
3. Điền các thông số:
   * **Repository**: Chọn repo bạn vừa tạo (`<tai-khoan-cua-ban>/hose-trading-strategy-backtest`).
   * **Branch**: `main`.
   * **Main file path**: `app.py`.
4. Nhấn **"Deploy!"**. Streamlit Cloud sẽ tự động cài đặt các thư viện trong `requirements.txt` và khởi chạy Web App trong khoảng 1-2 phút.

---

## 💻 3. Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local Machine)

Nếu muốn chạy thử nghiệm trên máy tính cá nhân:

### 1. Chuẩn bị môi trường Python
Khuyến nghị sử dụng Python phiên bản 3.9, 3.10 hoặc 3.11.

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv

# Kích hoạt môi trường ảo:
# Trên Windows:
venv\Scripts\activate
# Trên macOS / Linux:
source venv/bin/activate
```

### 2. Cài đặt các gói phụ thuộc
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng
```bash
streamlit run app.py
```
Sau khi chạy lệnh trên, trình duyệt web sẽ tự động mở địa chỉ: `http://localhost:8501`.

---

## 📂 4. Cấu Trúc Thư Mục Dự Án

```plaintext
├── app.py                     # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt           # Danh sách các thư viện Python cần thiết
├── README.md                  # Tài liệu hướng dẫn sử dụng và triển khai
├── HOSE_2020_2023_in.csv      # Tệp dữ liệu lịch sử giá OHLCV của 100 mã HOSE
└── HOSE_SMA_OBV_EqualWeight_MPT (1).ipynb # Notebook gốc phân tích định lượng
```

---

## 🧠 5. Cơ Sở Lý Thuyết & Phương Pháp Luận Định Lượng

### 5.1. Chỉ báo & Quy Tắc Sinh Tín Hiệu
* **SMA (Simple Moving Average)**:
  * Điểm MUA (+1): $SMA_{short}$ cắt lên trên $SMA_{long}$ (Golden Cross).
  * Điểm BÁN (-1): $SMA_{short}$ cắt xuống dưới $SMA_{long}$ (Death Cross).
* **OBV (On-Balance Volume)**:
  * $OBV_t = OBV_{t-1} + \text{sgn}(Close_t - Close_{t-1}) \times Volume_t$.
  * Điểm MUA (+1): Đường $OBV$ vượt lên đường trung bình động $OBV\_MA$.
  * Điểm BÁN (-1): Đường $OBV$ gãy xuống dưới đường trung bình động $OBV\_MA$.
* **Cơ Chế Kết Hợp**:
  * **Chế độ AND**: Chỉ vào lệnh MUA khi cả SMA và OBV đều báo MUA; BÁN khi cả hai đều báo BÁN.
  * **Chế độ OR**: Vào lệnh MUA khi một trong hai chỉ báo phát tín hiệu MUA; BÁN khi một trong hai chỉ báo phát tín hiệu BÁN. Nếu tín hiệu xung đột (vừa Mua vừa Bán trong cùng một ngày), hệ thống duy trì vị thế trung lập (0).

### 5.2. Khắc Phục Look-Ahead Bias
Tín hiệu được xác định tại thời điểm kết thúc phiên ngày $t$. Hệ thống dịch chuyển vị thế nắm giữ 1 phiên (`executed_holding = holding.shift(1)`), đảm bảo việc khớp lệnh chỉ diễn ra ở phiên tiếp theo $t+1$ nhằm phản ánh đúng thực tế giao dịch.

### 5.3. Tối Ưu Hóa Danh Mục Hiện Đại (Markowitz MPT)
Vector trọng số $W = [w_1, w_2, ..., w_N]^T$ được tối ưu hóa chỉ trên tập **Train** bằng phương pháp **SLSQP**:

$$\max_{W} \frac{W^T \mu - r_f}{\sqrt{W^T \Sigma W}}$$

* **Ràng buộc**: $0 \le w_i \le 1$ (Long-only, không bán khống) và $\sum_{i=1}^N w_i = 1$.
* Trọng số này được giữ nguyên cố định khi mang sang đánh giá tại tập kiểm định ngoài mẫu **Test**.

---

## 📊 6. Kết Quả Thực Nghiệm Mẫu (Tham khảo từ Notebook)

* **Danh mục đại diện 3 ngành**:
  * Ngân hàng: `ACB` (Tham số tối ưu: SMA 60/270, OBV window 5)
  * Công nghệ: `FPT` (Tham số tối ưu: SMA 70/205, OBV window 65)
  * Thép/Sản xuất: `HPG` (Tham số tối ưu: SMA 25/220, OBV window 40)
* **Kết hợp tín hiệu**: `SMA + OBV OR`
* **So sánh hiệu quả danh mục**:

| Giai đoạn | Phương pháp phân bổ | Tổng Lợi Nhuận | Lợi Nhuận Năm | Biến Động Năm | Tỷ Số Sharpe | Mức Sụt Giảm Tối Đa |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train (2020 - 2021)** | **Equal Weight** | **+138.32%** | +54.64% | 21.34% | 2.151 | -11.89% |
| **Train (2020 - 2021)** | **MPT (Markowitz)** | **+152.56%** | +59.21% | 22.36% | 2.195 | -12.53% |
| **Test (2022 - Out of sample)** | **Equal Weight** | **-17.65%** | -17.84% | 14.29% | -1.304 | **-20.02%** |
| **Test (2022 - Out of sample)** | **MPT (Markowitz)** | **-18.76%** | -18.97% | 15.33% | -1.295 | -21.53% |

> 📌 **Nhận xét chuyên gia**: Trong năm 2022 khi chỉ số VN-Index sụt giảm hơn 33%, cả hai danh mục Equal Weight và MPT đều giảm dưới 19%, chứng minh hiệu quả giảm thiểu rủi ro đáng kể của chiến lược dòng tiền OBV kết hợp bộ lọc xu hướng SMA so với chiến lược Buy & Hold truyền thống.

---

## 🛠️ 7. Tính Năng Chính Trên Ứng Dụng Web

1. **Bộ lọc đa năng**: Tự do lựa chọn từ 1 đến 100 mã cổ phiếu HOSE, tùy chỉnh chu kỳ Train/Test linh hoạt.
2. **Biểu đồ trực quan (Plotly)**: Biểu đồ đường cong tăng trưởng vốn (Equity Curve), biểu đồ kỹ thuật tương tác đa khung (Price, SMA, OBV, Buy/Sell Signals), biểu đồ mức sụt giảm tài khoản (Underwater Drawdown).
3. **Mô-đun tối ưu hóa Hyperopt**: Chạy tìm kiếm Bayesian Optimization trực tiếp trên giao diện web hoặc tinh chỉnh thủ công từng tham số theo ý muốn.
4. **Nhật ký & Xuất dữ liệu**: Xem lại lịch sử tín hiệu 15 phiên gần nhất và tải về file CSV kết quả backtest (ma trận lợi nhuận, bảng chỉ số định lượng).

---

## 📜 8. Giấy Phép & Bản Quyền
Dự án được phát triển phục vụ mục đích nghiên cứu học thuật và kiểm định chiến lược giao dịch định lượng. Mã nguồn mở theo giấy phép MIT.
