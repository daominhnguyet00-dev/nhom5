"""
Hệ Thống Kiểm Định Chiến Lược Giao Dịch Định Lượng (Quantitative Backtesting System)
Chiến Lược Kết Hợp Tín Hiệu SMA + OBV & Tối Ưu Hóa Danh Mục MPT (Markowitz)
Nền tảng: Streamlit Web App
Dữ liệu: HOSE 2020 - 2023
"""

import os
import io
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

# Thư viện phân tích kỹ thuật và tối ưu hóa
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

try:
    from hyperopt import fmin, tpe, hp, Trials
    HAS_HYPEROPT = True
except ImportError:
    HAS_HYPEROPT = False

warnings.filterwarnings("ignore")

# ==========================================
# CẤU HÌNH TRANG STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Kiểm Định Chiến Lược SMA + OBV & MPT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0;
    }
    .metric-delta-pos {
        color: #10B981;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .metric-delta-neg {
        color: #EF4444;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #D1FAE5; color: #065F46; }
    .badge-purple { background-color: #EDE9FE; color: #5B21B6; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# CÁC HÀM XỬ LÝ DỮ LIỆU & CHỈ BÁO KỸ THUẬT
# ==========================================

@st.cache_data(show_spinner=False)
def load_raw_dataset(uploaded_file=None):
    """Đọc và chuẩn hóa dữ liệu HOSE thô từ file tải lên hoặc file mặc định."""
    possible_paths = [
        "HOSE_2020_2023_in.csv",
        "HOSE_2020_2023_in(1).csv",
        "../HOSE_2020_2023_in.csv"
    ]
    
    df = None
    file_source = ""
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig", low_memory=False)
            file_source = f"File tải lên: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Lỗi khi đọc file tải lên: {e}")
            return None, ""
    else:
        for p in possible_paths:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
                    file_source = f"Tệp mặc định: {p}"
                    break
                except Exception:
                    continue
    
    if df is None:
        return None, ""
        
    df.columns = df.columns.str.strip().str.lower()
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Dữ liệu thiếu các cột bắt buộc: {missing}")
        return None, ""
        
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["date", "ticker"])
    return df, file_source


def prepare_stock_data(df_full, ticker):
    """Lọc và chuẩn hóa chuỗi dữ liệu OHLCV cho một mã cổ phiếu cụ thể."""
    ticker = ticker.upper()
    df = df_full[df_full["ticker"] == ticker].copy()
    if df.empty:
        return pd.DataFrame()
        
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })
    df = df[["date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.set_index("date")
    return df


def calculate_sma(series, window):
    """Tính đường Trung bình Động Giản đơn (SMA)."""
    return series.rolling(window=int(window)).mean()


def calculate_obv(close, volume):
    """Tính chỉ số On-Balance Volume (OBV)."""
    if HAS_TA:
        try:
            return ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        except Exception:
            pass
    # Tính toán dự phòng thuần túy bằng pandas
    diff = close.diff()
    direction = pd.Series(0.0, index=close.index)
    direction[diff > 0] = 1.0
    direction[diff < 0] = -1.0
    return (direction * volume).cumsum()


# ==========================================
# HÀM SINH TÍN HIỆU GIAO DỊCH
# ==========================================

def find_position_sma(df, paras):
    """Tín hiệu giao dịch theo đường trung bình SMA."""
    position = pd.Series(0.0, index=df.index, name="position")
    ma_short = int(paras["ma_short"])
    ma_long = int(paras["ma_long"])
    
    if ma_short >= ma_long:
        return position
        
    ma_s = calculate_sma(df["Close"], ma_short)
    ma_l = calculate_sma(df["Close"], ma_long)
    
    buy_signal = (ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))
    sell_signal = (ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))
    
    position.loc[buy_signal] = 1.0
    position.loc[sell_signal] = -1.0
    return position


def find_position_obv(df, paras):
    """Tín hiệu giao dịch theo chỉ báo khối lượng OBV và đường MA của OBV."""
    position = pd.Series(0.0, index=df.index, name="position")
    obv_window = int(paras["obv_window"])
    
    obv = calculate_obv(df["Close"], df["Volume"])
    obv_ma = obv.rolling(window=obv_window).mean()
    
    buy_signal = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell_signal = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))
    
    position.loc[buy_signal] = 1.0
    position.loc[sell_signal] = -1.0
    return position


def find_position_combined_and(df, sma_paras, obv_paras):
    """Kết hợp tín hiệu AND: Mua khi cả SMA & OBV cùng báo Mua; Bán khi cả 2 cùng báo Bán."""
    sma = find_position_sma(df, sma_paras)
    obv = find_position_obv(df, obv_paras)
    
    position = pd.Series(0.0, index=df.index, name="position")
    position.loc[(sma == 1.0) & (obv == 1.0)] = 1.0
    position.loc[(sma == -1.0) & (obv == -1.0)] = -1.0
    return position


def find_position_combined_or(df, sma_paras, obv_paras):
    """Kết hợp tín hiệu OR: Mua khi ít nhất 1 bên báo Mua, Bán khi ít nhất 1 bên báo Bán."""
    sma = find_position_sma(df, sma_paras)
    obv = find_position_obv(df, obv_paras)
    
    position = pd.Series(0.0, index=df.index, name="position")
    buy = (sma == 1.0) | (obv == 1.0)
    sell = (sma == -1.0) | (obv == -1.0)
    conflict = buy & sell
    
    position.loc[buy & ~conflict] = 1.0
    position.loc[sell & ~conflict] = -1.0
    return position


def events_to_holding(events):
    """Chuyển đổi tín hiệu rời rạc (+1 mua, -1 bán) thành trạng thái nắm giữ (1 = giữ, 0 = tiền mặt)."""
    holding = pd.Series(0.0, index=events.index)
    current = 0.0
    for i, signal in enumerate(events):
        if signal == 1.0:
            current = 1.0
        elif signal == -1.0:
            current = 0.0
        holding.iloc[i] = current
    return holding


def strategy_returns(df, events, commission=0.0):
    """
    Tính chuỗi lợi nhuận hàng ngày của chiến lược.
    QUAN TRỌNG: Dịch 1 phiên (shift 1) để triệt tiêu look-ahead bias.
    Tín hiệu kết phiên t chỉ được thực thi với giá phiên t+1.
    """
    asset_ret = df["Close"].pct_change().fillna(0.0)
    holding = events_to_holding(events)
    executed_holding = holding.shift(1).fillna(0.0)
    
    strat_ret = executed_holding * asset_ret
    turnover = executed_holding.diff().abs().fillna(executed_holding.abs())
    strat_ret = strat_ret - turnover * commission
    return strat_ret, executed_holding


def performance_stats(returns, trading_days=252):
    """Tính các chỉ số hiệu quả định lượng: Total Return, Sharpe, Volatility, Max Drawdown."""
    r = returns.dropna()
    if len(r) == 0:
        return {
            "Total Return [%]": np.nan,
            "Annual Return [%]": np.nan,
            "Annual Volatility [%]": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown [%]": np.nan,
            "Win Rate [%]": np.nan
        }
        
    equity = (1.0 + r).cumprod()
    total_return = equity.iloc[-1] - 1.0
    years = len(r) / trading_days
    
    annual_return = (
        equity.iloc[-1] ** (1.0 / years) - 1.0
        if years > 0 and equity.iloc[-1] > 0
        else np.nan
    )
    
    std = r.std()
    annual_vol = std * np.sqrt(trading_days) if std > 0 else 0.0
    sharpe = (r.mean() / std * np.sqrt(trading_days)) if std > 0 else np.nan
    
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = drawdown.min()
    
    active_days = r[r != 0]
    win_rate = (active_days > 0).mean() * 100 if len(active_days) > 0 else 0.0
    
    return {
        "Total Return [%]": total_return * 100.0,
        "Annual Return [%]": annual_return * 100.0,
        "Annual Volatility [%]": annual_vol * 100.0,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": max_dd * 100.0,
        "Win Rate [%]": win_rate
    }


# ==========================================
# TỐI ƯU HÓA THAM SỐ (HYPEROPT TPE)
# ==========================================

def score_sma(paras, df, commission=0.0):
    paras = {"ma_short": int(paras["ma_short"]), "ma_long": int(paras["ma_long"])}
    if paras["ma_short"] >= paras["ma_long"]:
        return 999999.0
    events = find_position_sma(df, paras)
    ret, _ = strategy_returns(df, events, commission=commission)
    stats = performance_stats(ret)
    sharpe = stats["Sharpe Ratio"]
    if pd.isna(sharpe):
        return 999999.0
    return -sharpe


def score_obv(paras, df, commission=0.0):
    paras = {"obv_window": int(paras["obv_window"])}
    events = find_position_obv(df, paras)
    ret, _ = strategy_returns(df, events, commission=commission)
    stats = performance_stats(ret)
    sharpe = stats["Sharpe Ratio"]
    if pd.isna(sharpe):
        return 999999.0
    return -sharpe


def optimize_one_stock(df_train, max_evals=40, commission=0.0):
    """Tối ưu hóa độc lập SMA và OBV trên tập Train sử dụng Hyperopt TPE."""
    if not HAS_HYPEROPT:
        return {"ma_short": 50, "ma_long": 200}, {"obv_window": 20}
        
    space_sma = {
        "ma_short": hp.quniform("ma_short", 25, 150, 5),
        "ma_long": hp.quniform("ma_long", 200, 400, 5)
    }
    trials_sma = Trials()
    best_sma_raw = fmin(
        fn=lambda p: score_sma(p, df_train, commission),
        space=space_sma,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials_sma,
        verbose=False
    )
    sma_best = {
        "ma_short": int(best_sma_raw["ma_short"]),
        "ma_long": int(best_sma_raw["ma_long"])
    }
    
    space_obv = {
        "obv_window": hp.quniform("obv_window", 5, 100, 5)
    }
    trials_obv = Trials()
    best_obv_raw = fmin(
        fn=lambda p: score_obv(p, df_train, commission),
        space=space_obv,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials_obv,
        verbose=False
    )
    obv_best = {
        "obv_window": int(best_obv_raw["obv_window"])
    }
    return sma_best, obv_best


# ==========================================
# TỐI ƯU HÓA DANH MỤC MPT (MARKOWITZ)
# ==========================================

def portfolio_annual_return(weights, returns, trading_days=252):
    mean_daily = returns.mean().values
    return float(weights @ mean_daily * trading_days)


def portfolio_annual_volatility(weights, returns, trading_days=252):
    cov_annual = returns.cov().values * trading_days
    variance = float(weights.T @ cov_annual @ weights)
    return np.sqrt(max(variance, 0.0))


def negative_sharpe(weights, returns, risk_free_rate=0.0, trading_days=252):
    p_return = portfolio_annual_return(weights, returns, trading_days)
    p_vol = portfolio_annual_volatility(weights, returns, trading_days)
    if p_vol <= 1e-7 or np.isnan(p_vol):
        return 1e9
    return -(p_return - risk_free_rate) / p_vol


def optimize_mpt(returns, risk_free_rate=0.0, trading_days=252):
    """
    Tối ưu hóa danh mục MPT: Tìm trọng số Long-only tối đa hóa tỷ số Sharpe trên tập Train.
    Ràng buộc: sum(w) = 1, 0 <= w_i <= 1.
    """
    n = returns.shape[1]
    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    
    try:
        result = minimize(
            negative_sharpe,
            x0=x0,
            args=(returns, risk_free_rate, trading_days),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            w = np.clip(result.x, 0.0, 1.0)
            if w.sum() > 0:
                return w / w.sum()
    except Exception:
        pass
    return x0


def portfolio_returns(return_matrix, weights):
    """Tính chuỗi lợi nhuận tổng thể của danh mục theo trọng số."""
    return return_matrix.mul(weights, axis=1).sum(axis=1)


# ==========================================
# KHỞI TẠO SESSION STATE & THAM SỐ MẪU
# ==========================================

DEFAULT_PARAMS = {
    "ACB": {"SMA": {"ma_short": 60, "ma_long": 270}, "OBV": {"obv_window": 5}},
    "FPT": {"SMA": {"ma_short": 70, "ma_long": 205}, "OBV": {"obv_window": 65}},
    "HPG": {"SMA": {"ma_short": 25, "ma_long": 220}, "OBV": {"obv_window": 40}}
}

if "stock_params" not in st.session_state:
    st.session_state["stock_params"] = DEFAULT_PARAMS.copy()


# ==========================================
# GIAO DIỆN THANH BÊN (SIDEBAR)
# ==========================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910312.png", width=64)
    st.title("⚙️ Cấu Hình Hệ Thống")
    
    # 1. Nạp Dữ Liệu
    st.subheader("1. Nguồn Dữ Liệu")
    uploaded_file = st.file_uploader(
        "Tải lên file CSV dữ liệu HOSE:",
        type=["csv"],
        help="Định dạng yêu cầu: date, ticker, open, high, low, close, volume"
    )
    
    df_raw, data_source_desc = load_raw_dataset(uploaded_file)
    
    if df_raw is None:
        st.error("⚠️ Không tìm thấy file dữ liệu HOSE_2020_2023_in.csv. Vui lòng tải file lên.")
        st.stop()
        
    all_tickers = sorted(df_raw["ticker"].unique())
    st.caption(f"📁 {data_source_desc}")
    st.caption(f"Tổng số mã có sẵn: **{len(all_tickers)}** mã")
    
    # 2. Chọn Mã Cổ Phiếu
    st.subheader("2. Danh Mục Cổ Phiếu")
    default_selected = [t for t in ["ACB", "FPT", "HPG"] if t in all_tickers]
    if not default_selected:
        default_selected = all_tickers[:3]
        
    selected_tickers = st.multiselect(
        "Chọn các mã cổ phiếu phân tích:",
        options=all_tickers,
        default=default_selected,
        help="Khuyến nghị chọn tối thiểu 2-3 mã đại diện các ngành khác nhau."
    )
    
    if len(selected_tickers) < 1:
        st.warning("Vui lòng chọn ít nhất 1 mã cổ phiếu.")
        st.stop()
        
    # 3. Phân Chia Khoảng Thời Gian
    st.subheader("3. Chu Kỳ Phân Tích (Train/Test)")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        train_start = st.date_input("Train Bắt đầu", pd.to_datetime("2020-01-01"))
        train_end   = st.date_input("Train Kết thúc", pd.to_datetime("2021-12-31"))
    with col_d2:
        test_start  = st.date_input("Test Bắt đầu", pd.to_datetime("2022-01-01"))
        test_end    = st.date_input("Test Kết thúc", pd.to_datetime("2022-12-31"))
        
    if train_start >= train_end or test_start >= test_end or train_end > test_start:
        st.error("Cảnh báo: Thời gian không hợp lệ. Train phải diễn ra trước Test!")
        st.stop()
        
    # 4. Tham Số Giao Dịch & Danh Mục
    st.subheader("4. Tham Số Đầu Tư")
    portfolio_mode = st.selectbox(
        "Cơ chế kết hợp tín hiệu cho Danh Mục:",
        options=["OR", "AND"],
        index=0,
        help="OR: Nới lỏng - Vào lệnh khi SMA hoặc OBV báo mua. AND: Chặt chẽ - Chỉ vào lệnh khi cả 2 cùng đồng thuận."
    )
    
    commission = st.number_input(
        "Phí giao dịch mỗi lượt (%) :",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Phí giao dịch và thuế tính trên vòng quay vị thế."
    ) / 100.0
    
    risk_free_rate = st.number_input(
        "Lãi suất phi rủi ro năm (%) :",
        min_value=0.0,
        max_value=15.0,
        value=0.0,
        step=0.5,
        help="Lãi suất phi rủi ro (Risk-Free Rate) dùng tính Sharpe Ratio trong MPT."
    ) / 100.0
    
    initial_capital = st.number_input(
        "Vốn đầu tư ban đầu (VNĐ):",
        min_value=10_000_000,
        max_value=10_000_000_000,
        value=1_000_000_000,
        step=50_000_000,
        format="%d"
    )


# ==========================================
# TIỀN XỬ LÝ DỮ LIỆU CÁC MÃ ĐÃ CHỌN
# ==========================================

train_data = {}
test_data = {}
full_stock_data = {}

for ticker in selected_tickers:
    df_stock = prepare_stock_data(df_raw, ticker)
    if df_stock.empty:
        st.error(f"Mã {ticker} không có dữ liệu hợp lệ trong file.")
        st.stop()
    full_stock_data[ticker] = df_stock
    
    tr = df_stock.loc[(df_stock.index >= pd.to_datetime(train_start)) & (df_stock.index <= pd.to_datetime(train_end))].copy()
    te = df_stock.loc[(df_stock.index >= pd.to_datetime(test_start)) & (df_stock.index <= pd.to_datetime(test_end))].copy()
    
    if tr.empty or te.empty:
        st.error(f"Mã {ticker} thiếu dữ liệu trong khoảng Train hoặc Test được chọn.")
        st.stop()
        
    train_data[ticker] = tr
    test_data[ticker] = te

# Đảm bảo các mã mới chọn đều có tham số khởi tạo
for ticker in selected_tickers:
    if ticker not in st.session_state["stock_params"]:
        st.session_state["stock_params"][ticker] = {
            "SMA": {"ma_short": 50, "ma_long": 200},
            "OBV": {"obv_window": 20}
        }


# ==========================================
# TIÊU ĐỀ CHÍNH & TÓM TẮT DỰ ÁN
# ==========================================

st.markdown('<div class="main-title">📈 Hệ Thống Kiểm Định Chiến Lược SMA + OBV & Tối Ưu Hóa MPT</div>', unsafe_allow_html=True)
st.markdown("""
<div class="sub-title">
Ứng dụng định lượng tài chính phục vụ kiểm định chiến lược giao dịch kết hợp kỹ thuật (Đường trung bình SMA & Dòng tiền On-Balance Volume) 
kèm phân bổ tỷ trọng danh mục Hiện Đại (Modern Portfolio Theory - Markowitz) trên dữ liệu Sở Giao dịch Chứng khoán TP.HCM (HOSE).
</div>
""", unsafe_allow_html=True)

# Hiển thị thanh trạng thái tổng quan
col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    st.markdown(f'<span class="badge badge-blue">📌 Danh mục: {", ".join(selected_tickers)}</span>', unsafe_allow_html=True)
with col_b2:
    st.markdown(f'<span class="badge badge-green">⏱️ Train: {train_start} ➜ {train_end}</span>', unsafe_allow_html=True)
with col_b3:
    st.markdown(f'<span class="badge badge-purple">🎯 Test (Out-of-sample): {test_start} ➜ {test_end}</span>', unsafe_allow_html=True)
with col_b4:
    st.markdown(f'<span class="badge badge-blue">⚙️ Tín hiệu Danh mục: SMA + OBV {portfolio_mode}</span>', unsafe_allow_html=True)

st.write("")

# ==========================================
# CÁC TAB CHỨC NĂNG CHÍNH
# ==========================================

tab_portfolio, tab_single, tab_optimize, tab_logs, tab_theory = st.tabs([
    "📊 Phân Tích Danh Mục (MPT vs EW)",
    "🔍 Chi Tiết Từng Cổ Phiếu",
    "⚙️ Cấu Hình & Tối Ưu Tham Số",
    "📜 Nhật Ký & Xuất Dữ Liệu",
    "📚 Cơ Sở Phương Pháp Luận"
])


# ==========================================
# HÀM ĐÁNH GIÁ TỔNG THỂ CHO CÁC MÃ
# ==========================================

def evaluate_all_stocks(stock_dict, params_dict, comm):
    """Tính toán kết quả 4 chiến lược cho tất cả cổ phiếu."""
    results = {}
    for ticker, df in stock_dict.items():
        sma_p = params_dict[ticker]["SMA"]
        obv_p = params_dict[ticker]["OBV"]
        
        strats = {}
        events_sma = find_position_sma(df, sma_p)
        r_sma, h_sma = strategy_returns(df, events_sma, comm)
        strats["SMA"] = r_sma
        
        events_obv = find_position_obv(df, obv_p)
        r_obv, h_obv = strategy_returns(df, events_obv, comm)
        strats["OBV"] = r_obv
        
        events_and = find_position_combined_and(df, sma_p, obv_p)
        r_and, h_and = strategy_returns(df, events_and, comm)
        strats["SMA + OBV AND"] = r_and
        
        events_or = find_position_combined_or(df, sma_p, obv_p)
        r_or, h_or = strategy_returns(df, events_or, comm)
        strats["SMA + OBV OR"] = r_or
        
        # Chiến lược Mua và Nắm giữ (Buy & Hold) để đối sánh
        asset_ret = df["Close"].pct_change().fillna(0.0)
        strats["Buy & Hold"] = asset_ret
        
        stats = pd.DataFrame({
            name: performance_stats(ret)
            for name, ret in strats.items()
        }).T
        
        results[ticker] = {
            "returns": strats,
            "holdings": {"SMA": h_sma, "OBV": h_obv, "AND": h_and, "OR": h_or},
            "stats": stats
        }
    return results


# Tính toán kết quả
train_eval = evaluate_all_stocks(train_data, st.session_state["stock_params"], commission)
test_eval  = evaluate_all_stocks(test_data, st.session_state["stock_params"], commission)

strategy_choice_name = f"SMA + OBV {portfolio_mode.upper()}"

# Tạo ma trận lợi nhuận
train_ret_matrix = pd.concat(
    {t: train_eval[t]["returns"][strategy_choice_name] for t in selected_tickers},
    axis=1
).dropna()

test_ret_matrix = pd.concat(
    {t: test_eval[t]["returns"][strategy_choice_name] for t in selected_tickers},
    axis=1
).dropna()

# Trọng số Equal Weight
n_assets = len(selected_tickers)
equal_weights = np.repeat(1.0 / n_assets, n_assets)

# Tối ưu trọng số MPT từ dữ liệu TRAIN duy nhất
mpt_weights = optimize_mpt(train_ret_matrix, risk_free_rate=risk_free_rate)

# Chuỗi lợi nhuận danh mục
ew_train_ret = portfolio_returns(train_ret_matrix, equal_weights)
mpt_train_ret = portfolio_returns(train_ret_matrix, mpt_weights)

ew_test_ret = portfolio_returns(test_ret_matrix, equal_weights)
mpt_test_ret = portfolio_returns(test_ret_matrix, mpt_weights)

# Thống kê hiệu quả danh mục
train_ew_stats = performance_stats(ew_train_ret)
train_mpt_stats = performance_stats(mpt_train_ret)

test_ew_stats = performance_stats(ew_test_ret)
test_mpt_stats = performance_stats(mpt_test_ret)

port_compare_train = pd.DataFrame({"Equal Weight": train_ew_stats, "MPT (Markowitz)": train_mpt_stats}).T
port_compare_test  = pd.DataFrame({"Equal Weight": test_ew_stats, "MPT (Markowitz)": test_mpt_stats}).T


# ==========================================
# TAB 1: PHÂN TÍCH DANH MỤC
# ==========================================
with tab_portfolio:
    st.subheader("1. Hiệu Suất Ngoài Mẫu (Out-of-sample Test 2022)")
    st.caption("Đây là kết quả quan trọng nhất: Áp dụng nguyên vẹn tham số tối ưu và trọng số tìm từ Train (2020-2021) vào dữ liệu chưa từng thấy Test (2022).")
    
    # KPI Metric Cards cho tập TEST
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tổng Lợi Nhuận (Test) - EW</div>
            <div class="metric-value">{test_ew_stats['Total Return [%]']:.2f}%</div>
            <div class="metric-title">MPT: <b>{test_mpt_stats['Total Return [%]']:.2f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tỷ Số Sharpe (Test) - EW</div>
            <div class="metric-value">{test_ew_stats['Sharpe Ratio']:.2f}</div>
            <div class="metric-title">MPT: <b>{test_mpt_stats['Sharpe Ratio']:.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Biến Động Hàng Năm (Test) - EW</div>
            <div class="metric-value">{test_ew_stats['Annual Volatility [%]']:.2f}%</div>
            <div class="metric-title">MPT: <b>{test_mpt_stats['Annual Volatility [%]']:.2f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mức Sụt Giảm Tối Đa (Test) - EW</div>
            <div class="metric-value">{test_ew_stats['Max Drawdown [%]']:.2f}%</div>
            <div class="metric-title">MPT: <b>{test_mpt_stats['Max Drawdown [%]']:.2f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    # Biểu đồ đường cong vốn danh mục (Equity Curve)
    col_chart_left, col_chart_right = st.columns([7, 3])
    
    with col_chart_left:
        period_choice = st.radio("Chọn giai đoạn hiển thị biểu đồ vốn:", ["Tập Test (Ngoài Mẫu)", "Tập Train (Trong Mẫu)", "Toàn Bộ Chu Kỳ"], horizontal=True)
        
        if period_choice == "Tập Test (Ngoài Mẫu)":
            ew_curve = initial_capital * (1.0 + ew_test_ret).cumprod()
            mpt_curve = initial_capital * (1.0 + mpt_test_ret).cumprod()
            chart_title = f"Đường Cong Tăng Trưởng Vốn (Equity Curve) - Tập TEST ({strategy_choice_name})"
        elif period_choice == "Tập Train (Trong Mẫu)":
            ew_curve = initial_capital * (1.0 + ew_train_ret).cumprod()
            mpt_curve = initial_capital * (1.0 + mpt_train_ret).cumprod()
            chart_title = f"Đường Cong Tăng Trưởng Vốn (Equity Curve) - Tập TRAIN ({strategy_choice_name})"
        else:
            full_ret_matrix = pd.concat([train_ret_matrix, test_ret_matrix])
            ew_full = portfolio_returns(full_ret_matrix, equal_weights)
            mpt_full = portfolio_returns(full_ret_matrix, mpt_weights)
            ew_curve = initial_capital * (1.0 + ew_full).cumprod()
            mpt_curve = initial_capital * (1.0 + mpt_full).cumprod()
            chart_title = f"Đường Cong Tăng Trưởng Vốn - Toàn Bộ Chu Kỳ ({strategy_choice_name})"
            
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=ew_curve.index, y=ew_curve.values,
            mode='lines', name='Equal Weight (Tỷ trọng đều 1/N)',
            line=dict(color='#2563EB', width=2.5)
        ))
        fig_equity.add_trace(go.Scatter(
            x=mpt_curve.index, y=mpt_curve.values,
            mode='lines', name='MPT (Tối ưu Sharpe Markowitz)',
            line=dict(color='#10B981', width=2.5)
        ))
        
        # Mốc vốn ban đầu
        fig_equity.add_hline(y=initial_capital, line_dash="dash", line_color="#9CA3AF", annotation_text="Vốn ban đầu")
        
        fig_equity.update_layout(
            title=chart_title,
            xaxis_title="Thời gian",
            yaxis_title="Giá trị Danh mục (VNĐ)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20),
            height=430
        )
        st.plotly_chart(fig_equity, use_container_width=True)
        
    with col_chart_right:
        st.write("**So Sánh Phân Bổ Trọng Số**")
        df_weights = pd.DataFrame({
            "Mã CK": selected_tickers,
            "Equal Weight": equal_weights,
            "MPT": mpt_weights
        })
        
        fig_weights = go.Figure()
        fig_weights.add_trace(go.Bar(
            x=df_weights["Mã CK"], y=df_weights["Equal Weight"],
            name="Equal Weight", marker_color='#93C5FD'
        ))
        fig_weights.add_trace(go.Bar(
            x=df_weights["Mã CK"], y=df_weights["MPT"],
            name="MPT", marker_color='#34D399'
        ))
        fig_weights.update_layout(
            barmode='group',
            yaxis=dict(title="Tỷ trọng phân bổ", tickformat=".1%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20),
            height=430
        )
        st.plotly_chart(fig_weights, use_container_width=True)
        
    # Bảng số liệu chi tiết đối sánh Train và Test
    st.write("---")
    st.subheader("2. Bảng Đối So sánh Toàn Diện Hiệu Quả Đầu Tư")
    
    col_tbl1, col_tbl2 = st.columns(2)
    with col_tbl1:
        st.write("📌 **TẬP HUẤN LUYỆN (TRAIN 2020 - 2021)**")
        st.dataframe(port_compare_train.style.format({
            "Total Return [%]": "{:.2f}%",
            "Annual Return [%]": "{:.2f}%",
            "Annual Volatility [%]": "{:.2f}%",
            "Sharpe Ratio": "{:.3f}",
            "Max Drawdown [%]": "{:.2f}%",
            "Win Rate [%]": "{:.2f}%"
        }), use_container_width=True)
        
    with col_tbl2:
        st.write("📌 **TẬP KIỂM ĐỊNH NGOÀI MẪU (TEST 2022)**")
        st.dataframe(port_compare_test.style.format({
            "Total Return [%]": "{:.2f}%",
            "Annual Return [%]": "{:.2f}%",
            "Annual Volatility [%]": "{:.2f}%",
            "Sharpe Ratio": "{:.3f}",
            "Max Drawdown [%]": "{:.2f}%",
            "Win Rate [%]": "{:.2f}%"
        }), use_container_width=True)
        
    # Biểu đồ Drawdown (Mức sụt giảm từ đỉnh)
    st.write("---")
    st.subheader("3. Biểu Đồ Mức Sụt Giảm Tài Khoản (Underwater Drawdown Chart)")
    
    def calc_dd(cum_ret_series):
        eq = (1.0 + cum_ret_series).cumprod()
        rm = eq.cummax()
        return (eq / rm - 1.0) * 100.0
        
    dd_ew = calc_dd(ew_test_ret)
    dd_mpt = calc_dd(mpt_test_ret)
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd_ew.index, y=dd_ew.values,
        mode='lines', name='Equal Weight Drawdown',
        line=dict(color='#DC2626', width=1.8),
        fill='tozeroy'
    ))
    fig_dd.add_trace(go.Scatter(
        x=dd_mpt.index, y=dd_mpt.values,
        mode='lines', name='MPT Drawdown',
        line=dict(color='#D97706', width=1.8)
    ))
    fig_dd.update_layout(
        title="Mức sụt giảm từ đỉnh cao nhất (Drawdown %) trên Tập TEST 2022",
        xaxis_title="Thời gian",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        height=320
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ==========================================
# TAB 2: CHI TIẾT TỪNG CỔ PHIẾU
# ==========================================
with tab_single:
    st.subheader("Phân Tích Kỹ Thuật & Hiệu Suất Từng Mã")
    
    selected_stock = st.selectbox("Chọn mã cổ phiếu cần xem chi tiết:", selected_tickers)
    df_stock = full_stock_data[selected_stock]
    
    stock_p = st.session_state["stock_params"][selected_stock]
    sma_short = stock_p["SMA"]["ma_short"]
    sma_long = stock_p["SMA"]["ma_long"]
    obv_win = stock_p["OBV"]["obv_window"]
    
    col_pinfo1, col_pinfo2, col_pinfo3 = st.columns(3)
    with col_pinfo1:
        st.info(f"SMA Ngắn hạn: **{sma_short} phiên** | SMA Dài hạn: **{sma_long} phiên**")
    with col_pinfo2:
        st.info(f"Chu kỳ OBV Moving Average: **{obv_win} phiên**")
    with col_pinfo3:
        st.info(f"Chiến lược kết hợp: **{portfolio_mode.upper()}**")
        
    # Tính toán chỉ báo cho cổ phiếu
    ma_s_series = calculate_sma(df_stock["Close"], sma_short)
    ma_l_series = calculate_sma(df_stock["Close"], sma_long)
    obv_series = calculate_obv(df_stock["Close"], df_stock["Volume"])
    obv_ma_series = obv_series.rolling(window=obv_win).mean()
    
    # Tín hiệu
    pos_combined = find_position_combined_or(df_stock, stock_p["SMA"], stock_p["OBV"]) if portfolio_mode == "OR" else find_position_combined_and(df_stock, stock_p["SMA"], stock_p["OBV"])
    buy_signals = df_stock.loc[pos_combined == 1.0]
    sell_signals = df_stock.loc[pos_combined == -1.0]
    
    # Biểu đồ kỹ thuật tương tác gồm 2 phân vùng (Price + SMA, OBV + OBV-MA)
    fig_tech = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=[f"Biểu đồ Giá & Tín hiệu Giao dịch: {selected_stock}", "Chỉ báo Dòng Tiền On-Balance Volume (OBV)"]
    )
    
    # Đường giá
    fig_tech.add_trace(go.Scatter(
        x=df_stock.index, y=df_stock["Close"],
        mode='lines', name='Giá Đóng Cửa',
        line=dict(color='#1E293B', width=1.5)
    ), row=1, col=1)
    
    # SMA ngắn & dài
    fig_tech.add_trace(go.Scatter(
        x=ma_s_series.index, y=ma_s_series.values,
        mode='lines', name=f'SMA {sma_short}',
        line=dict(color='#F59E0B', width=1.8)
    ), row=1, col=1)
    
    fig_tech.add_trace(go.Scatter(
        x=ma_l_series.index, y=ma_l_series.values,
        mode='lines', name=f'SMA {sma_long}',
        line=dict(color='#3B82F6', width=1.8)
    ), row=1, col=1)
    
    # Điểm MUA (Buy Marker)
    if not buy_signals.empty:
        fig_tech.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals["Close"],
            mode='markers', name='Điểm MUA',
            marker=dict(symbol='triangle-up', size=11, color='#10B981', line=dict(width=1, color='black'))
        ), row=1, col=1)
        
    # Điểm BÁN (Sell Marker)
    if not sell_signals.empty:
        fig_tech.add_trace(go.Scatter(
            x=sell_signals.index, y=sell_signals["Close"],
            mode='markers', name='Điểm BÁN',
            marker=dict(symbol='triangle-down', size=11, color='#EF4444', line=dict(width=1, color='black'))
        ), row=1, col=1)
        
    # Phân vùng 2: OBV
    fig_tech.add_trace(go.Scatter(
        x=obv_series.index, y=obv_series.values,
        mode='lines', name='OBV',
        line=dict(color='#8B5CF6', width=1.5)
    ), row=2, col=1)
    
    fig_tech.add_trace(go.Scatter(
        x=obv_ma_series.index, y=obv_ma_series.values,
        mode='lines', name=f'OBV MA {obv_win}',
        line=dict(color='#EC4899', width=1.5, dash='dash')
    ), row=2, col=1)
    
    fig_tech.update_layout(
        height=620,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_tech, use_container_width=True)
    
    # So sánh 4 chiến lược trên mã này
    st.write("---")
    st.subheader(f"So Sánh 4 Chiến Lược Trên Mã {selected_stock}")
    
    col_str1, col_str2 = st.columns(2)
    with col_str1:
        st.write(f"📊 **Kết quả trên tập TRAIN ({train_start} đến {train_end})**")
        st.dataframe(train_eval[selected_stock]["stats"].style.format({
            "Total Return [%]": "{:.2f}%",
            "Annual Return [%]": "{:.2f}%",
            "Annual Volatility [%]": "{:.2f}%",
            "Sharpe Ratio": "{:.3f}",
            "Max Drawdown [%]": "{:.2f}%",
            "Win Rate [%]": "{:.2f}%"
        }), use_container_width=True)
        
    with col_str2:
        st.write(f"📊 **Kết quả trên tập TEST ({test_start} đến {test_end})**")
        st.dataframe(test_eval[selected_stock]["stats"].style.format({
            "Total Return [%]": "{:.2f}%",
            "Annual Return [%]": "{:.2f}%",
            "Annual Volatility [%]": "{:.2f}%",
            "Sharpe Ratio": "{:.3f}",
            "Max Drawdown [%]": "{:.2f}%",
            "Win Rate [%]": "{:.2f}%"
        }), use_container_width=True)


# ==========================================
# TAB 3: TỐI ƯU HÓA THAM SỐ (HYPEROPT)
# ==========================================
with tab_optimize:
    st.subheader("⚙️ Quản Lý Tham Số & Tự Động Tối Ưu Bằng Hyperopt (TPE)")
    st.markdown("""
    Hệ thống cho phép bạn:
    1. **Tự điều chỉnh thủ công**: Thay đổi trực tiếp các tham số SMA và OBV cho từng mã để kiểm tra độ nhạy (Sensitivity Analysis).
    2. **Tối ưu hóa tự động (Hyperopt)**: Sử dụng thuật toán Bayesian Optimization (TPE) để tự động tìm bộ tham số tối đa hóa tỷ số Sharpe trên tập **Train**.
    """)
    
    col_opt_left, col_opt_right = st.columns([6, 4])
    
    with col_opt_left:
        st.write("##### 1. Bảng Tham Số Hiện Tại")
        current_params_df = pd.DataFrame({
            ticker: {
                "SMA Ngắn (ma_short)": st.session_state["stock_params"][ticker]["SMA"]["ma_short"],
                "SMA Dài (ma_long)": st.session_state["stock_params"][ticker]["SMA"]["ma_long"],
                "OBV Window (obv_window)": st.session_state["stock_params"][ticker]["OBV"]["obv_window"]
            }
            for ticker in selected_tickers
        }).T
        st.dataframe(current_params_df, use_container_width=True)
        
        st.write("##### 2. Điều Chỉnh Thủ Công Từng Mã")
        stock_to_edit = st.selectbox("Chọn mã cần tinh chỉnh tham số:", selected_tickers, key="edit_ticker")
        
        col_ed1, col_ed2, col_ed3 = st.columns(3)
        with col_ed1:
            new_short = st.number_input(
                f"SMA Ngắn ({stock_to_edit}):",
                min_value=5, max_value=200,
                value=int(st.session_state["stock_params"][stock_to_edit]["SMA"]["ma_short"]),
                step=5
            )
        with col_ed2:
            new_long = st.number_input(
                f"SMA Dài ({stock_to_edit}):",
                min_value=50, max_value=500,
                value=int(st.session_state["stock_params"][stock_to_edit]["SMA"]["ma_long"]),
                step=5
            )
        with col_ed3:
            new_obv = st.number_input(
                f"OBV Window ({stock_to_edit}):",
                min_value=2, max_value=150,
                value=int(st.session_state["stock_params"][stock_to_edit]["OBV"]["obv_window"]),
                step=5
            )
            
        if st.button("Cập Nhật Tham Số Mã Này"):
            if new_short >= new_long:
                st.error("SMA ngắn phải nhỏ hơn SMA dài!")
            else:
                st.session_state["stock_params"][stock_to_edit]["SMA"]["ma_short"] = int(new_short)
                st.session_state["stock_params"][stock_to_edit]["SMA"]["ma_long"] = int(new_long)
                st.session_state["stock_params"][stock_to_edit]["OBV"]["obv_window"] = int(new_obv)
                st.success(f"Đã cập nhật tham số cho {stock_to_edit} thành công!")
                st.rerun()

    with col_opt_right:
        st.write("##### 3. Chạy Tối Ưu Hóa Tự Động (Hyperopt)")
        evals_slider = st.slider(
            "Số vòng lặp thử nghiệm (max_evals):",
            min_value=10, max_value=100, value=30, step=10,
            help="Số vòng lặp càng cao kết quả càng tối ưu nhưng thời gian tính toán sẽ tăng theo."
        )
        
        if not HAS_HYPEROPT:
            st.warning("⚠️ Thư viện `hyperopt` chưa được cài đặt. Không thể chạy tối ưu hóa tự động.")
        else:
            if st.button("🚀 Bắt Đầu Tối Ưu Hóa (Chỉ trên Train)", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()
                
                new_optimized = {}
                for idx, t in enumerate(selected_tickers):
                    status_text.text(f"Đang tối ưu hóa mã {t} ({idx+1}/{len(selected_tickers)})...")
                    best_sma, best_obv = optimize_one_stock(
                        train_data[t],
                        max_evals=evals_slider,
                        commission=commission
                    )
                    new_optimized[t] = {
                        "SMA": best_sma,
                        "OBV": best_obv
                    }
                    progress_bar.progress((idx + 1) / len(selected_tickers))
                    
                st.session_state["stock_params"].update(new_optimized)
                elapsed = time.time() - start_time
                status_text.text(f"Hoàn thành tối ưu trong {elapsed:.1f} giây!")
                st.success("Đã tối ưu hóa xong toàn bộ danh mục! Dữ liệu biểu đồ và hiệu quả đã được cập nhật tự động.")
                st.rerun()
                
        if st.button("🔄 Khôi Phục Tham Số Mặc Định (Từ Notebook)"):
            for t in DEFAULT_PARAMS:
                if t in st.session_state["stock_params"]:
                    st.session_state["stock_params"][t] = DEFAULT_PARAMS[t].copy()
            st.success("Đã khôi phục tham số mặc định của notebook.")
            st.rerun()


# ==========================================
# TAB 4: NHẬT KÝ & XUẤT DỮ LIỆU
# ==========================================
with tab_logs:
    st.subheader("📜 Nhật Ký Giao Dịch & Xuất Dữ Liệu Lợi Nhuận")
    
    col_log1, col_log2 = st.columns(2)
    with col_log1:
        st.write("##### Ma Trận Lợi Nhuận Hàng Ngày (Test)")
        st.dataframe(test_ret_matrix.tail(20), use_container_width=True)
        
        # Nút xuất CSV cho chuỗi lợi nhuận
        csv_buffer = io.StringIO()
        test_ret_matrix.to_csv(csv_buffer)
        st.download_button(
            label="📥 Tải xuống Ma trận Lợi nhuận Test (CSV)",
            data=csv_buffer.getvalue(),
            file_name="test_return_matrix.csv",
            mime="text/csv"
        )
        
    with col_log2:
        st.write("##### Bảng Thống Kê Hiệu Năng Danh Mục (Test)")
        st.dataframe(port_compare_test, use_container_width=True)
        
        csv_stats = io.StringIO()
        port_compare_test.to_csv(csv_stats)
        st.download_button(
            label="📥 Tải xuống Bảng Chỉ số Hiệu năng Test (CSV)",
            data=csv_stats.getvalue(),
            file_name="portfolio_test_performance.csv",
            mime="text/csv"
        )

    st.write("---")
    st.write("##### Tín hiệu Giao dịch Gần nhất (15 phiên cuối)")
    recent_signals = []
    for t in selected_tickers:
        pos = find_position_combined_or(full_stock_data[t], st.session_state["stock_params"][t]["SMA"], st.session_state["stock_params"][t]["OBV"])
        h = events_to_holding(pos)
        tail_df = pd.DataFrame({
            "Mã CK": t,
            "Ngày": pos.tail(15).index.strftime("%Y-%m-%d"),
            "Giá Đóng": full_stock_data[t]["Close"].tail(15).values,
            "Tín Hiệu Rời Rạc": pos.tail(15).values,
            "Vị Thế Nắm Giữ": h.tail(15).values
        })
        recent_signals.append(tail_df)
    st.dataframe(pd.concat(recent_signals).reset_index(drop=True), use_container_width=True)


# ==========================================
# TAB 5: LÝ THUYẾT & PHƯƠNG PHÁP LUẬN
# ==========================================
with tab_theory:
    st.markdown("""
    ### 📖 Cơ Sở Lý Thuyết & Phương Pháp Luận Định Lượng

    #### 1. Chiến Lược Giao Dịch Kết Hợp SMA & OBV
    * **Simple Moving Average (SMA)**: Chỉ báo xu hướng trễ kinh điển.
      - **Tín hiệu MUA**: Đường SMA ngắn cắt lên trên đường SMA dài ($SMA_{short} > SMA_{long}$).
      - **Tín hiệu BÁN**: Đường SMA ngắn cắt xuống dưới đường SMA dài ($SMA_{short} < SMA_{long}$).
    * **On-Balance Volume (OBV)**: Chỉ báo động lượng khối lượng luỹ kế do Joe Granville phát triển, đo lường áp lực mua/bán của dòng tiền thông minh trước khi giá biến động.
      - $OBV_t = OBV_{t-1} + Volume_t$ nếu $Close_t > Close_{t-1}$
      - $OBV_t = OBV_{t-1} - Volume_t$ nếu $Close_t < Close_{t-1}$
      - **Tín hiệu MUA**: Đường OBV vượt lên đường trung bình $OBV\_MA$.
      - **Tín hiệu BÁN**: Đường OBV gãy xuống dưới đường trung bình $OBV\_MA$.
    * **Cơ Chế Kết Hợp**:
      - **Chế độ AND**: Yêu cầu xác nhận đồng thời từ cả Xu hướng (SMA) và Dòng tiền (OBV), giảm số lượng lệnh nhưng gia tăng độ chính xác.
      - **Chế độ OR**: Tận dụng độ nhạy sớm của dòng tiền OBV hoặc xu hướng bền vững của SMA. Khi có tín hiệu mâu thuẫn (vừa Mua vừa Bán), đưa vị thế về 0 (Trung lập).

    #### 2. Triệt Tiêu Tuyệt Đối Look-Ahead Bias (Thiên Kiến Nhìn Trước Tương Lai)
    * Trong thực tế giao dịch, tín hiệu ngày $t$ chỉ được tính toán đầy đủ sau khi thị trường đóng cửa (14h45 tại HOSE).
    * Do đó, nhà đầu tư không thể mua được ở mức giá đóng cửa của ngày $t$.
    * Hệ thống đã thực hiện **Dịch Chuyển 1 Phiên (`shift(1)`)**: Vị thế nắm giữ xác lập tại ngày $t$ chỉ bắt đầu sinh lời từ ngày $t+1$.

    #### 3. Tối Ưu Hóa Danh Mục Hiện Đại (Modern Portfolio Theory - MPT)
    * Thay vì phân bổ tỷ trọng cào bằng **Equal Weight** ($w_i = 1/N$), Markowitz MPT tối ưu hóa vector trọng số $W = [w_1, w_2, ..., w_N]^T$ bằng thuật toán phi tuyến **SLSQP**:
      $$\\max_{W} \\frac{W^T \\mu - r_f}{\\sqrt{W^T \\Sigma W}}$$
      * Với $\\mu$ là vector lợi nhuận kỳ vọng của các chiến lược trên tập Train.
      * $\\Sigma$ là ma trận hiệp phương sai lợi nhuận chiến lược trên tập Train.
      * Ràng buộc: $0 \\le w_i \\le 1$ (Không bán khống - Long Only) và $\\sum w_i = 1$.

    #### 4. Ý Nghĩa Phân Tách Train (2020-2021) và Test (2022)
    * Giai đoạn **Train 2020 - 2021**: Thị trường Uptrend mạnh mẽ hậu Covid.
    * Giai đoạn **Test 2022**: Thị trường Downtrend khốc liệt (VN-Index giảm hơn 33%).
    * Việc kiểm định trên dữ liệu Test 2022 giúp đánh giá khách quan khả năng quản trị rủi ro và bảo vệ vốn của chiến lược khi bước vào điều kiện thị trường gấu khắc nghiệt.
    """)

# Footer
st.markdown("---")
st.caption("Ứng dụng Định lượng Tài chính | Triển khai trên Streamlit Cloud | Tương thích mã nguồn Notebook")
