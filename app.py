import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Crypto Signal Pro", layout="wide")
st.title("📈 Crypto Signal Pro – 15m Entry with HTF Trend")
st.markdown("Professional‑grade signals with entry zone, multiple TPs, stop loss, leverage recommendation, and position sizing.")

# -------------------- Sidebar Settings --------------------
with st.sidebar:
    st.header("⚙️ Settings")
    account_balance = st.number_input("Account Balance (USDT)", value=1000, step=100)
    risk_percent = st.slider("Risk per trade (%)", 0.5, 5.0, 2.0) / 100
    max_leverage = st.selectbox("Max Leverage", [50, 100, 200, 300, 500], index=1)

    st.markdown("---")
    st.header("🕒 Timeframes")
    tf_entry = st.selectbox("Entry Timeframe", ["15m"], index=0, disabled=True)  # fixed to 15m
    tf_trend = st.selectbox("Trend Timeframe", ["1h", "4h"], index=0)

    st.markdown("---")
    st.header("📊 Pairs")
    pairs = st.multiselect(
        "Select Pairs",
        ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT"],
        default=["BTC/USDT", "ETH/USDT"]
    )

    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# -------------------- Data Fetching --------------------
@st.cache_data(ttl=60)
def fetch_ohlcv(symbol, timeframe, limit=500):
    """Fetch OHLCV data from Binance (public)"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 10000})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.warning(f"Error fetching {symbol}: {e}")
        return None

# -------------------- Indicator Functions --------------------
def add_indicators(df):
    """Add RSI, ATR, EMAs to dataframe"""
    close = df['close']
    high = df['high']
    low = df['low']

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # EMAs
    df['ema20'] = close.ewm(span=20).mean()
    df['ema50'] = close.ewm(span=50).mean()
    df['ema200'] = close.ewm(span=200).mean()

    return df

def detect_pivot_points(df, window=5):
    """Return lists of (timestamp, price) for swing highs and lows"""
    highs = df['high'].values
    lows = df['low'].values
    timestamps = df['timestamp'].values
    pivot_highs = []
    pivot_lows = []
    for i in range(window, len(df)-window):
        if highs[i] == max(highs[i-window:i+window+1]):
            pivot_highs.append((timestamps[i], highs[i]))
        if lows[i] == min(lows[i-window:i+window+1]):
            pivot_lows.append((timestamps[i], lows[i]))
    return pivot_highs, pivot_lows

def find_support_resistance(df, window=10, tolerance=0.005):
    """Group pivot points into key S/R levels"""
    pivots_high, pivots_low = detect_pivot_points(df, window)
    levels = [price for _, price in pivots_high] + [price for _, price in pivots_low]
    if not levels:
        return []
    levels.sort()
    grouped = []
    current_group = [levels[0]]
    for l in levels[1:]:
        if l <= current_group[-1] * (1 + tolerance):
            current_group.append(l)
        else:
            grouped.append(np.mean(current_group))
            current_group = [l]
    grouped.append(np.mean(current_group))
    return grouped

def find_trend_line(df, pivot_points, ascending=True):
    """Fit a line through pivot points (simplified)"""
    if len(pivot_points) < 2:
        return None
    # Use last two pivots
    x1, y1 = pivot_points[-2]
    x2, y2 = pivot_points[-1]
    if (ascending and y2 > y1) or (not ascending and y2 < y1):
        slope = (y2 - y1) / ((x2 - x1).total_seconds())
        intercept = y1 - slope * x1.timestamp()
        return slope, intercept
    return None

# -------------------- Signal Generation --------------------
def generate_signal(pair, df_entry, df_trend):
    """Return a signal dict or None"""
    last_entry = df_entry.iloc[-1]
    last_trend = df_trend.iloc[-1]

    # Higher timeframe trend (200 EMA)
    htf_trend_up = last_trend['close'] > last_trend['ema200']
    htf_trend_down = last_trend['close'] < last_trend['ema200']

    # Detect HTF support/resistance levels
    htf_levels = find_support_resistance(df_trend, window=10, tolerance=0.005)
    if not htf_levels:
        htf_levels = []

    # Find nearest support and resistance
    support = max([l for l in htf_levels if l < last_entry['close']], default=None)
    resistance = min([l for l in htf_levels if l > last_entry['close']], default=None)

    # Detect HTF trend lines
    pivots_high, pivots_low = detect_pivot_points(df_trend, window=5)
    uptrend_line = find_trend_line(df_trend, pivots_low, ascending=True)
    downtrend_line = find_trend_line(df_trend, pivots_high, ascending=False)

    # Check proximity to trend lines (within 1%)
    near_uptrend = False
    near_downtrend = False
    if uptrend_line:
        slope, intercept = uptrend_line
        trend_price = slope * df_entry['timestamp'].iloc[-1].timestamp() + intercept
        near_uptrend = abs(last_entry['close'] - trend_price) / last_entry['close'] < 0.01
    if downtrend_line:
        slope, intercept = downtrend_line
        trend_price = slope * df_entry['timestamp'].iloc[-1].timestamp() + intercept
        near_downtrend = abs(last_entry['close'] - trend_price) / last_entry['close'] < 0.01

    # Candle patterns (simple)
    body = abs(last_entry['close'] - last_entry['open'])
    lower_shadow = last_entry['open'] - last_entry['low'] if last_entry['close'] > last_entry['open'] else last_entry['close'] - last_entry['low']
    upper_shadow = last_entry['high'] - last_entry['close'] if last_entry['close'] > last_entry['open'] else last_entry['high'] - last_entry['open']
    hammer = lower_shadow > body * 2 and upper_shadow < body * 0.3
    shooting_star = upper_shadow > body * 2 and lower_shadow < body * 0.3

    # Entry conditions
    long_signal = False
    short_signal = False
    reason = []

    # Long: HTF uptrend + price near support or uptrend line + bullish candle
    if htf_trend_up and (support or near_uptrend):
        if hammer or last_entry['rsi'] < 40:
            long_signal = True
            if support:
                reason.append(f"near HTF support {support:.2f}")
            if near_uptrend:
                reason.append("near HTF uptrend line")
            if hammer:
                reason.append("hammer candle")
            if last_entry['rsi'] < 40:
                reason.append(f"RSI {last_entry['rsi']:.1f} (oversold)")

    # Short: HTF downtrend + price near resistance or downtrend line + bearish candle
    if htf_trend_down and (resistance or near_downtrend):
        if shooting_star or last_entry['rsi'] > 60:
            short_signal = True
            if resistance:
                reason.append(f"near HTF resistance {resistance:.2f}")
            if near_downtrend:
                reason.append("near HTF downtrend line")
            if shooting_star:
                reason.append("shooting star")
            if last_entry['rsi'] > 60:
                reason.append(f"RSI {last_entry['rsi']:.1f} (overbought)")

    if not (long_signal or short_signal):
        return None

    # Determine direction and compute levels
    direction = "LONG" if long_signal else "SHORT"
    entry = last_entry['close']
    atr = last_entry['atr']

    # Entry zone: entry ± 0.5*ATR
    entry_zone_low = entry - atr * 0.5
    entry_zone_high = entry + atr * 0.5

    # Stop loss: for long, below support or recent low; for short, above resistance or recent high
    if direction == "LONG":
        sl = min(support, last_entry['low'] * 0.99) if support else last_entry['low'] * 0.99
    else:
        sl = max(resistance, last_entry['high'] * 1.01) if resistance else last_entry['high'] * 1.01

    # Take profits (ATR multiples)
    tp1 = entry + atr * 2 if direction == "LONG" else entry - atr * 2
    tp2 = entry + atr * 3.5 if direction == "LONG" else entry - atr * 3.5
    tp3 = entry + atr * 5 if direction == "LONG" else entry - atr * 5

    # Confidence score / grade (0-100)
    score = 50
    if htf_trend_up or htf_trend_down:
        score += 15
    if support or resistance or near_uptrend or near_downtrend:
        score += 15
    if hammer or shooting_star:
        score += 10
    if last_entry['rsi'] < 40 or last_entry['rsi'] > 60:
        score += 10
    score = min(score, 100)

    if score >= 90:
        grade = "A+"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B+"
    elif score >= 60:
        grade = "B"
    elif score >= 50:
        grade = "C+"
    else:
        grade = "C"

    # Leverage suggestion: based on ATR% (lower ATR% → higher leverage)
    atr_percent = atr / entry * 100
    if atr_percent < 0.3:
        leverage_rec = "100x – 200x"
    elif atr_percent < 0.5:
        leverage_rec = "50x – 100x"
    else:
        leverage_rec = "20x – 50x"

    # Position size
    risk_amount = account_balance * risk_percent
    stop_distance = abs(entry - sl)
    position_size = risk_amount / stop_distance if stop_distance > 0 else 0
    position_percent = position_size / account_balance * 100 if account_balance > 0 else 0

    # Reasoning paragraph
    reasoning = f"""
**Market Context:** {'Uptrend' if htf_trend_up else 'Downtrend'} on higher timeframe.  
**Key Levels:** {'Support near ' + f'{support:.2f}' if support else ''} {'Resistance near ' + f'{resistance:.2f}' if resistance else ''}.  
**Entry Signal:** {', '.join(reason)}.  
**Trade Plan:** Enter within {entry_zone_low:.2f}–{entry_zone_high:.2f} zone. Stop at {sl:.2f}. Target 1: {tp1:.2f}, Target 2: {tp2:.2f}, Target 3: {tp3:.2f}.  
**Risk:** {risk_percent*100:.1f}% of account = ${risk_amount:.2f}. Position size ≈ {position_percent:.1f}% of account (${position_size:.2f}).  
**Recommended Leverage:** {leverage_rec} (ATR = {atr_percent:.2f}%).
"""

    return {
        'pair': pair,
        'direction': direction,
        'entry': entry,
        'entry_zone_low': entry_zone_low,
        'entry_zone_high': entry_zone_high,
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2,
        'tp3': tp3,
        'confidence': score,
        'grade': grade,
        'leverage_rec': leverage_rec,
        'position_size': position_size,
        'position_percent': position_percent,
        'reasoning': reasoning,
        'timestamp': datetime.now()
    }

# -------------------- Main UI --------------------
signals = []
progress_bar = st.progress(0)
status_text = st.empty()

if st.button("🔍 Generate Signals Now"):
    for i, pair in enumerate(pairs):
        status_text.text(f"Analyzing {pair}...")
        df_entry = fetch_ohlcv(pair, tf_entry, 300)
        df_trend = fetch_ohlcv(pair, tf_trend, 200)
        if df_entry is not None and df_trend is not None:
            df_entry = add_indicators(df_entry)
            df_trend = add_indicators(df_trend)
            signal = generate_signal(pair, df_entry, df_trend)
            if signal:
                signals.append(signal)
        progress_bar.progress((i + 1) / len(pairs))
        time.sleep(0.5)  # avoid rate limits

    status_text.text("Analysis complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

# Display signals
if signals:
    st.subheader("📡 Current Trading Signals")
    for sig in signals:
        with st.expander(f"{sig['pair']} – {sig['direction']} (Grade: {sig['grade']})", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entry Zone", f"{sig['entry_zone_low']:.2f} – {sig['entry_zone_high']:.2f}")
                st.metric("Stop Loss", f"{sig['sl']:.2f}")
            with col2:
                st.metric("TP1", f"{sig['tp1']:.2f}")
                st.metric("TP2", f"{sig['tp2']:.2f}")
                st.metric("TP3", f"{sig['tp3']:.2f}")
            with col3:
                st.metric("Confidence", f"{sig['confidence']}%")
                st.metric("Leverage Rec", sig['leverage_rec'])
                st.metric("Position Size", f"{sig['position_percent']:.1f}% (${sig['position_size']:.2f})")
            st.markdown(sig['reasoning'])
            st.caption(f"Generated: {sig['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    # Download CSV
    df_out = pd.DataFrame([{
        'Pair': s['pair'],
        'Direction': s['direction'],
        'Entry_Zone': f"{s['entry_zone_low']:.2f}-{s['entry_zone_high']:.2f}",
        'SL': s['sl'],
        'TP1': s['tp1'],
        'TP2': s['tp2'],
        'TP3': s['tp3'],
        'Confidence': s['confidence'],
        'Grade': s['grade'],
        'Leverage_Rec': s['leverage_rec'],
        'Position_Size_USD': round(s['position_size'], 2),
        'Reasoning': s['reasoning']
    } for s in signals])
    csv = df_out.to_csv(index=False)
    st.download_button("📥 Download Signals CSV", csv, "signals.csv", mime="text/csv")

else:
    st.info("👆 Click 'Generate Signals Now' to start analysis.")

st.caption("Data source: Binance public API • Signals are for educational purposes only.")
