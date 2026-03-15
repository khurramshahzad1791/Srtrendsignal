import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pytz

# -------------------- Page Config --------------------
st.set_page_config(page_title="Crypto Pro Scanner", layout="wide")
st.title("📈 Crypto Pro Scanner – Multi‑Session, Multi‑TF, Volume‑Aware")
st.markdown("""
- **Top 100 pairs by volume** (updated hourly)
- **Session‑aware** (Pakistan time: US, London, Asia)
- **Multi‑timeframe trend** (1h/4h trend + 15m entry)
- **Volume surge detection**
- **A+ to C grade signals**
""")

# -------------------- Helper Functions --------------------
def get_pakistan_time():
    """Return current datetime in Pakistan (UTC+5)"""
    return datetime.now(pytz.timezone('Asia/Karachi'))

def get_session(now_pk):
    """Determine trading session based on hour in Pakistan time"""
    hour = now_pk.hour
    # Approximate sessions (UTC+5):
    # Asia: 0–8 (overlaps with some US after-hours)
    # London: 8–17
    # US: 17–0 (next day)
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 17:
        return "London"
    else:
        return "US"

def get_top_pairs(limit=100):
    """Fetch top USDT pairs by 24h volume from Binance (public tickers)"""
    exchange = ccxt.binance({'enableRateLimit': True})
    tickers = exchange.fetch_tickers()
    usdt_pairs = []
    for symbol, ticker in tickers.items():
        if symbol.endswith('/USDT') and ticker.get('quoteVolume'):
            usdt_pairs.append((symbol, ticker['quoteVolume']))
    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in usdt_pairs[:limit]]

@st.cache_data(ttl=3600)  # cache for 1 hour
def get_top_pairs_cached():
    return get_top_pairs(100)

@st.cache_data(ttl=300)  # cache for 5 minutes
def fetch_ohlcv(symbol, timeframe, limit=500):
    try:
        exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 10000})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.warning(f"Error fetching {symbol}: {e}")
        return None

def add_indicators(df):
    """Add RSI, ATR, EMAs to dataframe"""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

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

    # Session volume average (for same session in past 7 days)
    # For simplicity, we'll just compare current volume to 20-period average
    df['vol_ma20'] = volume.rolling(20).mean()
    df['vol_surge'] = volume / df['vol_ma20']

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
    x1, y1 = pivot_points[-2]
    x2, y2 = pivot_points[-1]
    if (ascending and y2 > y1) or (not ascending and y2 < y1):
        slope = (y2 - y1) / ((x2 - x1).total_seconds())
        intercept = y1 - slope * x1.timestamp()
        return slope, intercept
    return None

def generate_signal(pair, df_entry, df_trend, session, top_rank):
    """Return a signal dict or None"""
    last_entry = df_entry.iloc[-1]
    last_trend = df_trend.iloc[-1]

    # Higher timeframe trend
    htf_trend_up = last_trend['close'] > last_trend['ema200']
    htf_trend_down = last_trend['close'] < last_trend['ema200']

    # Detect HTF support/resistance
    htf_levels = find_support_resistance(df_trend, window=10, tolerance=0.005)
    support = max([l for l in htf_levels if l < last_entry['close']], default=None)
    resistance = min([l for l in htf_levels if l > last_entry['close']], default=None)

    # Trend lines
    pivots_high, pivots_low = detect_pivot_points(df_trend, window=5)
    uptrend_line = find_trend_line(df_trend, pivots_low, ascending=True)
    downtrend_line = find_trend_line(df_trend, pivots_high, ascending=False)

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

    # Volume surge
    vol_surge = last_entry['vol_surge'] if not pd.isna(last_entry['vol_surge']) else 1.0
    vol_ok = vol_surge > 1.5

    # Candle patterns (simple)
    body = abs(last_entry['close'] - last_entry['open'])
    lower_shadow = last_entry['open'] - last_entry['low'] if last_entry['close'] > last_entry['open'] else last_entry['close'] - last_entry['low']
    upper_shadow = last_entry['high'] - last_entry['close'] if last_entry['close'] > last_entry['open'] else last_entry['high'] - last_entry['open']
    hammer = lower_shadow > body * 2 and upper_shadow < body * 0.3
    shooting_star = upper_shadow > body * 2 and lower_shadow < body * 0.3

    # Determine direction
    long_candidates = []
    short_candidates = []

    # Long: HTF uptrend + price near support/uptrend line + (hammer or RSI <40 or volume surge)
    if htf_trend_up and (support or near_uptrend):
        score = 30
        reasons = []
        if support:
            score += 15
            reasons.append(f"near support {support:.2f}")
        if near_uptrend:
            score += 15
            reasons.append("near uptrend line")
        if hammer:
            score += 15
            reasons.append("hammer candle")
        if last_entry['rsi'] < 40:
            score += 10
            reasons.append(f"RSI {last_entry['rsi']:.1f} (oversold)")
        if vol_ok:
            score += 10
            reasons.append("volume surge")
        long_candidates.append((score, reasons))

    # Short: HTF downtrend + price near resistance/downtrend line + (shooting star or RSI >60 or volume surge)
    if htf_trend_down and (resistance or near_downtrend):
        score = 30
        reasons = []
        if resistance:
            score += 15
            reasons.append(f"near resistance {resistance:.2f}")
        if near_downtrend:
            score += 15
            reasons.append("near downtrend line")
        if shooting_star:
            score += 15
            reasons.append("shooting star")
        if last_entry['rsi'] > 60:
            score += 10
            reasons.append(f"RSI {last_entry['rsi']:.1f} (overbought)")
        if vol_ok:
            score += 10
            reasons.append("volume surge")
        short_candidates.append((score, reasons))

    # Pick the best direction
    direction = None
    best_score = 0
    best_reasons = []
    if long_candidates:
        best_long = max(long_candidates, key=lambda x: x[0])
        best_score = best_long[0]
        best_reasons = best_long[1]
        direction = "LONG"
    if short_candidates:
        best_short = max(short_candidates, key=lambda x: x[0])
        if best_short[0] > best_score:
            best_score = best_short[0]
            best_reasons = best_short[1]
            direction = "SHORT"

    if direction is None:
        return None

    # Compute entry zone, SL, TPs
    entry = last_entry['close']
    atr = last_entry['atr']
    entry_zone_low = entry - atr * 0.5
    entry_zone_high = entry + atr * 0.5

    if direction == "LONG":
        sl = support if support else last_entry['low'] * 0.99
        tp1 = entry + atr * 2
        tp2 = entry + atr * 3.5
        tp3 = entry + atr * 5
    else:
        sl = resistance if resistance else last_entry['high'] * 1.01
        tp1 = entry - atr * 2
        tp2 = entry - atr * 3.5
        tp3 = entry - atr * 5

    # Grade based on score
    if best_score >= 80:
        grade = "A+"
    elif best_score >= 70:
        grade = "A"
    elif best_score >= 60:
        grade = "B+"
    elif best_score >= 50:
        grade = "B"
    else:
        grade = "C"

    # Leverage suggestion based on ATR%
    atr_percent = atr / entry * 100
    if atr_percent < 0.3:
        leverage_rec = "100x – 200x"
    elif atr_percent < 0.5:
        leverage_rec = "50x – 100x"
    else:
        leverage_rec = "20x – 50x"

    # Session boost
    session_boost = {
        "US": 1.2,
        "London": 1.1,
        "Asia": 0.9
    }.get(session, 1.0)
    # Not directly used, but could adjust confidence

    # Build reasoning paragraph
    reasoning = f"""
**{pair} – {direction} (Grade: {grade})**  
**Session:** {session} (Pakistan time)  
**Trend:** {'Uptrend' if htf_trend_up else 'Downtrend'} on higher timeframe.  
**Key Levels:** {'Support at ' + f'{support:.2f}' if support else ''} {'Resistance at ' + f'{resistance:.2f}' if resistance else ''}.  
**Entry Signal:** {', '.join(best_reasons)}.  
**Trade Plan:** Enter within {entry_zone_low:.2f}–{entry_zone_high:.2f}. Stop at {sl:.2f}. Targets: {tp1:.2f} (TP1), {tp2:.2f} (TP2), {tp3:.2f} (TP3).  
**Volume:** {'Surge detected' if vol_ok else 'Normal'}.  
**Recommended Leverage:** {leverage_rec} (ATR = {atr_percent:.2f}%).  
"""

    return {
        'pair': pair,
        'rank': top_rank,
        'direction': direction,
        'grade': grade,
        'score': best_score,
        'entry_zone_low': entry_zone_low,
        'entry_zone_high': entry_zone_high,
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2,
        'tp3': tp3,
        'leverage_rec': leverage_rec,
        'reasoning': reasoning,
        'session': session,
        'timestamp': datetime.now()
    }

# -------------------- Main UI --------------------
st.sidebar.header("Settings")

# Pair source
pair_limit = st.sidebar.slider("Number of top pairs to scan", 10, 100, 50, step=10)

# Timeframes
tf_entry = st.sidebar.selectbox("Entry Timeframe", ["15m"], index=0, disabled=True)
tf_trend = st.sidebar.selectbox("Trend Timeframe", ["1h", "4h"], index=0)

# Minimum grade to display
min_grade = st.sidebar.selectbox("Minimum Grade to Show", ["All", "C", "B", "B+", "A", "A+"], index=0)

# Manual refresh
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Display current Pakistan time and session
now_pk = get_pakistan_time()
session = get_session(now_pk)
st.sidebar.info(f"🇵🇰 Pakistan Time: {now_pk.strftime('%H:%M')}\nSession: {session}")

# Get top pairs
top_pairs = get_top_pairs_cached()[:pair_limit]
st.sidebar.write(f"Scanning top {len(top_pairs)} pairs by volume")

# Scan button
if st.button("🚀 Generate Signals"):
    signals = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, pair in enumerate(top_pairs):
        status_text.text(f"Analyzing {pair} ({i+1}/{len(top_pairs)})")
        df_entry = fetch_ohlcv(pair, tf_entry, 300)
        df_trend = fetch_ohlcv(pair, tf_trend, 200)
        if df_entry is not None and df_trend is not None:
            df_entry = add_indicators(df_entry)
            df_trend = add_indicators(df_trend)
            signal = generate_signal(pair, df_entry, df_trend, session, i+1)
            if signal:
                signals.append(signal)
        progress_bar.progress((i + 1) / len(top_pairs))
        time.sleep(0.3)

    status_text.text("Analysis complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

    # Filter by grade
    grade_order = {"A+": 5, "A": 4, "B+": 3, "B": 2, "C": 1}
    if min_grade != "All":
        signals = [s for s in signals if grade_order.get(s['grade'], 0) >= grade_order[min_grade]]

    # Store in session state
    st.session_state.signals = signals

# Display signals if present
if 'signals' in st.session_state and st.session_state.signals:
    st.subheader(f"📡 Trading Signals – {len(st.session_state.signals)} found")
    for sig in st.session_state.signals:
        with st.expander(f"Rank #{sig['rank']} – {sig['pair']} {sig['direction']} (Grade: {sig['grade']})", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entry Zone", f"{sig['entry_zone_low']:.2f} – {sig['entry_zone_high']:.2f}")
                st.metric("Stop Loss", f"{sig['sl']:.2f}")
            with col2:
                st.metric("TP1", f"{sig['tp1']:.2f}")
                st.metric("TP2", f"{sig['tp2']:.2f}")
                st.metric("TP3", f"{sig['tp3']:.2f}")
            with col3:
                st.metric("Score", f"{sig['score']}")
                st.metric("Leverage Rec", sig['leverage_rec'])
                st.metric("Session", sig['session'])
            st.markdown(sig['reasoning'])
            st.caption(f"Generated: {sig['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    # Download CSV
    df_out = pd.DataFrame([{
        'Rank': s['rank'],
        'Pair': s['pair'],
        'Direction': s['direction'],
        'Grade': s['grade'],
        'Entry_Zone': f"{s['entry_zone_low']:.2f}-{s['entry_zone_high']:.2f}",
        'SL': s['sl'],
        'TP1': s['tp1'],
        'TP2': s['tp2'],
        'TP3': s['tp3'],
        'Leverage_Rec': s['leverage_rec'],
        'Session': s['session'],
        'Reasoning': s['reasoning']
    } for s in st.session_state.signals])
    csv = df_out.to_csv(index=False)
    st.download_button("📥 Download Signals CSV", csv, "signals.csv", mime="text/csv")
else:
    if 'signals' in st.session_state:
        st.info("No signals match the current criteria. Try adjusting the grade filter or scanning again.")
    else:
        st.info("👆 Click 'Generate Signals' to start analysis.")

st.caption("Data source: Binance public API. Sessions based on Pakistan time (UTC+5). For educational purposes only.")
