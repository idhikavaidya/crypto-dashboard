"""Crypto dashboard – Streamlit rebuild of the 36104 Session 3 Part B app.

Six panels (one large, five small) of Binance USDT pairs. Each panel cycles
through time windows automatically, and the coin in the large panel rotates,
matching the timings in the original config.json. Prices refresh every few
seconds by polling Binance's public REST API (no API key needed).
"""
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------- config ---
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
COLORS = {
    "BTCUSDT": "#F7931A", "ETHUSDT": "#627EEA", "BNBUSDT": "#F3BA2F",
    "XRPUSDT": "#23292F", "ADAUSDT": "#0033AD", "DOGEUSDT": "#C3A634",
}
# interval -> (window length in minutes, label, cache seconds)
FREQUENCIES = {
    "1s": (30, "Thirty Minutes", 3),
    "1m": (60 * 3, "Three Hours", 5),
    "5m": (60 * 24 * 3, "Three Days", 15),
    "30m": (60 * 24 * 7 * 3, "Three Weeks", 60),
    "4h": (60 * 24 * 31 * 3, "Three Months", 120),
    "1d": (60 * 24 * 365 * 3, "Three Years", 300),
}
ORDER = list(FREQUENCIES)
CHART_ROTATION = (30, 40)   # seconds between large-panel coin changes
TIME_SWITCH = (10, 15)      # seconds between time-window changes per panel
REFRESH_SECONDS = 5

# Binance blocks some regions (incl. the US, where Streamlit Cloud runs), so
# try the global API, then Binance's public market-data mirror, then Binance.US.
API_BASES = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api.binance.us",
]
MOCK = os.environ.get("CRYPTO_MOCK") == "1"

st.set_page_config(page_title="Crypto", page_icon="📈", layout="wide")


# ------------------------------------------------------------------ data ---
_GOOD_BASE = None

def _get(path, params):
    """GET from the first Binance base that answers; remember which one worked."""
    global _GOOD_BASE
    bases = API_BASES
    if _GOOD_BASE:
        bases = [_GOOD_BASE] + [b for b in API_BASES if b != _GOOD_BASE]
    last_err = None
    for base in bases:
        try:
            r = requests.get(base + path, params=params, timeout=8)
            if r.status_code == 200:
                _GOOD_BASE = base
                return r.json(), base
            last_err = f"{base}: HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = f"{base}: {e.__class__.__name__}"
    raise RuntimeError(last_err)


def _mock_klines(symbol, interval):
    minutes, _, _ = FREQUENCIES[interval]
    step = {"1s": 1, "1m": 60, "5m": 300, "30m": 1800, "4h": 14400, "1d": 86400}[interval]
    n = min(int(minutes * 60 / step), 1500)
    rng = random.Random(hash((symbol, interval)) & 0xFFFF)
    p = {"BTCUSDT": 70000, "ETHUSDT": 3500, "BNBUSDT": 600, "XRPUSDT": 0.6,
         "ADAUSDT": 0.45, "DOGEUSDT": 0.15}[symbol]
    now = int(time.time() * 1000)
    rows = []
    for i in range(n):
        o = p
        p *= 1 + rng.gauss(0, 0.002)
        rows.append({"time": now - (n - i) * step * 1000, "open": o,
                     "high": max(o, p) * 1.001, "low": min(o, p) * 0.999,
                     "price": p, "volume": rng.random()})
    return pd.DataFrame(rows), "mock"


def _fetch_klines(symbol, interval):
    if MOCK:
        return _mock_klines(symbol, interval)
    minutes = FREQUENCIES[interval][0]
    start = int((time.time() - minutes * 60) * 1000)
    rows, base = [], None
    while True:  # Binance returns at most 1000 records per request
        batch, base = _get("/api/v3/klines", {
            "symbol": symbol, "interval": interval, "startTime": start, "limit": 1000})
        rows += batch
        if len(batch) < 1000:
            break
        start = batch[-1][0] + 1
    df = pd.DataFrame([{
        "time": e[6],                 # interval closing time (ms)
        "open": float(e[1]), "high": float(e[2]), "low": float(e[3]),
        "price": float(e[4]), "volume": float(e[5]),
    } for e in rows])
    return df, base


# Small shared cache: fast intervals refresh often, slow ones rarely.
_CACHE, _LOCK = {}, threading.Lock()


def get_klines(symbol, interval):
    key, ttl = (symbol, interval), FREQUENCIES[interval][2]
    with _LOCK:
        hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        value = _fetch_klines(symbol, interval)
    except Exception:
        if hit:               # keep showing the last good data on a blip
            return hit[1]
        raise
    with _LOCK:
        _CACHE[key] = (time.time(), value)
    return value


def prefetch(pairs):
    """Download all panels' series in parallel so a refresh finishes quickly."""
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda p: _safe(get_klines, *p), pairs))


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 – the panel shows its own error
        return None


@st.cache_data(ttl=10, show_spinner=False)
def get_tickers():
    if MOCK:
        return {s: (get_klines(s, "1m")[0]["price"].iloc[-1], random.uniform(-5, 5)) for s in SYMBOLS}
    data, _ = _get("/api/v3/ticker/24hr", {"symbols": str(SYMBOLS).replace("'", '"').replace(" ", "")})
    return {d["symbol"]: (float(d["lastPrice"]), float(d["priceChangePercent"])) for d in data}


# ------------------------------------------------------------- formatting ---
def fmt_price(v):
    if v >= 1000:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:,.3f}"
    return f"{v:,.5f}"


def tick_format(interval):
    return {"1s": "%H:%M", "1m": "%H:%M", "5m": "%d %b %H:%M", "30m": "%d %b",
            "4h": "%d %b", "1d": "%b %Y"}[interval]


# ---------------------------------------------------------------- charts ---
def chart(symbol, interval, large):
    df, _ = get_klines(symbol, interval)
    df = df.sort_values("time")
    t = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert(TZ)
    color = COLORS[symbol]
    fig = go.Figure()
    for col in ("high", "low"):
        fig.add_trace(go.Scatter(x=t, y=df[col], mode="lines", name=col.title(),
                                 line=dict(color=color, width=1), opacity=0.25,
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=t, y=df["price"], mode="lines", name="Close",
        line=dict(color=color, width=1.5 if large else 1),
        customdata=df[["open", "high", "low"]].values,
        hovertemplate="%{x|%d %b %Y %H:%M:%S}<br>Close: %{y:,.6g} USDT"
                      "<br>Open: %{customdata[0]:,.6g}<br>High: %{customdata[1]:,.6g}"
                      "<br>Low: %{customdata[2]:,.6g}<extra></extra>"))
    fig.update_layout(
        height=515 if large else 210, margin=dict(l=4, r=4, t=4, b=4),
        showlegend=False, hovermode="closest",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickformat=tick_format(interval), nticks=6 if large else 4,
                   showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(side="right", gridcolor="#eee", tickfont=dict(size=10),
                   tickformat=",.5~g" if df["price"].iloc[-1] < 1 else ",.6~g"),
    )
    return fig, df["price"].iloc[-1]


def panel(slot, symbol, large):
    state = st.session_state.panels[slot]
    with st.container(border=True):
        try:
            fig, last = chart(symbol, state["interval"], large)
        except Exception as e:  # noqa: BLE001
            st.markdown(f"**{symbol}**")
            st.caption(f"Loading… ({e})")
            return
        label = FREQUENCIES[state["interval"]][1]
        st.markdown(
            f"<div class='ph'><span class='sym' style='color:{COLORS[symbol]}'>{symbol}</span>"
            f"<span class='win'>{label}</span><span class='px'>{fmt_price(last)} USDT</span></div>",
            unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch", key=f"c{slot}",
                        config={"displayModeBar": False})


def ticker_strip(tickers, reverse=False):
    items = []
    syms = SYMBOLS[::-1] if reverse else SYMBOLS
    for s in syms:
        if s not in tickers:
            continue
        p, ch = tickers[s]
        cls = "up" if ch >= 0 else "down"
        items.append(f"<span class='ti'><b style='color:{COLORS[s]}'>{s}</b> {fmt_price(p)} "
                     f"<span class='{cls}'>{ch:+.2f}%</span></span>")
    row = "".join(items) * 4
    direction = "reverse" if reverse else "normal"
    st.markdown(f"<div class='tape'><div class='track' style='animation-direction:{direction}'>"
                f"{row}</div></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ page ---
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1600px;}
.ph {display:flex; gap:.6rem; align-items:baseline; font-size:.9rem; margin-bottom:-.4rem;}
.ph .sym {font-weight:700;} .ph .win {color:#666;} .ph .px {margin-left:auto; font-variant-numeric:tabular-nums;}
.tape {overflow:hidden; white-space:nowrap; background:#f3f4f6; color:#222; border-radius:6px;
       padding:.35rem 0; font-size:.85rem; margin:.25rem 0 .6rem;}
.track {display:inline-block; animation: scroll 60s linear infinite;}
.ti {margin: 0 1.6rem;} .up {color:#15803d;} .down {color:#b91c1c;}
@keyframes scroll {from {transform: translateX(0);} to {transform: translateX(-50%);}}
</style>""", unsafe_allow_html=True)

TZ = st.context.timezone or "UTC"

if "panels" not in st.session_state:
    now = time.time()
    st.session_state.panels = [
        {"interval": random.choice(ORDER), "next": now + random.uniform(*TIME_SWITCH)}
        for _ in range(6)]
    st.session_state.large = 0
    st.session_state.next_rotation = now + random.uniform(*CHART_ROTATION)

st.title("Crypto")
st.caption("Binance public market data · prices in USDT · each panel has its own price scale")


@st.fragment(run_every=REFRESH_SECONDS)
def dashboard():
    now = time.time()
    # time-window timers
    for p in st.session_state.panels:
        if now >= p["next"]:
            p["interval"] = ORDER[(ORDER.index(p["interval"]) + 1) % len(ORDER)]
            p["next"] = now + random.uniform(*TIME_SWITCH)
    # large-panel coin rotation
    if now >= st.session_state.next_rotation:
        st.session_state.large = (st.session_state.large + 1) % len(SYMBOLS)
        st.session_state.next_rotation = now + random.uniform(*CHART_ROTATION)

    large = SYMBOLS[st.session_state.large]
    others = [s for s in SYMBOLS if s != large]
    prefetch([(s, p["interval"]) for s, p in zip([large] + others, st.session_state.panels)])

    try:
        tickers = get_tickers()
    except Exception:  # noqa: BLE001
        tickers = {}
    ticker_strip(tickers)

    left, right = st.columns([2, 1])
    with left:
        panel(0, large, True)
    with right:
        panel(1, others[0], False)
        panel(2, others[1], False)
    c = st.columns(3)
    for i, s in enumerate(others[2:]):
        with c[i]:
            panel(3 + i, s, False)

    ticker_strip(tickers, reverse=True)
    st.caption(f"Updated {pd.Timestamp.now(tz=TZ).strftime('%H:%M:%S')} · "
               f"refreshes every {REFRESH_SECONDS}s · source: {_GOOD_BASE or 'n/a'}")


dashboard()
