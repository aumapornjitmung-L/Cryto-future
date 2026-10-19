"""Streamlit: พยากรณ์ทิศทาง/ราคา crypto (QUSDT, BRUSDT, AGTUSDT) ด้วย 2 โมเดลที่ดีที่สุด"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crypto_ml import (DEFAULT_SYMBOLS, INTERVAL_MIN, MODEL_LABELS, SYMBOL_INFO, DataError,
                       fetch_klines, forecast, load_csv)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
FALLBACK_MODELS = ["random_forest", "hist_gb"]

st.set_page_config(page_title="พยากรณ์ Crypto ด้วย ML", page_icon="📈", layout="wide")


def load_config() -> tuple[list[str], bool]:
    p = ROOT / "model_config.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))["models"][:2], True
        except Exception:  # noqa: BLE001
            pass
    return FALLBACK_MODELS, False


MODELS, FROM_BENCH = load_config()


@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch(symbol, interval, total, allow_fallback):
    return fetch_klines(symbol, interval, total, DATA_DIR, allow_fallback)


@st.cache_data(ttl=900, show_spinner=False)
def cached_forecast(df, models, horizon):
    return forecast(df, list(models), horizon)


def fmt_price(x: float) -> str:
    return f"{x:,.6f}".rstrip("0").rstrip(".") if abs(x) < 1000 else f"{x:,.2f}"


def signal(p: float) -> tuple[str, str]:
    if p >= 0.55:
        return "แนวโน้มขึ้น 🟢", "#2e9e5b"
    if p <= 0.45:
        return "แนวโน้มลง 🔴", "#d64545"
    return "ไม่ชัดเจน ⚪", "#888888"


def card(title: str, prob: float, ret: float, price: float) -> str:
    label, color = signal(prob)
    return (f"<div style='border:1px solid {color};border-left:8px solid {color};border-radius:8px;padding:12px 16px;'>"
            f"<div style='font-size:.85rem;opacity:.75'>{title}</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{color}'>{label}</div>"
            f"<div style='font-size:.9rem'>โอกาสขึ้น {prob*100:.1f}%<br>"
            f"คาดการณ์เปลี่ยนแปลง {ret*100:+.2f}% → {fmt_price(price)}</div></div>")


# ------------------------------------------------------------ Sidebar
st.title("📈 พยากรณ์ Crypto ด้วย Machine Learning — Binance Futures")
with st.sidebar:
    st.header("ตั้งค่า")
    choice = st.selectbox("เหรียญ (Binance USDT-M Futures)", DEFAULT_SYMBOLS + ["อื่น ๆ..."])
    symbol = (st.text_input("ระบุสัญลักษณ์", "BTCUSDT").upper().strip()
              if choice == "อื่น ๆ..." else choice)
    if symbol in SYMBOL_INFO:
        st.caption(SYMBOL_INFO[symbol])
    interval = st.selectbox("กรอบเวลา", list(INTERVAL_MIN), index=2)
    horizon = st.slider("พยากรณ์ล่วงหน้า (จำนวนแท่ง)", 1, 6, 1)
    total = st.slider("จำนวนแท่งที่ใช้ฝึก", 500, 3000, 1500, step=250)
    mode = st.radio("แหล่งข้อมูล", ["ดึงอัตโนมัติ (API)", "อัปโหลด CSV"])
    allow_fb = st.checkbox("ใช้ Bybit สำรองถ้า Binance ถูกบล็อก", value=False,
                           help="ราคาอาจต่างจาก Binance เล็กน้อย ถ้าไม่ติ๊กและดึงไม่ได้ แอปจะใช้ไฟล์ snapshot ใน data/") if mode == "ดึงอัตโนมัติ (API)" else False
    upload = st.file_uploader("CSV (time,open,high,low,close,volume)", type="csv") if mode == "อัปโหลด CSV" else None
    st.divider()
    st.caption("โมเดลที่ใช้: " + " + ".join(MODEL_LABELS[m] for m in MODELS))
    if not FROM_BENCH:
        st.warning("ยังไม่พบ `model_config.json` จึงใช้โมเดลเริ่มต้น — รัน `benchmark_colab.py` เพื่อเลือก 2 โมเดลที่ดีที่สุดจากข้อมูลจริง")

# ------------------------------------------------------------ Data
try:
    if mode == "อัปโหลด CSV":
        if upload is None:
            st.info("อัปโหลดไฟล์ CSV ที่แถบด้านซ้าย")
            st.stop()
        df, source = load_csv(upload), f"CSV ที่อัปโหลด ({upload.name})"
    else:
        with st.spinner(f"กำลังดึงข้อมูล {symbol} {interval}..."):
            df, source = cached_fetch(symbol, interval, total, allow_fb)
    with st.spinner("กำลังฝึกโมเดลและประเมินแบบ walk-forward (ครั้งแรกอาจใช้เวลา 10–40 วินาที)..."):
        result = cached_forecast(df, tuple(MODELS), horizon)
except DataError as e:
    st.error(f"ใช้งาน {symbol} ไม่ได้: {e}")
    st.info("เป็นไปได้ว่าสัญลักษณ์ไม่มีอยู่จริง, API ถูกบล็อกตามประเทศของเซิร์ฟเวอร์ (เช่น HTTP 451), "
            "หรือข้อมูลน้อยเกินไป ทางแก้: อัปโหลด CSV ที่แถบซ้าย หรือวางไฟล์ `data/" f"{symbol}_{interval}.csv` ลงใน repo")
    st.stop()

last = df.iloc[-1]
delta = pd.Timedelta(minutes=INTERVAL_MIN[interval])
target_time = last["time"] + delta * horizon

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"ราคาปิดล่าสุด {symbol}", fmt_price(last["close"]))
m2.metric("เวลาแท่งล่าสุด (UTC)", f"{last['time']:%d/%m/%Y %H:%M}")
m3.metric("จำนวนแท่ง", f"{len(df):,}")
m4.metric("แหล่งข้อมูล", source)

tab_fc, tab_perf, tab_data, tab_about = st.tabs(["ผลพยากรณ์", "ประสิทธิภาพโมเดล", "ข้อมูล", "ข้อจำกัด"])

with tab_fc:
    st.subheader(f"พยากรณ์ปิดของแท่งที่ +{horizon} ({interval})")
    cols = st.columns(3)
    avg_p = sum(r["prob_up"] for r in result.values()) / len(result)
    avg_r = sum(r["pred_return"] for r in result.values()) / len(result)
    for col, (name, r) in zip(cols, result.items()):
        col.markdown(card(MODEL_LABELS[name], r["prob_up"], r["pred_return"], r["pred_price"]),
                     unsafe_allow_html=True)
    cols[2].markdown(card("ค่าเฉลี่ย 2 โมเดล", avg_p, avg_r, float(last["close"]) * float(np.exp(avg_r))),
                     unsafe_allow_html=True)
    ups = [r["prob_up"] > 0.5 for r in result.values()]
    (st.success if all(ups) or not any(ups) else st.warning)(
        "ทั้ง 2 โมเดลเห็นทิศทางตรงกัน" if all(ups) or not any(ups) else "2 โมเดลเห็นทิศทางต่างกัน — สัญญาณไม่ชัดเจน")

    view = df.tail(150)
    fig = go.Figure(go.Candlestick(x=view["time"], open=view["open"], high=view["high"],
                                   low=view["low"], close=view["close"], name="ราคา"))
    colors = ["#4c78a8", "#f58518"]
    for color, (name, r) in zip(colors, result.items()):
        fig.add_trace(go.Scatter(x=[last["time"], target_time], y=[last["close"], r["pred_price"]],
                                 mode="lines+markers", name=MODEL_LABELS[name],
                                 line=dict(color=color, dash="dash")))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(t=20),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

with tab_perf:
    st.markdown("ประเมินแบบ **walk-forward** (แบ่งตามเวลา 4 ช่วง เว้นช่วงกันข้อมูลรั่ว) — "
                "คอลัมน์ Baseline คือผลของการเดาแบบไม่ใช้โมเดล")
    rows = []
    for name, r in result.items():
        m = r["metrics"]
        rows.append({"โมเดล": MODEL_LABELS[name], "AUC (ทิศทาง)": m["auc"], "Accuracy": m["acc"],
                     "Baseline Acc": m["acc_baseline"], "Hit-rate (ถดถอย)": m["reg_hit"],
                     "MAE เทียบเดา 'ไม่เปลี่ยน'": m["mae_ratio"]})
    st.dataframe(pd.DataFrame(rows).set_index("โมเดล").round(4), width="stretch")
    st.caption("AUC ≈ 0.50 = ไม่ต่างจากเดาสุ่ม • MAE เทียบเดา < 1 = ดีกว่าการเดาว่าราคาไม่เปลี่ยน "
               "ราคา crypto รายแท่งมีสัญญาณรบกวนสูง ค่าที่เห็นมักใกล้ 0.50 ซึ่งเป็นเรื่องปกติ")

with tab_data:
    st.dataframe(df.tail(200).iloc[::-1], width="stretch", hide_index=True)

with tab_about:
    st.markdown("""
- ผลลัพธ์เป็นการประมาณทางสถิติ **ไม่ใช่คำแนะนำการลงทุน** และไม่รับประกันผลตอบแทน
- ตลาด crypto ผันผวนสูง โดยเฉพาะเหรียญสภาพคล่องต่ำ และสัญญา Perpetual ที่ใช้เลเวอเรจอาจขาดทุนเกินเงินต้น
- โมเดลใช้เฉพาะข้อมูลราคาและปริมาณในอดีต ไม่รู้ข่าว เหตุการณ์ในโครงการ หรือการปรับเงื่อนไขสัญญาของ Exchange
- เหรียญที่เพิ่งเปิดเทรดมีข้อมูลน้อย ผลประเมินจึงไม่เสถียร
- การเลือก 2 โมเดลที่ดีที่สุดจากข้อมูลจำนวนจำกัดอาจเป็นผลของความบังเอิญ ควรรัน benchmark ซ้ำเป็นระยะ
""")
