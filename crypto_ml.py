"""crypto_ml.py — ดึงข้อมูล OHLCV, สร้างฟีเจอร์, เทียบโมเดล (walk-forward) และพยากรณ์"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # xgboost ไม่ได้ติดตั้ง -> ข้ามโมเดลนี้
    XGBClassifier = XGBRegressor = None

DEFAULT_SYMBOLS = ["QUSDT", "BRUSDT", "AGTUSDT", "PHAUSDT"]  # Binance USDT-M Futures (Perpetual)
SYMBOL_INFO = {
    "QUSDT": "Quack AI (Q) — เปิดเทรดบน Binance Futures 2 ก.ย. 2025",
    "BRUSDT": "เปิดเทรดบน Binance Futures 21 มี.ค. 2025",
    "AGTUSDT": "Alaya AI (AGT) — Binance Futures",
    "PHAUSDT": "Phala Network (PHA) — เปิดเทรดบน Binance Futures 30 ธ.ค. 2024",
}
INTERVAL_MIN = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
BYBIT_IV = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
HEADERS = {"User-Agent": "Mozilla/5.0"}
MIN_ROWS = 200

MODEL_LABELS = {
    "linear": "Logistic/Ridge (เส้นฐาน)",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gb": "HistGradientBoosting",
    "xgboost": "XGBoost",
}


class DataError(Exception):
    pass


# ============================================================ ดึงข้อมูล
def _to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(pd.to_numeric(df["time"]), unit="ms")  # UTC (tz-naive)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _binance_like(base: str, path: str, symbol: str, interval: str, total: int) -> pd.DataFrame:
    rows, end = [], None
    while len(rows) < total:
        params = {"symbol": symbol, "interval": interval, "limit": min(1000, total - len(rows))}
        if end:
            params["endTime"] = end
        r = requests.get(base + path, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            raise DataError(f"HTTP {r.status_code}: {r.text[:120]}")
        chunk = r.json()
        if not isinstance(chunk, list) or not chunk:
            break
        rows = chunk + rows
        end = chunk[0][0] - 1
        if len(chunk) < params["limit"]:
            break
    if not rows:
        raise DataError("ไม่มีข้อมูล")
    return _to_frame(rows)


def _bybit(symbol: str, interval: str, total: int) -> pd.DataFrame:
    rows, end = [], None
    while len(rows) < total:
        params = {"category": "linear", "symbol": symbol, "interval": BYBIT_IV[interval],
                  "limit": min(1000, total - len(rows))}
        if end:
            params["end"] = end
        r = requests.get("https://api.bybit.com/v5/market/kline", params=params,
                         headers=HEADERS, timeout=15)
        if r.status_code != 200:
            raise DataError(f"HTTP {r.status_code}")
        j = r.json()
        if j.get("retCode") != 0:
            raise DataError(str(j.get("retMsg")))
        chunk = j["result"]["list"][::-1]  # เก่า -> ใหม่
        if not chunk:
            break
        rows = chunk + rows
        end = int(chunk[0][0]) - 1
        if len(chunk) < params["limit"]:
            break
    if not rows:
        raise DataError("ไม่มีข้อมูล")
    return _to_frame(rows)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates("time").sort_values("time").dropna(subset=["close"])
    return df.reset_index(drop=True)


def load_csv(src) -> pd.DataFrame:
    """อ่าน CSV (path หรือไฟล์ที่อัปโหลด) ต้องมีคอลัมน์เวลา และ open/high/low/close/volume"""
    df = pd.read_csv(src)
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = next((c for c in ("time", "timestamp", "date", "datetime", "open_time") if c in df.columns), None)
    if tcol is None or "close" not in df.columns:
        raise DataError("CSV ต้องมีคอลัมน์เวลา (time/timestamp/date) และ close")
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t):
        unit = "ms" if t.iloc[0] > 1e11 else "s"
        df["time"] = pd.to_datetime(t, unit=unit)
    else:
        df["time"] = pd.to_datetime(t, utc=True).dt.tz_localize(None)
    for c in ("open", "high", "low", "volume"):
        if c not in df.columns:
            df[c] = df["close"] if c != "volume" else 0.0
    return _clean(df[["time", "open", "high", "low", "close", "volume"]])


def fetch_klines(symbol: str, interval: str = "4h", total: int = 1500,
                 snapshot_dir: str | Path | None = None,
                 allow_fallback: bool = False) -> tuple[pd.DataFrame, str]:
    """ดึงแท่งเทียนจาก Binance USDT-M Futures เป็นหลัก
    ถ้าไม่ได้ -> (ถ้า allow_fallback) Bybit Linear -> ไฟล์ snapshot data/SYMBOL_interval.csv
    คืนค่า (DataFrame, ชื่อแหล่งข้อมูล)"""
    symbol = symbol.upper().strip()
    sources = [("Binance USDT-M Futures",
                lambda: _binance_like("https://fapi.binance.com", "/fapi/v1/klines", symbol, interval, total))]
    if allow_fallback:  # ราคาอาจต่างจาก Binance เล็กน้อย จึงไม่เปิดเป็นค่าเริ่มต้น
        sources.append(("Bybit Linear (สำรอง)", lambda: _bybit(symbol, interval, total)))
    errors = []
    for name, fn in sources:
        try:
            df = _clean(fn())
            return df.iloc[:-1].reset_index(drop=True), name  # ตัดแท่งสุดท้ายที่ยังไม่ปิด
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {str(e)[:100]}")
    if snapshot_dir:
        p = Path(snapshot_dir) / f"{symbol}_{interval}.csv"
        if p.exists():
            return load_csv(p), f"ไฟล์ snapshot ({p.name})"
        errors.append(f"ไม่พบไฟล์ {p.name}")
    raise DataError(" | ".join(errors))


# ============================================================ ฟีเจอร์
def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return rsi.where(dn != 0, 100.0)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    lc = np.log(c)
    r = lc.diff()
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 5, 10):
        f[f"ret_{k}"] = lc.diff(k)
    for w in (5, 10, 20):
        f[f"vol_{w}"] = r.rolling(w).std()
    f["rsi14"] = _rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    for s in (10, 30):
        f[f"ema{s}_ratio"] = c / c.ewm(span=s, adjust=False).mean() - 1
    sd20 = c.rolling(20).std().replace(0, np.nan)
    f["bb_pos"] = (c - c.rolling(20).mean()) / (2 * sd20)
    f["range"] = (h - l) / c
    f["close_pos"] = (c - l) / (h - l).replace(0, np.nan)
    f["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
    t = df["time"]
    f["dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7)
    if t.diff().median() < pd.Timedelta(days=1):
        f["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    return f.replace([np.inf, -np.inf], np.nan)


def build_dataset(df: pd.DataFrame, horizon: int):
    X = make_features(df)
    y = np.log(df["close"].shift(-horizon) / df["close"])  # log-return ล่วงหน้า h แท่ง
    return X, y


# ============================================================ โมเดล
def available_models() -> list[str]:
    names = ["linear", "random_forest", "extra_trees", "hist_gb"]
    if XGBClassifier is not None:
        names.append("xgboost")
    return names


def make_model(name: str, task: str, seed: int = 42):
    clf = task == "clf"
    if name == "linear":
        est = LogisticRegression(C=0.5, max_iter=2000) if clf else Ridge(alpha=10.0)
        return make_pipeline(StandardScaler(), est)
    if name == "random_forest":
        cls = RandomForestClassifier if clf else RandomForestRegressor
        return cls(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=seed)
    if name == "extra_trees":
        cls = ExtraTreesClassifier if clf else ExtraTreesRegressor
        return cls(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=seed)
    if name == "hist_gb":
        cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        return cls(max_iter=150, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=seed)
    if name == "xgboost" and XGBClassifier is not None:
        kw = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, random_state=seed, n_jobs=-1)
        return XGBClassifier(eval_metric="logloss", **kw) if clf else XGBRegressor(**kw)
    raise ValueError(f"ไม่รู้จักโมเดล: {name}")


def _prob_up(model, X) -> np.ndarray:
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    p = model.predict_proba(X)
    return p[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def walk_forward(X: pd.DataFrame, y: pd.Series, name: str, horizon: int,
                 n_splits: int = 4, with_reg: bool = True) -> dict:
    """ประเมินแบบ walk-forward (ไม่สุ่ม เว้นช่วง gap เท่า horizon เพื่อกันข้อมูลอนาคตรั่ว)"""
    yd = (y > 0).astype(int)
    acc, auc, base, hit, mae_r = [], [], [], [], []
    for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=horizon).split(X):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], yd.iloc[tr], yd.iloc[te]
        p = _prob_up(make_model(name, "clf").fit(Xtr, ytr), Xte)
        acc.append(accuracy_score(yte, p > 0.5))
        auc.append(roc_auc_score(yte, p) if yte.nunique() == 2 else np.nan)
        base.append(float((yte == int(ytr.mean() >= 0.5)).mean()))  # เดาคลาสส่วนใหญ่ของชุดฝึก
        if with_reg:
            pr = make_model(name, "reg").fit(Xtr, y.iloc[tr]).predict(Xte)
            yt = y.iloc[te].values
            hit.append(float((np.sign(pr) == np.sign(yt)).mean()))
            mae_r.append(float(np.mean(np.abs(yt - pr)) / np.mean(np.abs(yt))))  # <1 = ดีกว่าเดา "ไม่เปลี่ยน"
    out = {"acc": np.nanmean(acc), "auc": np.nanmean(auc), "acc_baseline": np.mean(base)}
    if with_reg:
        out.update({"reg_hit": np.mean(hit), "mae_ratio": np.mean(mae_r)})
    return out


def _train_xy(df: pd.DataFrame, horizon: int):
    X, y = build_dataset(df, horizon)
    mask = X.notna().all(axis=1) & y.notna()
    if mask.sum() < MIN_ROWS:
        raise DataError(f"ข้อมูลไม่พอ ({int(mask.sum())} แถว ต้องการอย่างน้อย {MIN_ROWS}) — ลองช่วงเวลาที่เล็กลงหรือเพิ่มจำนวนแท่ง")
    return X, y, mask


def forecast(df: pd.DataFrame, model_names: list[str], horizon: int = 1, evaluate: bool = True) -> dict:
    X, y, mask = _train_xy(df, horizon)
    x_last = X.iloc[[-1]]
    if x_last.isna().any(axis=None):
        raise DataError("แท่งล่าสุดคำนวณฟีเจอร์ไม่ได้ (ข้อมูลสั้นเกินไป)")
    last_close = float(df["close"].iloc[-1])
    out = {}
    for name in model_names:
        clf = make_model(name, "clf").fit(X[mask], (y[mask] > 0).astype(int))
        reg = make_model(name, "reg").fit(X[mask], y[mask])
        ret = float(reg.predict(x_last)[0])
        out[name] = {
            "prob_up": float(_prob_up(clf, x_last)[0]),
            "pred_return": ret,
            "pred_price": last_close * float(np.exp(ret)),
            "metrics": walk_forward(X[mask], y[mask], name, horizon) if evaluate else None,
        }
    return out


def benchmark(symbols, interval="4h", horizon=1, total=1500, names=None, snapshot_dir="data",
              save_snapshots=True, allow_fallback=False):
    """เทียบโมเดลทุกตัวบนทุกเหรียญ คืน (ตารางละเอียด, ตารางสรุปเรียงตาม AUC เฉลี่ย)"""
    names = names or available_models()
    snapshot_dir = Path(snapshot_dir)
    rows = []
    for s in symbols:
        try:
            df, src = fetch_klines(s, interval, total, snapshot_dir, allow_fallback)
        except DataError as e:
            print(f"[ข้าม] {s}: {e}")
            continue
        if save_snapshots and not src.startswith("ไฟล์"):
            snapshot_dir.mkdir(exist_ok=True)
            df.to_csv(snapshot_dir / f"{s}_{interval}.csv", index=False)
        try:
            X, y, mask = _train_xy(df, horizon)
        except DataError as e:
            print(f"[ข้าม] {s}: {e}")
            continue
        print(f"{s}: {len(df)} แท่ง จาก {src}")
        for n in names:
            rows.append({"symbol": s, "model": n, **walk_forward(X[mask], y[mask], n, horizon)})
    if not rows:
        raise DataError("ไม่มีเหรียญใดดึงข้อมูลได้")
    res = pd.DataFrame(rows)
    summary = (res.groupby("model")[["auc", "acc", "acc_baseline", "reg_hit", "mae_ratio"]].mean()
               .sort_values(["auc", "acc"], ascending=False))
    return res, summary
