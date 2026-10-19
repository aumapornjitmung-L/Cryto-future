"""ดึงแท่งเทียน Binance USDT-M Futures ของทุกเหรียญแล้วบันทึกเป็น data/<SYMBOL>_<interval>.csv
รันบนเครื่องของคุณ (IP ที่ Binance ไม่บล็อก) แล้ว commit โฟลเดอร์ data/ ขึ้น GitHub
ใช้เป็นข้อมูลสำรองให้แอปบน Streamlit Cloud (ซึ่งอยู่ในสหรัฐฯ และอาจเรียก Binance ไม่ได้)

    python update_data.py                # ค่าเริ่มต้น: 4h และ 1d
    python update_data.py 1h 4h 1d
"""
import sys
from pathlib import Path

from crypto_ml import DEFAULT_SYMBOLS, INTERVAL_MIN, DataError, fetch_klines

intervals = sys.argv[1:] or ["4h", "1d"]
out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)

for sym in DEFAULT_SYMBOLS:
    for iv in intervals:
        if iv not in INTERVAL_MIN:
            print(f"ข้ามกรอบเวลาที่ไม่รองรับ: {iv}")
            continue
        try:
            df, src = fetch_klines(sym, iv, total=3000)  # Binance Futures เท่านั้น ไม่ใช้ snapshot/สำรอง
        except DataError as e:
            print(f"[ล้มเหลว] {sym} {iv}: {e}")
            continue
        df.to_csv(out / f"{sym}_{iv}.csv", index=False)
        print(f"[OK] {sym} {iv}: {len(df)} แท่ง ({df['time'].iloc[0]:%Y-%m-%d} → {df['time'].iloc[-1]:%Y-%m-%d})")
