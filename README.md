# 📈 พยากรณ์ Crypto ด้วย Machine Learning — Binance USDT-M Futures
เหรียญ: **QUSDT, BRUSDT, AGTUSDT, PHAUSDT** (เพิ่มเหรียญอื่นได้ในแอปผ่านช่อง "อื่น ๆ...")

ใช้ 2 โมเดลที่ดีที่สุดจากการเทียบแบบ walk-forward พยากรณ์ทิศทาง (ขึ้น/ลง) และราคาปิดล่วงหน้า 1–6 แท่ง

## ขั้นตอน
1. บนเครื่องของคุณ: `pip install -r requirements.txt` แล้ว `python update_data.py` → ได้ `data/*.csv` (ข้อมูลจาก Binance Futures)
2. เลือก 2 โมเดลที่ดีที่สุด: รัน `benchmark_colab.py` (ใน Colab หรือในเครื่อง) → ได้ `model_config.json`
   (บน Colab ให้อัปโหลด `crypto_ml.py` และโฟลเดอร์ `data/` ก่อน)
3. ทดสอบ: `streamlit run app.py`
4. push ขึ้น GitHub (รวม `data/` และ `model_config.json`) แล้ว Deploy ที่ https://share.streamlit.io (Main file: `app.py`)

## แหล่งข้อมูลที่แอปใช้ (ตามลำดับ)
1. Binance USDT-M Futures API
2. Bybit Linear (เฉพาะเมื่อติ๊กตัวเลือกในแอป ราคาอาจต่างเล็กน้อย)
3. ไฟล์ `data/<SYMBOL>_<interval>.csv` ใน repo
4. อัปโหลด CSV ในแอป

เซิร์ฟเวอร์ Streamlit Cloud อยู่ในสหรัฐฯ ซึ่ง Binance อาจบล็อก (HTTP 451) จึงควรมีไฟล์ใน `data/` ไว้เสมอ
และรัน `update_data.py` ซ้ำแล้ว commit เป็นระยะเพื่อให้ข้อมูลไม่เก่า (ดูเวลาแท่งล่าสุดและแหล่งข้อมูลบนหน้าแอป)

> ไม่ใช่คำแนะนำการลงทุน
