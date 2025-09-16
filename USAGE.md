# วิธีการใช้งาน Traffic Prediction Demo

## การเริ่มต้นใช้งาน

### 1. การเตรียมสภาพแวดล้อม

```bash
# สร้าง virtual environment (ถ้ายังไม่มี)
python -m venv .venv

# เปิดใช้งาน virtual environment
source .venv/bin/activate  # สำหรับ Linux/Mac
# หรือ .venv\Scripts\activate  # สำหรับ Windows

# ติดตั้ง dependencies
pip install -r requirements.txt
```

### 2. การรันแอปพลิเคชัน

```bash
streamlit run app.py
```

## การใช้งานแต่ละหน้า

### 🔮 หน้า Prediction

#### ขั้นตอนการทำนาย:
1. ป้อนข้อมูลในฟอร์ม:
   - **Current Vehicle Count**: จำนวนยานพาหนะปัจจุบัน (0-100)
   - **Lag 1-3**: ข้อมูลยอดรถ 5, 10, 15 นาทีที่แล้ว
   - **Day of Week**: เลือกวันในสัปดาห์
   - **Hour**: เลือกชั่วโมง (0-23)

2. กดปุ่ม "🚀 Predict Traffic"

3. ดูผลการทำนาย:
   - ค่าทำนายแสดงเป็นตัวเลข
   - กราฟ Gauge แสดงระดับการจราจร
   - เปรียบเทียบกับค่าปัจจุบัน

#### ตัวอย่างข้อมูลป้อนเข้า:
```
Current Vehicle Count: 15
Lag 1: 12
Lag 2: 8
Lag 3: 10
Day of Week: Tuesday
Hour: 14
```

### 📊 หน้า Data Analysis

#### ฟีเจอร์ที่มี:
1. **Dataset Overview**:
   - จำนวนรุกคอร์ดทั้งหมด
   - ช่วงวันที่ของข้อมูล
   - ค่าเฉลี่ยและค่าสูงสุดของยานพาหนะ

2. **Traffic Patterns Over Time**:
   - กราฟเส้นแสดงแนวโน้มการจราจร

3. **Hourly และ Daily Patterns**:
   - กราฟแท่งแสดงเฉลี่ยตามชั่วโมง
   - กราฟแท่งแสดงเฉลี่ยตามวัน

4. **Correlation Matrix**:
   - ความสัมพันธ์ระหว่าง features ต่างๆ

### 🤖 หน้า Model Info

#### ข้อมูลที่แสดง:
1. **Model Details**:
   - ประเภทโมเดล: XGBoost Regressor
   - วัตถุประสงค์: ทำนายจำนวนรถในชั่วโมงถัดไป
   - รายการ Input Features

2. **Dataset Information**:
   - จำนวนรุกคอร์ดและ features
   - ช่วงวันที่ของข้อมูล

3. **Feature Statistics**:
   - สถิติเชิงพรรณนาของแต่ละ feature

4. **Sample Data**:
   - ข้อมูลตัวอย่าง 10 แถวแรก

## เคล็ดลับการใช้งาน

### การป้อนข้อมูลที่ดี:
1. **Lag Values**: ควรป้อนค่าที่สมเหตุสมผล (ไม่กระโดดมากเกินไป)
2. **Vehicle Count**: ค่าปัจจุบันควรสอดคล้องกับ lag values
3. **Time Context**: พิจารณาเวลาและวันให้เหมาะสม (เช่น ชั่วโมงเร่งด่วน)

### การตีความผลลัพธ์:
1. **Delta Value**: บอกความแตกต่างจากค่าปัจจุบัน
2. **Gauge Chart**: 
   - เขียว (0-10): การจราจรน้อย
   - เหลือง (10-25): การจราจรปานกลาง
   - แดง (25-50): การจราจรหนาแน่น

### การแก้ไขปัญหาเบื้องต้น:

#### ถ้าแอปไม่เปิด:
```bash
# ตรวจสอบว่าไฟล์ครบถ้วน
ls -la
# ควรมี: app.py, xgb_model.pkl, traffic_dataset1.csv, requirements.txt

# ตรวจสอบ dependencies
pip list | grep streamlit
```

#### ถ้า prediction ให้ผลแปลกๆ:
1. ตรวจสอบค่า input ว่าอยู่ในช่วงที่เหมาะสม
2. ลองดูข้อมูลใน Data Analysis เพื่อเข้าใจรูปแบบข้อมูล

## การปรับแต่งเพิ่มเติม

### เพิ่ม Features ใหม่:
1. แก้ไขในไฟล์ `app.py` ที่ฟังก์ชัน `prediction_page()`
2. เพิ่ม input widgets ใหม่
3. อัพเดทการเตรียมข้อมูลสำหรับโมเดล

### เปลี่ยนธีม:
```python
# ในไฟล์ app.py
st.set_page_config(
    page_title="Traffic Prediction Demo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### เพิ่มกราฟใหม่:
ใช้ Plotly Express หรือ Plotly Graph Objects ในหน้า Data Analysis
