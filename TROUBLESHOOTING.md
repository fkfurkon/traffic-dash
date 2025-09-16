# Troubleshooting Guide - Traffic Prediction Demo

## ปัญหาที่พบบ่อยและวิธีแก้ไข

### 1. Feature Names Mismatch Error

**ปัญหา:**
```
Error making prediction: feature_names mismatch: [...] expected [...] in input data
```

**สาเหตุ:** โมเดลที่ train มาใช้ One-Hot Encoding และมี features เพิ่มเติม

**วิธีแก้:**
- ใช้ฟังก์ชัน `prepare_model_input()` ที่จัดการ One-Hot Encoding อัตโนมัติ
- ป้อนข้อมูล month, day, minute ให้ครบถ้วน

### 2. DataFrame.dtypes Error

**ปัญหา:**
```
DataFrame.dtypes for data must be int, float, bool or category
```

**สาเหตุ:** XGBoost ไม่รองรับ string/object columns

**วิธีแก้:**
- แปลง day_of_week เป็น One-Hot Encoding
- ใช้ `pd.to_numeric()` สำหรับข้อมูลตัวเลข

### 3. Timestamp Type Error

**ปัญหา:**
```
TypeError: strptime() argument 1 must be str, not Timestamp
```

**สาเหตุ:** ฟังก์ชันพยายาม parse Pandas Timestamp เป็น string

**วิธีแก้:**
- ตรวจสอบประเภทของ timestamp object ก่อน
- ใช้ `isinstance()` เพื่อจัดการทั้ง string และ Timestamp

### 4. Arrow Serialization Warning

**ปัญหา:**
```
Serialization of dataframe to Arrow table was unsuccessful
```

**สาเหตุ:** DataFrame มี mixed data types

**วิธีแก้:**
- แปลงข้อมูลทั้งหมดเป็น string สำหรับการแสดงผล
- ใช้ `str()` wrapper สำหรับ display data

### 5. XGBoost Version Warning

**ปัญหา:**
```
WARNING: [...] please export the model by calling Booster.save_model
```

**สาเหตุ:** โมเดลถูก train ด้วย XGBoost version เก่า

**วิธีแก้:**
- Warning นี้ไม่ส่งผลต่อการทำงาน
- สามารถ ignore ได้หรือ retrain โมเดลด้วย version ใหม่

## การป้องกันปัญหา

### 1. Input Validation
```python
# ตรวจสอบประเภทข้อมูลก่อนส่งไปยังโมเดล
for col in input_data.columns:
    input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
```

### 2. Error Handling
```python
try:
    prediction = model.predict(input_data)[0]
except Exception as e:
    st.error(f"Error making prediction: {e}")
    return
```

### 3. Data Type Consistency
```python
# แปลงข้อมูลเป็น string สำหรับการแสดงผล
display_data = pd.DataFrame({
    'Feature': ['Vehicle Count', 'Lag 1', ...],
    'Value': [str(vehicle_count), str(lag_1), ...]
})
```

## การ Debug

### 1. ตรวจสอบ Features
```python
print("Expected features:", model.get_booster().feature_names)
print("Input features:", list(input_data.columns))
```

### 2. ตรวจสอบ Data Types
```python
print("Input data types:")
print(input_data.dtypes)
```

### 3. ตรวจสอบ Shape
```python
print("Input shape:", input_data.shape)
print("Expected shape: (1, 14)")  # 1 row, 14 features
```

## Best Practices

1. **ใช้ Type Hints**: ระบุประเภทข้อมูลในฟังก์ชัน
2. **Error Handling**: จัดการ exceptions ให้ครบถ้วน
3. **Data Validation**: ตรวจสอบข้อมูลก่อนการประมวลผล
4. **Logging**: บันทึก error logs สำหรับการ debug
5. **Testing**: ทดสอบด้วยข้อมูลหลากหลายประเภท

## การติดต่อสำหรับช่วยเหลือ

หากพบปัญหาที่ไม่สามารถแก้ไขได้:
1. ตรวจสอบ error message ใน terminal
2. ดู logs ใน Streamlit app
3. ตรวจสอบข้อมูล input ที่ส่งไปยังโมเดล
4. ลองใช้ sample data จาก dataset
