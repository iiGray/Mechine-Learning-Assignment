"""
合并两个数据集
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Step 1: 加载电力数据")
print("="*60)

# 读取电力数据
power = pd.read_csv('household_power_consumption.txt', sep=';',
                     na_values=['?'], low_memory=False)

# 解析日期
power['Date'] = pd.to_datetime(power['Date'], format='%d/%m/%Y')
print(f"原始形状: {power.shape}")
print(f"日期范围: {power['Date'].min()} ~ {power['Date'].max()}")

# 按天汇总
# global_active_power, global_reactive_power, sub_metering_1, sub_metering_2: 日求和
# voltage, global_intensity: 日平均
daily = power.groupby('Date').agg({
    'Global_active_power': 'sum',      # 日总有用功 (kW)
    'Global_reactive_power': 'sum',    # 日总无用功 (kW)
    'Voltage': 'mean',                 # 日平均电压 (V)
    'Global_intensity': 'mean',        # 日平均电流 (A)
    'Sub_metering_1': 'sum',           # 厨房 (Wh)
    'Sub_metering_2': 'sum',           # 洗衣房 (Wh)
    'Sub_metering_3': 'sum',           # 气候控制 (Wh)
}).reset_index()

# 计算 sub_metering_remainder
# sub_metering_remainder = (global_active_power * 1000 / 60) - (sub_1 + sub_2 + sub_3)
# 注意：global_active_power 单位是 kW，乘以1000/60转换为 Wh/min
daily['Sub_metering_remainder'] = (
    daily['Global_active_power'] * 1000 / 60 -
    (daily['Sub_metering_1'] + daily['Sub_metering_2'] + daily['Sub_metering_3'])
)

print(f"按天汇总后形状: {daily.shape}")
print(f"缺失值统计:\n{daily.isnull().sum()}")
print(f"日总数: {len(daily)}")

print("\n" + "="*60)
print("Step 2: 加载并处理天气数据")
print("="*60)

weather = pd.read_csv('weather_200612_201011.csv')
print(f"原始形状: {weather.shape}")

# 转换 AAAAMM 为日期 (取每月1号)
weather['Date'] = pd.to_datetime(weather['AAAAMM'].astype(str) + '01', format='%Y%m%d')
print(f"日期范围: {weather['Date'].min()} ~ {weather['Date'].max()}")

# 按月份聚合所有站点（取均值）
monthly_weather = weather.groupby('Date').agg({
    'RR': 'mean',           # 月降水均值 (mm/10, 需/10转mm)
    'NBJRR1': 'mean',       # 降水>=1mm天数均值
    'NBJRR5': 'mean',       # 降水>=5mm天数均值
    'NBJRR10': 'mean',      # 降水>=10mm天数均值
    'NBJBROU': 'mean',      # 雾天天数均值 (缺失多, NaN会变成均值)
}).reset_index()

# RR 转换为 mm
monthly_weather['RR_mm'] = monthly_weather['RR'] / 10

print(f"月度天气形状: {monthly_weather.shape}")
print(f"缺失值统计:\n{monthly_weather.isnull().sum()}")

print("\n" + "="*60)
print("Step 3: 合并电力数据与天气数据")
print("="*60)

# 添加月份列用于合并
daily['YearMonth'] = daily['Date'].dt.to_period('M')
monthly_weather['YearMonth'] = monthly_weather['Date'].dt.to_period('M')

# 合并
merged = daily.merge(
    monthly_weather[['YearMonth', 'RR_mm', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']],
    on='YearMonth', how='left'
)
merged = merged.drop(columns=['YearMonth'])

print(f"合并后形状: {merged.shape}")
print(f"缺失值统计:\n{merged.isnull().sum()}")

print("\n" + "="*60)
print("Step 4: 处理缺失值")
print("="*60)

# 线性插值处理电力缺失值
power_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
              'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
              'Sub_metering_3', 'Sub_metering_remainder']
for col in power_cols:
    merged[col] = merged[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# 天气缺失值用均值填充（仅NBJBROU可能有缺失）
weather_cols = ['RR_mm', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
for col in weather_cols:
    merged[col] = merged[col].fillna(merged[col].mean())

print(f"处理后缺失值:\n{merged.isnull().sum().sum()}")

print("\n" + "="*60)
print("Step 5: 保存预处理数据")
print("="*60)

merged.to_csv('preprocessed_data.csv', index=False)
print(f"已保存 preprocessed_data.csv, 形状: {merged.shape}")
print(f"列: {merged.columns.tolist()}")
print(f"\n数据预览:")
print(merged.head(10).to_string())
print(f"\n数据尾行:")
print(merged.tail(5).to_string())
