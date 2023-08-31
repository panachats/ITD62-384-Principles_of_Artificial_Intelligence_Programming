import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# 1.Load the dataset
df = pd.read_csv("ปณชัช เอี่ยมน้ํา - example_dataset.csv")
# print(df.to_string())


# 2. Finding the missimg values in each column
print(df.isnull().sum(), '\n')
# print(df.dtypes)
# print(df.info())
# print(df.columns)

#Replacing the missing values
df2 = df.fillna(df.mean())
consuming_mode = df['Gender'].mode()[0]
df2['Gender'].fillna(consuming_mode, inplace=True)
# print(df2)
# print("Data:\n", df2.to_string())

# 3. Normalizing the dataset
for index, col in df2.iteritems():
    if col.dtypes == 'object':  # ตรวจสอบว่าเป็น string หรือไม่
        continue  # ข้ามไป
    r = col.max() - col.min()
    print(f"R of {index}: {r}")



MinMaxScaler = MinMaxScaler()
StandardScaler = StandardScaler()
RobustScaler = RobustScaler()

# [row, column] = : ในส่วนของ rows เพื่อระบุว่าเราจะแสดงทุกแถว
# df2["InTemp_MinMax"] = MinMaxScaler.fit_transform(df2["IndoorTemperature"].to_numpy().reshape(-1,1))
# df2['BMI_MinMax'] = MinMaxScaler.fit_transform(df2["BMI"].to_numpy().reshape(-1,1))
# df2['Age_Standard'] = StandardScaler.fit_transform(df2["Age"].to_numpy().reshape(-1, 1))
# df2['MeanHR_Standard'] = StandardScaler.fit_transform(df2["MeanHR"].to_numpy().reshape(-1,1))

# print(df2.loc[:, ["BMI","BMI_MinMax"]].describe(),'\n######################################')
# print(df2.loc[:, ["Age","Age_Standard"]].describe(),'\n######################################')
# print(df2.loc[:, ["IndoorTemperature","InTemp_MinMax"]].describe(),'\n######################################')
# print(df2.loc[:, ["MeanHR","MeanHR_Standard"]].describe(),'\n######################################')

# Standard Scaler เหมาะกับข้อมูลที่มีการกระจายแบบ Normal Distribution ข้อมูลไม่มีค่า outlier
# MinMax เหมาะข้อมูลที่มี outlier แต่ไม่เหมาะกับ Normal Distribution
# Robust ปรับปรุงการกระจายของข้อมูล เพื่อลดผลกระทบจาก outliers ที่อาจมีอยู่ในข้อมูล


df2["InHumid_MinMax"] = MinMaxScaler.fit_transform(df2["IndoorHumidity"].to_numpy().reshape(-1, 1))
df2['MeanEDA_MinMax'] = MinMaxScaler.fit_transform(df2['MeanEDA'].to_numpy().reshape(-1,1))

df2['PM10_Robust'] = RobustScaler.fit_transform(df2["PM10"].to_numpy().reshape(-1,1))
df2['PM25_Robust'] = RobustScaler.fit_transform(df2["PM25"].to_numpy().reshape(-1,1))
df2['MeanAir_Robust'] = RobustScaler.fit_transform(df2["MeanAir"].to_numpy().reshape(-1,1))

print(df2.loc[:, ["IndoorHumidity","InHumid_MinMax"]].describe(),'\n######################################')
print(df2.loc[:, ["PM10","PM10_Robust"]].describe(),'\n######################################')
print(df2.loc[:, ["PM25","PM25_Robust"]].describe(),'\n######################################')
print(df2.loc[:, ["MeanAir","MeanAir_Robust"]].describe(),'\n######################################')
print(df2.loc[:, ["MeanAir","MeanEDA_MinMax"]].describe(),'\n######################################')


print(df2.describe().to_string())
# # 3.1 Calculating the 1st Quartile and the 3rd Quartile of the 'InTemp' column
Q1 = df['InTemp'].quantile(0.25)
Q3 = df['InTemp'].quantile(0.75)
print("The 1st and 2nd quartile value is {0:1f} and {1:1f} respectively".format(Q1,Q3))
# # 3.2 Calculating the Inter-quartile range
IQR = Q3 - Q1
print("The value of Inter Quartile Range is: ", IQR)
# # 3.3 Finding the Lower Fence and the Upper Fence
Lower_fence = Q1 - (1.5 * IQR)
Upper_fence = Q3 + (1.5 * IQR)
print("Lower Fence value: ", Lower_fence)
print("The upper Fence value: ", Upper_fence)



# 4. Feature Engineering
df3 = pd.DataFrame(df2['MeanHR'])
df3.loc[df3['MeanHR'].between(50,60,'both'),'binned_label'] = 'Slow_HR'
df3.loc[df3['MeanHR'].between(60,100,'right'),'binned_label'] = 'Normal_HR'
df3.loc[df3['MeanHR'].between(100,150,'right'),'binned_label'] = 'Fast_HR'
df3.loc[df3['MeanHR'] >150,'binned_label'] = 'Dangerously_Fast_HR'
print(df3.sample(n=5))

#5 Quartile of the 'PM2.5' and 'PM10' column
# การหาค่าที่ต่ำกว่าหรือเท่ากับ 25% ของข้อมูลในคอลัมน์
PM25_Q1 = df2['PM25'].quantile(0.25)
PM25_Q3 = df2['PM25'].quantile(0.75)
PM10_Q1 = df2['PM10'].quantile(0.25)
PM10_Q3 = df2['PM10'].quantile(0.75)

print(f"PM2.5 Q1 คือ: {PM25_Q1:.2f}")
print(f"PM2.5 Q3 คือ: {PM25_Q3:.2f}")
print(f"PM10 Q1 คือ: {PM10_Q1:.2f}")
print(f"PM10 Q3 คือ: {PM10_Q3:.2f}")



# sns.boxplot(x=df2['IndoorHumidity'])
# plt.title("IndoorHumidity before outlier elimination")

# sns.boxplot(x=df2['IndoorTemperature'])
# plt.title("IndoorTemperature before outlier elimination")
# sns.boxplot(x=df2['BMI'])
# plt.title("BMI before outlier elimination")
#
# sns.boxplot(x=df2['Age'])
# plt.title("Age before outlier elimination")
#
# sns.boxplot(x=df2['MeanHR'])
# plt.title("MeanHR before outlier elimination")
#
# sns.boxplot(x=df2['MeanSKT'])
# plt.title("MeanSKT before outlier elimination")
#
# sns.boxplot(x=df2['disease'])
# plt.title("disease before outlier elimination")
#
# sns.boxplot(x=df2['MeanAir'])
# plt.title("MeanAir before outlier elimination")
#
# sns.boxplot(x=df2['PM10'])
# plt.title("PM10 before outlier elimination")
#
# sns.boxplot(x=df2['PM25'])
# plt.title("PM25 before outlier elimination")
# plt.show()


