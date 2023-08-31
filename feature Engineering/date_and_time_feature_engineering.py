import pandas as pd
# 1. Load the dataset
df = pd.read_csv("D:\\walailak\\Year_2_Semester_3\\pythonProject\\IMI_Dataset.csv")
# 2. Date and Time Feature Engineering
# 2.1 parse Timestamp into DateTime format
df['Timestamp'] = pd.to_datetime(df.Timestamp)
df2 = pd.DataFrame(df['Timestamp'])
# 2.2 extract date/quarter/week/weekday/time
df2["date"] = df2['Timestamp'].dt.date
df2['quarter'] = df2['Timestamp'].dt.quarter
df2['week'] = df2['Timestamp'].dt.isocalendar().week
df2['weekday'] = df2['Timestamp'].dt.weekday
df2['time'] = df2['Timestamp'].dt.time

# 2.3 Binning time into 4 bins, [0–5], [6–11], [12–17] and [18–23].
df2['time_session'] = pd.to_datetime(df2['Timestamp'], format='%H:%M:%S')
a = df2.assign(time_session=pd.cut(df2['time_session'].dt.hour, [0, 6, 12, 18, 24],
labels=['Night', 'Morning', 'Afternoon', 'Evening']))
df2['time_session'] = a['time_session']

print(df2)