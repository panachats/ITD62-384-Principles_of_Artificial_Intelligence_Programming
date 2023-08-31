import pandas as pd
# 1. Load the dataset
df = pd.read_csv("D:\\walailak\\Year_2_Semester_3\\pythonProject\\IMI_Dataset.csv")
# 2. Handling Categorical Data
print(df.info())
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in df.columns:
    if df[i].dtype == "object" and i != "Timestamp":
        df[i] = label_encoder.fit_transform(df[i])
print(df.info())