import pandas as pd
import numpy as np
# 1. Load the dataset
df = pd.read_csv("D:\\walailak\\Year_2_Semester_3\\Principles_of_Artificial_Intelligence_Programming\\NewExEnv-missing-values.csv")
# 2. Handling Categorical Data
# 2.1 Label encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df["Consuming"] = label_encoder.fit_transform(df["Consuming"])
 # Check label mapping (normally with alphabetical order)
mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(mapping)
print(df.head().to_string())
# df.to_csv('D:/Test.csv')