import pandas as pd
import numpy as np
# 1. Load the dataset
df = pd.read_csv("D:\\walailak\\Year_2_Semester_3\\Principles_of_Artificial_Intelligence_Programming\\NewExEnv-missing-values.csv")
# 2. Handling Categorical Data
# 2.2 One-hot encoding
from sklearn.preprocessing import OneHotEncoder
one_hot_encoding = OneHotEncoder(sparse=False)
one_hot_data = one_hot_encoding.fit_transform(df['Consuming'].to_numpy().reshape(-1, 1))
one_hot_df = pd.DataFrame(one_hot_data, columns=one_hot_encoding.get_feature_names_out())
print(one_hot_df)
df = df.drop('Consuming', axis=1)
df = df.join(one_hot_df)
print(df.head().to_string())