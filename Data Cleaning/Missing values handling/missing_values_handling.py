import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# 1. Load the dataset
df = pd.read_csv("D:\\งาน มหาวิทยาลัยวลัยลักษณ์\\ปี 2 เทอม 3\\Principles_of_Artificial_Intelligence_Programming\\NewExEnv-missing-values.csv")
print(df.to_string())

# 2. Check the datatype for each column
print(df.dtypes)

# 3. Data Summary
print(df.describe(include='object').to_string())

# 4. Finding the missing values
# 4.1 Check for missing data in any of the variables.
print(df.info())

# 4.2 Finding the missing values in each column
print(df.isnull().sum(), '\n')

datadrop = df.dropna() # ลบ NaN ออก
print(datadrop.describe(include='object').to_string(), '\n')

inTemp_mean = df.InTemp.mean()
df.InTemp.fillna(inTemp_mean, inplace=True)

# for categorical data (mode)
consuming_mode = df['Consuming'].mode()[0]
df['Consuming'].fillna(consuming_mode, inplace=True)
print(df.isnull().sum(), '\n')

columnsNum = df.iloc[:, 1:16].columns
columnsCat = df.iloc[:, 16:21].columns

imputerMean = SimpleImputer(missing_values=np.nan, strategy='mean')
dataimputerMean = pd.DataFrame(imputerMean.fit_transform(df.iloc[:, 1:16]))
dataimputerMean.columns = columnsNum
print(dataimputerMean.describe().to_string(), '\n')

imputerMode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
dataimputerMode = pd.DataFrame(imputerMode.fit_transform(df.iloc[:, 16:21]))
dataimputerMode.columns = columnsCat
print(dataimputerMode.describe(include='object'), '\n')



imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit_transform(df.iloc[:, 1:16])
df.iloc[:, 1:16] = np.round(imp.transform(df.iloc[:, 1:16]))
print(df.iloc[:, 1:16].describe().to_string(), '\n')
# In case they are duplicates data, we
df.drop_duplicates()