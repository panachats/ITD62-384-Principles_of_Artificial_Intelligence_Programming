import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pit

# 1 Load the dataset
df = pd.read_csv("NewExEnv-missing-values.csv")

# 2 Show outlier using boxplow
sns.boxplot(x=df["InTemp"])
plt.title("inTemp before outlier elimination")
plt.show()

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
# # 3.4 Checking the data which have the 'InTemp' less than the Lower Fence
# # or greater than the Upper Fence. Basically we are retrieving the outliers here
Outliers = df[(df['InTemp'] < Lower_fence) | (df['InTemp'] > Upper_fence)]
print(Outliers.describe())

# # 3.5 Checking the data which have the 'InTemp' within the Lower fence
# # and Upper Fence. Here we are negating the outlier data and printing only
# the potential data which is within the Lower and Upper Fence
InTempData = df[~((df['InTemp'] < Lower_fence) | (df['InTemp'] > Upper_fence))]
print(InTempData.describe())
sns.boxplot(x=InTempData['InTemp'])
plt.title("InTemp after outlier elimination")
plt.show()