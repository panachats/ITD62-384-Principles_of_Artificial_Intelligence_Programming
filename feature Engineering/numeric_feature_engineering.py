import pandas as pd
# 1. Load the dataset
df = pd.read_csv("D:\\walailak\\Year_2_Semester_3\\pythonProject\\IMI_Dataset.csv")
# 2. Numeric Feature Engineering
# # 2.1 Binning Data
df2 = pd.DataFrame(df["Age"])
# # Method 1: .cut(), use cut when you need to segment and sort data values
# into bins.
# Baby = 0-2, Toddler = 2-4, Children = 5-12, Teen = 13-17, Adult = 18- 59
# Older adults > 60
bins = [0, 2, 4, 12, 17, 59, 102]
df2['AgeBin'] = pd.cut(df['Age'], bins)
labels = ["Baby", "Toddler", "Children", "Teen", "Adult", "Older adults"]
df2["AgeLabel"] = pd.cut(df2["Age"], bins=bins, labels=labels)
print(df2.sample(n=5),'\n#################################')

# # Method 2: .between() and .loc(), returns a boolean vector containing True
# wherever the corresponding Series element is between the boundary values
# left and right
df3 =  pd.DataFrame(df['Age'])
df3.loc[df3['Age'].between(0, 2, 'both'), 'binned_label'] = "Baby"
df3.loc[df3['Age'].between(2, 4, 'right'), 'binned_label'] = "Toddler"
df3.loc[df3['Age'].between(4, 12, 'right'), 'binned_label'] = "Children"
df3.loc[df3['Age'].between(12, 17, 'right'), 'binned_label'] = "Teen"
df3.loc[df3['Age'].between(17, 59, 'right'), 'binned_label'] = "Adult"
df3.loc[df3['Age'].between(59, 102, 'right'), 'binned_label'] = "Older adults"
print(df3.sample(n=5),'\n#################################')
# # Method 3: .qcut(), Quantile-based discretization function.
df4 = pd.DataFrame(df['Age'])
df4['binned_label'], cut_bin = pd.qcut(df['Age'], q=6,
labels=["Baby", "Toddler", "Children", "Teen", "Adult", "Older adults"], retbins=True)
print("cut_bin: ", cut_bin)
print(df4.sample(n=5),'\n#################################')