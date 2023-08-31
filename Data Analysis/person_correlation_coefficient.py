import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Load the dataset
df = pd.read_csv("D:\\walailak\\Year_2_Semester_3\\pythonProject\\file\\ปณชัช เอี่ยมน้ํา - IMI_Dataset.csv")
# 2. Handling Categorical Data

label_encoder = LabelEncoder()
for i in df.columns:
    if df[i].dtype == "object" and i != "Timestamp":
        df[i] = label_encoder.fit_transform(df[i])
        # Correlatio


df2 = df[(df["FastingBS"] == 1) & (df["ExerciseAngina"] == 1) & (df["HeartDisease"] == 1)]
df3 = df[(df["FastingBS"] == 0) & (df["ExerciseAngina"] == 1) & (df["HeartDisease"] == 1)]
cor = df3[["Age","Cholesterol","MaxHR","RestingBP"]].corr()



# we could use a heatmap; a data visualization technique where each
# value is represented by a color, according to its intensity in each scale.
# sns.heatmap(cor, vmin=-1, vmax=1, annot=True)
# sns.heatmap(cor2, vmin=-1, vmax=1, annot=True)

group1 = df2['Age'][df2['Sex'] == 1][df2['HeartDisease']==1]
group2 = df2['Age'][df2['Sex'] == 0][df2['HeartDisease']==1]

Fmean, Mmean = group1.mean(), group2.mean()
ttest = stats.ttest_ind(group1, group2)
sns.distplot(group1, color="skyblue", label="M")
sns.distplot(group2, color="orchid", label="F")
plt.text(x=410, y=0.0075, s="Female's mean: %.2f" % Fmean, color='skyblue')
plt.text(x=410, y=0.007, s="Male's mean: %.2f" % Mmean, color='orchid')
plt.title("T-Test Cholesterol by Gender (t = %.2f, p = %.2f)" % (ttest))
plt.legend()
print(ttest)
plt.show()





