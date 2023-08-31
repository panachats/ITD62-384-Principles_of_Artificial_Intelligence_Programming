import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("D:\\walailak\\Year_2_Semester_3\\pythonProject\\IMI_Dataset.csv")
group1 =df['Cholestero'][df['Sex'] == 'F']
