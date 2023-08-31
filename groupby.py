import pandas as pd
import numpy as np

technologies = {
    'Coures':["Spark","PySpark","Hadoop","Python","Pandas"],
    'Fee':[22000,25000,23000,24000,26000],
    'Discount':[1000,2300,1000,1200,2500],
    'Duration':['35dats','35days','40days','30days','25days']
}

df = pd.DataFrame(technologies)
print(df,'\n---------------------------------')
# iloc คือ หา location ที่เป็น int
df1 = df.iloc[:2,:] #  หมายถึงการเลือกแถวตั้งแต่ index 0 จนถึง index 1
df2 = df.iloc[2:,:] # หมายถึงการเลือกแถวตั้งแต่ index 2 ไปจนถึงตัวสุดท้าย
print(df1,'\n---------------------------------')
print(df2,'\n---------------------------------')

#groupby
grouped = df.groupby(df.Duration)
df5 = grouped.get_group("35days")
print(df5)