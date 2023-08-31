import pandas as pd
df1 = pd.DataFrame({'Courses': ["Spark", "PySpark", "Python", "pandas"],
                    'Fee': [20000, 25000, 22000, 24000],
                    'Name': ["John", "James", "Peter", "David"]},
                    index=['r1', 'r2', 'r3', 'r4'])

df2 = pd.DataFrame({'Duration': ['30day', '40days', '35days', '60days', '55days'],
                    'Discount': [1000, 2300, 2500, 2000, 3000],
                    'Name': ["John", "Billy", "Marcus", "James", "David"]},
                    index=['r1', 'r2', 'r3', 'r5', 'r6'])

# Merge two DataFrames by column using pandas.merge()
result = pd.merge(df1, df2, on="Name")

