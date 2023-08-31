import pandas as pd
import numpy as np


np.random.seed(5000)
data = np.random.randint(0,20,size=(5,4)) # 5 rows and 4 columns
df = pd.DataFrame(data, columns=list("ABCD"))
print(df)

dfDrop = df.drop(index=[4], columns=["C","D"],inplace=True)

data = np.random.randint(0,20,size=(4,1))
dfAdd = df
dfAdd["E"] = data
# get brif information
print(df.index)
print(df.columns)
