import pandas as pd
import numpy as np

np.random.seed(5000)
data = np.random.randint(0,20,size=(5,4)) # 5 rows and 4 columns
df = pd.DataFrame(data, columns=list("ABCD"))
print(df)

# get brif information
print(df.index)
print(df.columns)