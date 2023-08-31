import pandas as pd

mydataset = {
    'cars':["BMW","Volvo","Ford"],
    'passings':[3,7,2]
}

df = pd.DataFrame(mydataset)

print(df,'\n--------------')
print(df['cars'],'\n--------------')
print(df.loc[0],'\n--------------')
print(df.loc[[0,1]],'\n--------------')