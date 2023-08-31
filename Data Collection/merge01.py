import pandas as pd
left = pd.DataFrame({"A": ["A0", "A1", "A2"], "B": ["B0", "B1", "B2"]}, index=["K0", "K1", "K2"])
right = pd.DataFrame({"C": ["C0", "C2", "C3"], "D": ["D0", "D2", "D3"]}, index=["K0", "K2", "K3"])
result1 = pd.concat([left, right])
# axis=0 to concat along rows, axis=1 to concat along columns.
result2 = pd.concat([left, right], axis=0)
result3 = pd.concat([left, right], axis=1)
print(result1,'\n------------------------')
print(result2,'\n------------------------')
print(result3,'\n------------------------')


