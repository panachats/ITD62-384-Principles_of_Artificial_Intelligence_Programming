import pandas as pd

# creating an empty DataFrame object and then add columns and rows
df = pd.DataFrame()
df["Name"] = ["Takashi", "Jason", "Nakamura", "Olivia"]
df["Grade"] = [100,90, 70, 80]

# creating from a dictionary
dictionary = {"Name":["Takashi", "Jason", "Nakamura", "Olivia"], "Grade":[100, 90, 70, 80]}
df2 = pd.DataFrame(dictionary)

# creating from a list, in which we then specify the column names
a = [90,100,95]
b = [81, 89, 85]
c = [75, 70, 77]
df3 = pd.DataFrame([a,b,c], columns=list("ABC"))


# crating from csv
df4 = pd.read_csv("data.csv")
# print(df1,"\n----------------------")
# print(df2,"\n----------------------")

# print(df4.info())
# print(df4.describe(include=object))

# print(df4)

