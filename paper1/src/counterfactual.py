import pandas as pd

df = pd.read_excel("../data/RailD.xlsx")

print(df.shape)
print(df.columns)
df.head()