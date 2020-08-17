import pandas as pd
import numpy as np

df = pd.read_csv("windgen450.csv")

timeStamp = df["timeStamp"][0:400]
p =[]
for i in timeStamp:
	idx = df[df["time"] == i].index.values
	k = df["p:1"][idx].values
	p.append(k)
array = np.array(p)
dff = pd.DataFrame(array, index =df["timeStamp"][0:400], columns =["P"])
print(dff.head())
dff.to_csv("windgen450fixed.csv")