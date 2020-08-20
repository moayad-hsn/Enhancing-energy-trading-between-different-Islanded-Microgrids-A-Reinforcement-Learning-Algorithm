import pandas as pd
import numpy as np
'''
df1 = pd.read_csv("data/wind/Hamza_Elsheikh_wind_generation.csv")
df2 =  pd.read_csv("data/Solar/Hamza_Elsheikh_solar_generation.csv")

p=[]

for i in range(len(df1)):
	if df1["timestamp"][i] != df2["TimeSeries"][i]:
		p.append(i)
		p.append(df1["timestamp"][i])
		break




print(p)

print (df1.head())

print (df2.head())
print (len(df1), len(df2))

'''
"""import gym
from gym import spaces


action_space = spaces.Box(low=np.array([0.0,0.0,0.0,10.0]), high=np.array([3.0,2.0,350.0,19.0]), dtype = np.float32)

print(action_space.sample())
"""
b = [10,20,30]
a = {b: 10}
a[10,20, 30] = 50
a[10,20,21] +=1
print(a)