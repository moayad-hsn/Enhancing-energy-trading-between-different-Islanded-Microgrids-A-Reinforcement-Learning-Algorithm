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
arr = np.array([1,0,0,1])
a, b, c, d = arr 

a = np.ones((1,))
b = np.ones((0,))

a+=b
print(a, b)