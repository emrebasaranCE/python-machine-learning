import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datas = pd.read_csv('data.csv')

print(datas)

height = datas['boy']
print(height)

heightAndWeight = datas[['boy', 'kilo']]
print(heightAndWeight)