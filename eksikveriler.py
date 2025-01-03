import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

corruptedData = pd.read_csv('eksikveriler.csv')
print(corruptedData)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
age = corruptedData.iloc[:, 1:4].values

imputer = imputer.fit(age[:, 1:4])
age[:, 1:4] = imputer.transform(age[:, 1:4])
print(age)