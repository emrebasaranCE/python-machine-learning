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

country = corruptedData.iloc[:, 0:1].values
print(country)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

country[:, 0] = le.fit_transform(corruptedData.iloc[:, 0])
print(country)

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)

sonuc = pd.DataFrame(data=country, index=range(22), columns=['fr', 'tr', 'us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=age, index=range(22), columns=['boy', 'kilo', 'yas'])
print(sonuc2)

cinsiyet = corruptedData.iloc[:, -1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])
print(sonuc3)

s = pd.concat([sonuc, sonuc2], axis=1)
print(s)

s = pd.concat([s, sonuc3], axis=1)
print(s)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)