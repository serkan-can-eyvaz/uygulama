import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Kidem_ve_Maas_VeriSeti.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3,
random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
plt.scatter(X_train, y_train, color = 'red')
plt.title('Kıdeme Göre Maaş Tahmini Regresyon Modeli')
plt.xlabel('Kıdem')
plt.ylabel('Maaş')
plt.show()
plt.scatter(X_train, y_train, color = 'red')
modelin_tahmin_ettigi_y = regressor.predict(X_train)
plt.scatter(X_train, modelin_tahmin_ettigi_y, color = 'blue')
plt.title('Kıdeme Göre Maaş Tahmini Regresyon Modeli')
plt.xlabel('Kıdem')
plt.ylabel('Maaş')
plt.show()
plt.scatter(X_train, y_train, color = 'red')
modelin_tahmin_ettigi_y = regressor.predict(X_train)
plt.plot(X_train, modelin_tahmin_ettigi_y, color = 'blue')
plt.title('Kıdeme Göre Maaş Tahmini Regresyon Modeli')
plt.xlabel('Kıdem')
plt.ylabel('Maaş')
plt.show()
print(3)
