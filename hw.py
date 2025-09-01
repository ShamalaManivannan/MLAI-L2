import numpy as np
import pandas as pd

data = pd.read_csv("iris.csv")

print(data.head())
print(data.info())

data["species"] = data["species"].map({'setosa' : 0,'versicolor' : 1, 'virginica' : 2})
print(data.info())

x = data[["sepal_length","sepal_width","petal_length","petal_width"]]
y = data["species"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state = 5)


#Multi-Variable Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg=reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)


from sklearn.metrics import mean_squared_error

rmse_le = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE of Multi-Variable Regression =", rmse_le)


#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 2)

x_train_poly = poly_features.fit_transform(x_train)
x_test_poly = poly_features.fit_transform(x_test)

poly_reg = LinearRegression()
poly_reg.fit(x_train_poly,y_train)

y_pred_poly = poly_reg.predict(x_test_poly)

rmse_poly = np.sqrt(mean_squared_error(y_test,y_pred_poly))
print("RMSE of Polynomial Regression =", rmse_poly)             