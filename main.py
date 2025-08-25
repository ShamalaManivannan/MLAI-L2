#Multiple Variable Regression 
import numpy as np
import pandas as pd

data = pd.read_csv("titanic.csv")

print(data.head())
print(data.info())

data["Sex"] = data["Sex"].map({'male' : 0,'female' : 1})

print(data.info())

X = data[["Age","Pclass","Sex"]]
Y = data["Survived"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2, random_state = 5)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg=reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

from sklearn.metrics import mean_squared_error,accuracy_score

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
