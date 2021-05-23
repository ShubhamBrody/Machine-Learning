#importing libraries------------------------ALWAYS USED
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset--------------------------ALWAYS USED
dataset = pd.read_csv("Salary_Data.csv");
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values




#Splitting Dataset--------------------------ALWAYS USED
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
print()
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)




#Implementing SLR for training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)




#SLR predicting values from our test arrays
Y_pred = regressor.predict(X_test)

print(regressor.predict([[0]]))




#Visualising using plotting graphs using MATPLOTLIB
plt.scatter(X_train, Y_train, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, Y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


