import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


advertising = pd.DataFrame(pd.read_csv("Data/simple_lr_application/advertising.csv"))


X = advertising['TV'].values
y = advertising['Sales'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))



def predict_out(val):
    val = np.array(int(val))
    req_inp = np.array(val.reshape(-1, 1))
    output = regressor.predict(req_inp)
    return output