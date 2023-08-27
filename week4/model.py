#importing required modules and libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#loading the dataset
tvmarketing = pd.read_csv("tvmarketing.csv")

#preapring for modelling 
X = tvmarketing["TV"]
y = tvmarketing["Sales"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)


lm = LinearRegression()
lm.fit(X_train,y_train)

pickle.dump(lm,open("model.pickle", "wb"))