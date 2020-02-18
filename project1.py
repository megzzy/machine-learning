import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split

missing_values = ["n/a", "na", "--","NA",""]
df = pd.read_csv("H:\guc\machine learning\house_data_complete.csv", na_values = missing_values)
df


X = df.iloc[2,:].values
y = df.iloc[3,:].values
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

X_train
X_test
X_val
y_train
y_test
y_val

#def plotData(X, y):
 #   fig = pyplot.figure()
pyplot.plot(df['bedrooms'], df['price'], 'ro', ms=10, mec='k')
pyplot.ylabel('prices')
pyplot.xlabel('bedrooms')
    
#plotData(X, y)

#sn = df.iloc[:,2]
#sn
m = len(sn)
m
normalize = (df["price"] - df["price"].mean())/ (df["price"].std())
print(normalize)
#df[sn] = normalize

#print ((df[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]==df.isnull()).sum())
print ((df[:]==df.isnull()).sum())
print (df[:].isnull())

#i can fill the missing data or remove it
#to fill
replaceNA = df.fillna(0)
replaceNA

replaceMISS = df.fillna(method = 'bfill' , axis = 0).fillna(0)
replaceMISS
#to remove
df.dropna()
#def  featureNormalize(X):

    #X_norm = X.copy()
    #mu = np.zeros(X.shape[1])
    #sigma = np.zeros(X.shape[1])
    #mu = np.mean(X)
   # sigma = np.std(X)
  #  X_norm = (X - mu)/sigma

 #   return X_norm, mu, sigma

#X_norm, mu, sigma = featureNormalize(X)

#print('Computed mean:', mu)
#print('Computed standard deviation:', sigma)

#X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

#normalize = (df[2,:] - df[2,:].mean())/ (df[2,:].std())
#print(normalize)

#def gradientDescentMulti(X, y, theta, alpha, num_iters):
       
 #   m = y.shape[0]

  #  theta = theta.copy()
    
   # J_history = []
    
    #for i in range(num_iters):
     #   sumofh0x=np.dot(X,theta)
      #  theta=theta-((alpha/m)*(np.dot(X.T,sumofh0x-y)))
       # J_history.append(computeCostMulti(X, y, theta))
    
    #return theta, J_history

#alpha = 0.03
#iterations = 100
#theta = np.zeros(3)

#theta, J_history = gradientDescentMulti(X_train ,y_train, theta, alpha, iterations)

    