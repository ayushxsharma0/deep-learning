import numpy as np
from sklearn.datasets import load_iris


data = load_iris()

X = data.data
y = data.target

X = X[y < 2]
y = y[y < 2]

shp = X.shape
#Perceptron algorithm

epochs = 100
#initial random weights
coeff = np.random.randn(shp[1] +1)

lr = 0.1

for _ in range(epochs):

    
    for idx in range(len(X)):

        x = np.array([1, *X[idx]])
        target = y[idx]

        z = coeff @ x

        #apply step function
        pred = 1 if z>=0 else 0

        if pred != target:
            #we need to update the weights

            coeff += lr*(target -pred)*x


print(coeff)

#check the accuracy
cnt = 0

for i in range(len(X)):

    x = np.array([1, *X[i]])

    z = coeff @ x

    pred = 0 if z<0 else 1

    if pred == y[i]:
        cnt +=1


print(f"Accuracy -> {cnt/len(X)}")