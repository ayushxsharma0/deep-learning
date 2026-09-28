from sklearn.datasets import load_iris
import numpy as np


dataset = load_iris()

X = dataset.data
y = dataset.target

#prune for our perceptron
X = X[y<2]
y = y[y<2]

rows = X.shape[0]
features = X.shape[1]

X = np.c_[X, np.ones(rows)]
y = np.where(y==0,-1,1)

#coeffients of our perceptron (wi, b)

coeff = np.random.randn(1 + features)

#now we need to apply perceptron algorithm 
lr = 0.1
for _ in range(100):
    total = 0
    gradient = np.zeros(1 + features)

    for i in range(rows):

        #calculate f(xi)

        fx = coeff @ X[i]

        fx *= y[i]

        loss = max(0,-fx)

        total += loss

        if fx<0:
            gradient += -y[i] * X[i]

    #get the average loss
    total /= rows

    #now we need to apply gradient descent
    gradient /= rows

    coeff -= lr* gradient 

print(coeff)

#Calculate the accuracy
correct = 0

for i in range(rows):

    pred = coeff @ X[i]

    pred = 1 if pred>=0 else -1

    if pred == y[i]:
        correct +=1 

print(f"Accuracy -> {(correct/rows)*100}")