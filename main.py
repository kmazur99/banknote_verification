import pandas as pd
import numpy as np
import time

#shape[0] : number of rows
#shape[1] : number of columns
def sigmoid(X, weight):
    z = np.dot(X, weight)
   # print("Z dim =", z.shape[0])
   # print("z =", z)
    return 1 / (1 + np.exp(-z))

#print(1/(1+np.exp(-0.17)))

#print(1+np.exp(1))
#print(1 + np.exp([1]))
#print(1 + np.exp([1,2]))
print(np.dot([[1,2,3],[4,5,6]],[1,2,3]))

def gradient_descent(X, h, y):
#X.T transpose of X
#y.shape[0] is sample size (m in the learning material), we divide by to find avg (batch mode)
    return np.dot(X.T, (h - y)) / y.shape[0]

def update_weight_loss(weight, learning_rate, gradient):
    return weight - learning_rate * gradient

data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("rows = {}".format(data.shape[0]))
print("columns = {}".format(data.shape[1]))
pd.DataFrame(data.dtypes).rename(columns = {0:"dtype"})

df = data.copy()

df['class'] = df['Churn'].apply(lambda x : 1 if x == "Yes" else 0)
#df.shape[1]
X = df[['tenure','MonthlyCharges']].copy()
y = df['class'].copy()

intercept = np.ones((X.shape[0], 1))
X = np.concatenate((intercept, X), axis=1)
#print(X.shape[0],X.shape[1])

#alpha: learning rate
alpha=0.001

start_time = time.time()

num_iter = 20000
# inputs
theta = np.zeros(X.shape[1])
for i in range(num_iter):
    #print("X=", X)
    #print("X.T=", X.T)
    #print("Theta=", theta)
    h = sigmoid(X, theta)
    #print("h=", h)
    #print("y=", y)
    gradient = gradient_descent(X, h, y)
    #print("gradient =", gradient)
    theta = update_weight_loss(theta, alpha, gradient)

print("Training time (Log Reg using Gradient descent):" + str(time.time() - start_time) + " seconds")
print("Learning rate: {}\nIteration: {}".format(alpha, num_iter))

result = sigmoid(X, theta)

f = pd.DataFrame(np.around(result, decimals=6)).join(y)
f['pred'] = f[0].apply(lambda x : 0 if x < 0.5 else 1)
print("Accuracy (Loss minimization):")
print(f.loc[f['pred']==f['class']].shape[0] / f.shape[0] * 100)

#For Confusion Matrix
YActual = f['class'].tolist()
YPredicted =  f['pred'].tolist()

#print(YActual)
#print(YPredicted)

TP = 0
TN = 0
FP = 0
FN = 0

for l1,l2 in zip(YActual, YPredicted):
    if (l1 == 1 and  l2 == 1):
        TP = TP + 1
    elif (l1 == 0 and l2 == 0):
        TN = TN + 1
    elif (l1 == 1 and l2 == 0):
        FN = FN + 1
    elif (l1 == 0 and l2 == 1):
        FP = FP + 1

print("Confusion Matrix: ")

print("TP=", TP)
print("TN=", TN)
print("FP=", FP)
print("FN=", FN)

# Precision = TruePositives / (TruePositives + FalsePositives)
# Recall = TruePositives / (TruePositives + FalseNegatives)


P = TP/(TP + FP)
R = TP/(TP + FN)

print("Precision = ", P)
print("Recall = ", R)

#F-Measure = (2 * Precision * Recall) / (Precision + Recall), sometimes called F1

F1 = (2* P * R)/(P + R)

print("F score = ")
print(F1)
