import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## Load Iris dataset
df = pd.read_csv('Iris.csv')
## Retrieve the target values and drop the Id along with it
target = df['Species']
df = df.drop(['Species','Id'],axis=1)
## Drop the two features we won't be using from the dataframe
df = df.drop(['SepalWidthCm','PetalWidthCm'],axis=1)

## Retrieve features
X = df.values.tolist()
Y = []
## Convert classes in Strings to Integers
for val in target:
    if(val == 'Iris-setosa'):
        Y.append(0)
    elif(val == 'Iris-virginica'):
        Y.append(2)
    else:
        Y.append(1)
## Make them as numpy array
X = np.array(X)
Y = np.array(Y)
## Shuffle and split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.9)
## Make them as numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


## K Nearest Neighbors

y_pred_knn = []
## Iterate through each value in test data
for val in x_test:
    euc_dis = []
    ## Finding eucledian distance for all points in training data
    for point in x_train:
        euc_dis.append(((val[0]-point[0])**2+(val[1]-point[1])**2)**0.5)
    temp_target = y_train.tolist()
    ## Use bubble sort to sort the euclidean distances
    for i in range(len(euc_dis)):
        for j in range(0,len(euc_dis)-i-1):
            if(euc_dis[j+1] < euc_dis[j]):
                euc_dis[j], euc_dis[j+1] = euc_dis[j+1], euc_dis[j]
                ## Sort the classes along with the eucledian distances
                ## to maintain relevancy
                temp_target[j], temp_target[j+1] = temp_target[j+1], temp_target[j]
    ## Finding majority among the neighbours
    vote = [0,0,0]
    ## We are using only the first three entries (K = 3)
    for i in range(3):
        vote[temp_target[i]] += 1
    y_pred_knn.append(vote.index(max(vote)))
## Print the accuracy score
print('Accuracy:',accuracy_score(y_test,y_pred_knn))