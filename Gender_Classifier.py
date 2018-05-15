from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np


X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

DTree = tree.DecisionTreeClassifier()
SVM = SVC()
Perceptron = Perceptron()
kNN = KNeighborsClassifier()

DTree.fit(X,Y)
SVM.fit(X,Y)
Perceptron.fit(X,Y)
kNN.fit(X,Y)

pred_DTree = DTree.predict(X)
acc_DTree = accuracy_score(Y,pred_DTree)
print('Accuracy of Decision Tree: {}'.format(acc_DTree))

pred_SVM = SVM.predict(X)
acc_SVM = accuracy_score(Y,pred_SVM)
print('Accuracy of SVM: {}'.format(acc_SVM))

pred_Perceptron = Perceptron.predict(X)
acc_Perceptron = accuracy_score(Y,pred_Perceptron)
print('Accuracy of Perceptron: {}'.format(acc_Perceptron))

pred_kNN = kNN.predict(X)
acc_kNN = accuracy_score(Y,pred_kNN)
print('Accuracy of kNN: {}'.format(acc_kNN))

index = np.argmax([acc_DTree,acc_SVM,acc_Perceptron,acc_kNN])

classifiers = {'Decision Tree':0, 'Support Vector Machine':1, 'Perceptron':2, 'K Nearest Neighbors':3}

print('The algorithm with the highest accuracy is: {}'.format(classifiers[index]))