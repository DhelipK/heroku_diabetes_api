import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv("diabetes.csv")
diabetes_dataset.head()

diabetes_dataset['Outcome'].value_counts()

X = diabetes_dataset.drop(columns=['Outcome'], axis=1)
y = diabetes_dataset['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train,y_train)

X_train_pred = classifier.predict(X_train)
training_accuracy = accuracy_score(X_train_pred,y_train)
print('Accuracy score of the training data : ', training_accuracy)

X_test_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(X_test_pred, y_test)
print('Accuracy score of the test data : ', test_accuracy)

filename = 'model.pkl'
pickle.dump(classifier, open(filename, 'wb'))