print(__doc__)

from time import time
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load
from sklearn.linear_model import SGDClassifier
import tensorflow as tf
from tensorflow import keras
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def loadDataset():	
	Xi = []
	data = pd.read_csv("labels.csv")
	filelist = data['id']
	Y = data['breed']
	for fname in filelist:
		Xi.append(np.load('Train_resized_hog_np/'+fname+'.npy'))
	X = np.array(Xi)
	print(X.shape)
	return X, Y



#loading dataset
print("loading dataset")
X, Y = loadDataset()
print("dataset loaded")


#split data into train, test
X_train, X_test, Y_train, Y_test = train_test_split(
	X,
	Y,
	test_size=0.1,
	shuffle=True,
	random_state=None,
)

#encode data set
enc = LabelEncoder()
enc.fit(Y_train.values)
Y_train_encoded = enc.transform(Y_train.values)
Y_test_encoded = enc.transform(Y_test.values)

#BEGIN SVM
print("Training model...")

C=1.0
clf = svm.SVC(kernel='linear', C=C,verbose=1)
#set degree param
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'degree': [2, 3, 4, 5], 'kernel': ['poly']},
 ]

grid_search = GridSearchCV(clf, param_grid=param_grid, cv=3)
start = time()
grid_search.fit(X_train, Y_train_encoded)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


#model = clf.fit(X_train, Y_train_encoded)
#dump(clf,'SVM_HOG_LINEAR.joblib')

#test score
# print("testing score: ")
# Y_predicted = []
# Y_predicted_8 = model.predict(X_test)
# # for i in Y_predicted_8:
# # 	Y_predicted.append(i.decode('UTF-8'))
#
#
#
# print("Percentage correct: ", 100*np.sum(Y_test_encoded == Y_predicted_8)/len(Y_test.values))

#SVM score 0.60




