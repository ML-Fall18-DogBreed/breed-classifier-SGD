print(__doc__)


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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


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
print(X[100].shape)
print("X: ",(X[100]==X[1005]).all())


#split data into train, test
X_train, X_test, Y_train, Y_test = train_test_split(
	X,
	Y,
	test_size=0.1,
	shuffle=True,
	random_state=None,
)


#BEGIN SVM
print("Training model...")

C=1.0
clf = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,eta0=0.0, fit_intercept=True,
                    l1_ratio=0.15, learning_rate='optimal', loss='log', max_iter=None, n_iter=None,n_jobs=-1, penalty='l2',
                    power_t=0.5, random_state=None, shuffle=True,tol=0.001, verbose=1, warm_start=False)
#test encoded data set
enc = LabelEncoder()
enc.fit(Y_train.values)
Y_train_encoded = enc.transform(Y_train.values)
Y_test_encoded = enc.transform(Y_test.values)

print("X_train: ",(X_train[100]==X_train[1001]).all())
print("X_test: ",(X_test[1]==X_test[1001]).all())

model = load("model.joblib")
#model = clf.fit(X_train, Y_train_encoded)
#dump(clf,'SGD_HOG_LOG.joblib')

#test score
Y_predicted = []
Y_predicted_8 = model.predict(X_test)
for i in Y_predicted_8:
	Y_predicted.append(i.decode('UTF-8'))


print("Percentage correct: ", 100*np.sum(Y_test.values == Y_predicted)/len(Y_test.values))

#SVM score 0.60





