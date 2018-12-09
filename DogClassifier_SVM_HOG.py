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
clf = svm.SVC(kernel='linear', C=C,verbose=1)
#test encoded data set
enc = LabelEncoder()
enc.fit(Y_train.values)
Y_train_encoded = enc.transform(Y_train.values)
Y_test_encoded = enc.transform(Y_test.values)


model = clf.fit(X_train, Y_train_encoded)
dump(clf,'SVM_HOG_LINEAR.joblib')

#test score
score = clf.score(X_test,Y_test_encoded)
print("The final test score is: ",score)

#SVM score 0.60




