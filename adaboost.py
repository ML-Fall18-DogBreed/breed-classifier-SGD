#matplotlib inline
import pandas as pd
from PIL import Image
import glob
import Transformers
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pprint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import skimage
from joblib import dump, load
from skimage.transform import resize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

def imagesToFeatures(X):
	X_new = np.empty((X.shape[0],30000))

	for image in range(X.shape[0]):
		print("processing image %s / %s"%(image, X.shape[0]))
		X_new[image]=imageToFeature(X[image])
	return X_new


def imageToFeature(x):
	#1xnumPixels
	x = x.reshape((1,30000))
	return x


pp = pprint.PrettyPrinter(indent=4)


#read image data
data = pd.read_csv("labels.csv")
filelist = data['id']
#filelist = filelist[0:100]
Y = data['breed']
#Y = Y[0:100]
#y_train = Y[:2]
#X_train = np.array([resize(np.array(Image.open('Train/'+filelist[a]+'.jpg')),(75,100),anti_aliasing=True) for a in range(2)])
print("loading images")
X = np.array([np.array(Image.open('Train_resized/'+fname+'.jpg')) for fname in filelist])




#split validation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.15,
    shuffle=True,
    random_state=None,
)

grayify = Transformers.RGB2GrayTransformer()
hogify = Transformers.HogTransformer(
    pixels_per_cell=(8, 8),
    cells_per_block=(2,2),
    orientations=9,
    block_norm='L2-Hys'
)
scalify = StandardScaler()

print("to gray")
X_train_gray = grayify.fit_transform(X_train)
X_test_gray = grayify.fit_transform(X_test)
del X_train
del X_test

print("to Hog")
X_train_hog = hogify.fit_transform(X_train_gray)
X_test_hog = hogify.fit_transform(X_test_gray)
print("scale")
X_train_hog = scalify.fit_transform(X_train_hog)
X_test_hog = scalify.fit_transform(X_test_hog)

# print("to pixels")
# X_train_pixels = imagesToFeatures(X_train_gray)
# X_test_pixels = imagesToFeatures(X_test_gray)


del X_train_gray
del X_test_gray

X_train_prepared = X_train_hog#np.hstack((X_train_hog, X_train_pixels))
X_test_prepared = X_test_hog#np.hstack((X_test_hog, X_test_pixels))

#del X_train_hog
#del X_train_pixels
#del X_test_hog
#del X_test_pixels


##need an accuracy > 1/120 ie random case .00833
print("begin training")
##Testing individual decision tree classifiers
# #max depth = 5 pixels only : score of 0.0156
# #max depth = 1 pixels only : score of .0136 in 1minute 30
# 
# #max dpeth = 1 hog only: score of 0.0123
# # max depth =2 hog only score of 0.117
# 

# with 150 estimators it is worthless - .0156

# Create adaboost-decision tree classifer object
depth = 10
n_est = 1
print("training with max depth %s and n_estimators %s"%(depth, n_est))
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth),
						 n_estimators=n_est,
                         learning_rate=1,
                         random_state=None)



#Train model
model = clf.fit(X_train_prepared, y_train)
dump(clf,'adaboostHogModel.joblib')
score = clf.score(X_test_prepared,y_test)
print("score: ",score)

