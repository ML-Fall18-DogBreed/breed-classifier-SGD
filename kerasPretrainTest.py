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




def loadDataset():	
	vgg16_feature_list = []
	data = pd.read_csv("labels.csv")
	filelist = data['id']
	Y = data['breed']
	model = VGG16(weights='imagenet', include_top=False)
	#shrink dataset for testing
	# filelist = filelist[0:100]
	# Y = Y[0:100]

	for fname in filelist:
		try:
			vgg16_feature_list.append(np.load('Train_resized_np/'+fname+'.npy'))
		except:
			img_path = 'Train_resized/'+fname+'.jpg'
			vgg16_feature_np = imageToFeatures(img_path,model)
			np.save('Train_resized_np/'+fname+'.npy', vgg16_feature_np)
			vgg16_feature_list.append(vgg16_feature_np)

	X = np.array(vgg16_feature_list)
	print(X.shape)

	return X, Y
def imageToFeatures(img_path, model):
	img = image.load_img(img_path, target_size=(224, 224))
	img_data = image.img_to_array(img)
	img_data = np.expand_dims(img_data, axis=0)
	img_data = preprocess_input(img_data)
	vgg16_feature = model.predict(img_data)
	vgg16_feature_np = np.array(vgg16_feature)
	return vgg16_feature_np.flatten()

print("loading dataset")
X, Y = loadDataset()
print("dataset loaded")

X_train, X_test, Y_train, Y_test = train_test_split(
	X,
	Y,
	test_size=0.15,
	shuffle=True,
	random_state=None,
)
print("Training model...")
depth = 2
n_est = 2400
# print("sgd classifier")
# print("training with max depth %s and n_estimators %s"%(depth, n_est))
# clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth),
# 						 n_estimators=n_est,
#                          learning_rate=1,
#                          random_state=None)
# clf = SGDClassifier(alpha=0.00001, average=False, class_weight=None, epsilon=0.1,
# 	       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
# 	       learning_rate='optimal', loss='log', max_iter=200, n_iter=None,
# 	       n_jobs=1, penalty='l2', power_t=0.5, random_state=None, shuffle=True,
# 	       tol=0.0001, verbose=1, warm_start=False,early_stopping=True,validation_fraction=0.05,n_iter_no_change=9)



# print(Y_train.values[:10])


##BEGIN SVM
# C=1.0
# clf = svm.SVC(kernel='linear', C=C)

# #Train model
# model = clf.fit(X_train, Y_train)
# dump(clf,'SGDVGG16.joblib')
# score = clf.score(X_test,Y_test)
# print("score: ",score)


# # model = KMeans(n_clusters=120, random_state=0).fit(X_train)
# #Kmeans performed very poorly < 4%
# print("predicting on test set")

# Y_predicted = model.predict(X_test)
# print(np.array(Y_test == Y_predicted)[:25])
# print("Percentage correct: ", 100*np.sum(Y_test == Y_predicted)/len(Y_test))

# score = metrics.adjusted_rand_score(Y_test, Y_predicted)

#adaboost with decision tree classifiers
# 2, 20 = 5%
# 200, 1 = 13%
# 5, 100 = .054
# 1200,2 = 12.64
# 10, 20, =10 %
# 10, 120 = 9.77%
# 2,600 = 3.78%
# 5,100 = 5.4%
# 
#SGD on VG16
#max iter 100, score = 50
#max iter 200, score = 53
#max iter 2000, score = 54
#max iter 200, tolerance = 0.0001, score = 52



#Begin neural network
print("nn")
enc = LabelEncoder()
enc.fit(Y_train.values)
Y_encoded = enc.transform(Y_train.values)
print(Y_encoded[:25])
Y_test_encoded = enc.transform(Y_test.values)

model = keras.Sequential([
    keras.layers.Dense(len(X_train[0]), activation=tf.nn.relu),
    keras.layers.Dense(120, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_encoded, epochs=5)

Y_predicted = model.predict(X_test)
print(np.array(Y_test_encoded == Y_predicted)[:25])
print("Percentage correct: ", 100*np.sum(Y_test_encoded == Y_predicted)/len(Y_test_encoded))
