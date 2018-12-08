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

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


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


#BEGIN SGD
print("Training model...")


clf = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
	       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
	       learning_rate='optimal', loss='log', max_iter=None, n_iter=None,
	       n_jobs=-1, penalty='l2', power_t=0.5, random_state=None, shuffle=True,
	       tol=0.001, verbose=1, warm_start=False,early_stopping=False,validation_fraction=0.05,n_iter_no_change=9)
#test encoded data set
enc = LabelEncoder()
enc.fit(Y_train.values)
Y_train_encoded = enc.transform(Y_train.values)
Y_test_encoded = enc.transform(Y_test.values)


#Train model
param_range = np.array([10,100,1000,10000,None])
train_scores, test_scores = validation_curve(
    clf, X_train, Y_train_encoded, param_name="n_iter", param_range=param_range,
    cv=3, scoring="accuracy", n_jobs=-1,verbose=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SGD")
plt.xlabel("n_iter")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
plt.savefig("SGD_plot_n_iter_log.png")
# model = clf.fit(X_train, Y_train_encoded)
# dump(clf,'SVCVGG16.joblib')


 
#test score
score = clf.score(X_test,Y_test_encoded)
print("The final test score is: ",score)

#SVM score 0.60








# title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# # SVC is more expensive so we do a lower number of CV iterations:
# cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)
# estimator = SVC(gamma=0.001)
# plot_learning_curve(estimator, title, X, y, ylim=(0.2, 1.01), cv=cv, n_jobs=4)

# plt.show()
