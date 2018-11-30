#matplotlib inline
import pandas as pd
from PIL import Image
import glob
import Transformers
import numpy as np
from skimage import io
import pprint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import skimage
#from joblib import dump, load
import joblib
from skimage.transform import resize
pp = pprint.PrettyPrinter(indent=4)


#read image data
data = pd.read_csv("labels.csv")
filelist = data['id']
Y = data['breed']
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
# plt.imshow(X_train[0])
# print(X_train.shape)
# plt.imshow(X_train[1000])
# print(type(y_train))
# print(y_train.shape)
# plt.show()


#process with hog
#create an instance of each transformer

grayify = Transformers.RGB2GrayTransformer()
hogify = Transformers.HogTransformer(
    pixels_per_cell=(8, 8),
    cells_per_block=(2,2),
    orientations=9,
    block_norm='L2-Hys'
)
scalify = StandardScaler()

print("transforming data")
# call fit_transform on each transform converting X_train step by step
X_train_gray = grayify.fit_transform(X_train)
X_train_hog = hogify.fit_transform(X_train_gray)
X_train_prepared = scalify.fit_transform(X_train_hog)

X_test_gray = grayify.fit_transform(X_test)
X_test_hog = hogify.fit_transform(X_test_gray)
X_test_prepared = scalify.fit_transform(X_test_hog)


print(X_test_prepared.shape)
print("begin training")

# #train
sgd_clf = SGDClassifier(alpha=0.00001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='log', max_iter=100, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       tol=0.001, verbose=1, warm_start=False,early_stopping=True,validation_fraction=0.1,n_iter_no_change=9)
model = sgd_clf.fit(X_train_prepared, y_train)

joblib.dump(sgd_clf,'model3.joblib')

# model = joblib.load('model2.joblib')
predicted = model.predict(X_test_prepared)

print(np.array(y_test == predicted)[:25])
print("Percentage correct: ", 100*np.sum(y_test == predicted)/len(y_test))


print("Done training")
score = model.score(X_test_prepared,y_test)
print("score: ",score)


#max_iter = 100, tol = .001 score = .77
#max_iter = 1000, tol = .0001, score = .03
#max_iter = 200, tol = .0005 score = .029
##max_iter = 50, tol = .001 score =  .0286

