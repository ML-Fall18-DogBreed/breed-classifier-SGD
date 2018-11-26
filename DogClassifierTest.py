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
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import skimage
from joblib import dump, load
import cv2


data = pd.read_csv("labels.csv")
filelist = data['id']
Y = data['breed']
#y_train = Y[:2]
#X = np.array([np.array(Image.open('Train/'+filelist[a]+'.jpg') for a in range(2)])
X = np.array([np.array(Image.open('Train_resized/'+fname+'.jpg')) for fname in filelist])
#print(X[0].shape)
#plt.imshow(X[0])
#plt.show()

#split validation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.85,
    shuffle=True,
    random_state=42,
)


#process with hog
# create an instance of each transformer

grayify = Transformers.RGB2GrayTransformer()
hogify = Transformers.HogTransformer(
    pixels_per_cell=(8, 8),
    cells_per_block=(2,2),
    orientations=9,
    block_norm='L2-Hys'
)
scalify = StandardScaler()

# call fit_transform on each transform converting X_train step by step
X_test_gray = grayify.fit_transform(X_test)
X_test_hog = hogify.fit_transform(X_test_gray)
X_test_prepared = scalify.fit_transform(X_test_hog)
sgd_clf = load('model.joblib')
y_pred = sgd_clf.predict(X_test_prepared)
print(np.array(y_pred == y_test)[:25])
print('')
print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

