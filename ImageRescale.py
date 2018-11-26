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


#read image data
data = pd.read_csv("labels.csv")
filelist = data['id']
for fname in filelist:
    img = cv2.imread('Train/'+fname+'.jpg')
    res = cv2.resize(img,dsize=(200,150),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('Train_resized/'+fname+'.jpg',res)
