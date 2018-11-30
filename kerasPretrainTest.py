from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn import metrics




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
model = KMeans(n_clusters=120, random_state=0).fit(X_train)
print("predicting on test set")
Y_predicted = model.predict(X_test)
score = metrics.adjusted_rand_score(Y_test, Y_predicted)

print("kmeans score = ", score)