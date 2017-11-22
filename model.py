import numpy as np
import keras
import matplotlib.pyplot as plt
import csv
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *

#resized image dimension in training
img_rows = 16
img_cols = 32

#batch size and epoch
batch_size=128
nb_epoch=15

delta = 0.35

def preprocess(img):
	return cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(img_cols,img_rows))

def load():
	images = []
	angles = []
	# load camera images
	with open('data/driving_log.csv','rt') as f:
		reader = csv.reader(f)
		next(reader, None)
		for line in reader:
			for j in range(2): #center, left, right images
				img = plt.imread('data/'+line[j].strip())
				images.append(preprocess(img))
				angle = float(line[3])
				if j == 1: angle += delta #for left image, we add delta 
				elif j == 2: angle -= delta #for right image we subtract delta
				angles.append(angle)

	# convert to numpy arrays
	X_train = np.array(images).astype('float32')
	y_train = np.array(angles).astype('float32')

	# add augmented data (reflect horizontally each image)
	X_train = np.append(X_train,X_train[:,:,::-1],axis=0)
	y_train = np.append(y_train,-y_train,axis=0)

	# shuffle data
	X_train, y_train = shuffle(X_train, y_train)

	# convert into required shape
	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

	#split into train and validation sets
	return train_test_split(X_train, y_train, random_state=0, test_size=0.25)

if __name__ == '__main__':

	#load data
	X_train, X_val, y_train, y_val = load()

	#Create an train the model
	model = Sequential([
			Lambda(lambda x: x/127.5 - 1.,input_shape=(img_rows,img_cols,1)),
			Conv2D(2, 3, 3, border_mode='valid', input_shape=(img_rows,img_cols,1), activation='relu'),
			MaxPooling2D((4,4),(4,4),'valid'),
			Dropout(0.25),
			Flatten(),
			Dense(1)
		])

	model.compile(loss='mean_squared_error',optimizer='adam')
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_val, y_val))

	# Save the model and weights
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights("model.h5")
	print("Model Saved.")