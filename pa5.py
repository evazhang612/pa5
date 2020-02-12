import tensorflow as tf
import numpy as np
from model import *
from load_data import *
import dlib
import matplotlib.pyplot as plt

# Task 2 see load_data.py
# Task 3 
def img_preprocess(img):
	face_detector = dlib.get_frontal_face_detector()
	# img = cv2.imread(PATH_TO_IMAGE)
	detections = face_detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
	boxes = [[d.left(), d.top(), d.right(), d.bottom()] for d in detections]
	
	# Task 1: Preprocess image using dlib
	# TODO: Use the face bounding box output by dlib to crop the image
	# and resize the resulting crop to 256 x 256 x 3.
	if len(detections) == 0:
		return None
	d = detections[0]
	crop = img[d.top():d.bottom(), d.left():d.right()]
	resized_img = cv2.resize(crop, (int(256), int(256)))

	return resized_img

def lm_preprocess(landmark):
	# Task 2: Preprocess ground truth landmarks
	resize_ratio = 256/d.width()
	# mat = scipy.io.loadmat(PATH_TO_LANDMARKS)
	# landmarks = np.array(mat['pts_2d'])
	translate = np.array([d.left(), d.top()])
	resized_landmarks = np.round(resize_ratio * (landmarks - translate))
	return resized_landmark

# Task 4 
def train(results):
	# Prepare data for training
	# @Lucia, this should work from batch preprocess, feel free to change it 
	# X_train = results[''] # Should have shape (16, 256, 256, 3)
	# Y_train = # Should have shape (16, 64, 64, 68)
	# X_val = # Should have shape (2, 256, 256, 3)
	# Y_val = # Should have shape (2, 64, 64, 68)
	X_train = np.array([img_preprocess(img) for img in results['images_train']][:16])
	Y_train = np.array([lm_preprocess(lm) for idx, lm in enumerate(results['landmarks_train']) if X_train[idx] is not None][:16])
	X_val = np.array([img_preprocess(img) for img in results['images_val']][:2])
	Y_val = np.array([lm_preprocess(lm) for idx,lm in enumerate(results['landmarks_val']) if X_val[idx] is not None][:2])
	X_test = np.array([img_preprocess(img) for img in results['images_test']][:1])
	Y_test = np.array([lm_preprocess(lm) for idx, lm in enumerate(results['landmarks_test']) if X_test[idx] is not None ][:1])

	X_train = tf.convert_to_tensor(X_train/255.0, dtype=tf.float64)
	Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float64)
	X_val = tf.convert_to_tensor(X_val/255.0, dtype=tf.float64)
	Y_val = tf.convert_to_tensor(Y_val, dtype=tf.float64)
	# Format the labels correctly for 2D-FAN
	Y_train = [Y_train for i in range(4)]
	Y_val = [Y_val for i in range(4)]
	# Train model
	history = model.fit(X_train, Y_train, epochs=30, validation_data=(X_val, Y_val))
	plot(history)

def plot(history):
	plt.plot(history.history['loss'], label='Training')
	plt.plot(history.history['val_loss'], label='Validation')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

# Task 1 
model = FAN(4)
model.build((1,256,256,3))
model.load_weights("tf_fan_2D_3layers.h5")

model.base.trainable = False # Freezes the weights for the base module
for i in range(3):
	# TODO: Freeze the weights for the first 3 hourglass modules
	# Each hourglass module is composed of elements from
	# model.hgs, model.ls, model.split_as, model.split_bs
	model.hgs[i].trainable = False 
	model.ls[i].trainable = False 
	model.split_as[i].trainable = False 
	model.split_bs[i].trainable = False 

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
results = load_data()
# preprocessed_results = batch_preprocess(results)

train(results)
