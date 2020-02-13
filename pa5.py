import tensorflow as tf
import numpy as np
from model import *
from load_data import *
import dlib
import math
import matplotlib.pyplot as plt

def init_model():
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

	return model

def heatmap_k(x_k, y_k, k):
	return np.array([math.exp(-((x - x_k)**2 + (y - y_k)**2)) for x in range(64) for y in range(64)]).reshape((64, 64))

def generate_heatmaps(resized_landmarks):
	resize_ratio = 64/256
	heatmap_landmarks = np.round(resize_ratio * resized_landmarks)
	heatmaps = []
	for k in range(heatmap_landmarks.shape[0]):
		x_k = heatmap_landmarks[k, 0]
		y_k = heatmap_landmarks[k, 1]
		heatmap = heatmap_k(x_k, y_k, k)
		heatmaps.append(heatmap)
	heatmaps = np.array(heatmaps).reshape((64, 64, 68))
	return heatmaps

def preprocess(img, landmarks):
	face_detector = dlib.get_frontal_face_detector()
    detections = face_detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    boxes = [[d.left(), d.top(), d.right(), d.bottom()] for d in detections]

    resized_img = None
    resized_landmarks = None
    heatmaps = None

    # Task 1: Preprocess image using dlib
    if img is not None :
        if len(detections) == 0:
            return None
        d = detections[0]
        crop = img[d.top():d.bottom(), d.left():d.right()]
        resized_img = cv2.resize(crop, (int(256), int(256)))

    # Task 2: Preprocess ground truth landmarks
    if landmarks is not None:
        resize_ratio = 256/d.width()
        translate = np.array([d.left(), d.top()])
        resized_landmarks = np.round(resize_ratio * (landmarks - translate))
        heatmaps = generate_heatmaps(resized_landmarks)

    return (resized_img, resized_landmarks, heatmaps)

def batch_preprocess(img_data, landmark_data, n):
	counter = 0
	img_store = []
	hm_store = []
	ind_store = []
	for index, (img, lm) in enumerate(zip(img_data, landmark_data)):
		processed_data = preprocess(img, lm)
		if processed_data:
			(resized_img, resized_landmarks, heatmaps) = processed_data
			img_store.append(resized_img)
			hm_store.append(heatmaps)
			ind_store.append(index)
			counter += 1
		if counter >= n:
			break
	return (np.array(img_store), np.array(hm_store), ind_store)

def plot_loss(history, filename):
	plt.plot(history.history['loss'], label='Training')
	plt.plot(history.history['val_loss'], label='Validation')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(filename)
	# plt.show()
	plt.close()
 
def train(model, X_train, Y_train, X_val, Y_val):
	X_train = tf.convert_to_tensor(X_train/255.0, dtype=tf.float64)
	Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float64)
	X_val = tf.convert_to_tensor(X_val/255.0, dtype=tf.float64)
	Y_val = tf.convert_to_tensor(Y_val, dtype=tf.float64)
	
	# Format the labels correctly for 2D-FAN
	Y_train = [Y_train for i in range(4)]
	Y_val = [Y_val for i in range(4)]
	
	# Train model
	history = model.fit(X_train, Y_train, epochs=30, validation_data=(X_val, Y_val))
	plot_loss(history, 'train_val_loss.png')

	return model

def plot_visual(img_data, lm_xs, lm_ys, filename):
	img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
	plt.imshow(img_data)
	plt.scatter(lm_xs, lm_ys, c='r', marker='.')
	plt.savefig(filename)
	plt.close()

def test(model, X_test, Y_test):
	img = tf.convert_to_tensor(X_test/255.0, dtype=tf.float64)
	preds = model(img)

	# Use 4th hourglass module element as heatmap prediction
	# preds[i] has shape: (1, 64, 64, 68), heatmap_preds has shape: (64, 64, 68)
	heatmap_preds = preds[3][0, :, :, :] #TODO: change this to 3 after training
	# print(heatmap_preds.shape)

	# Argmax to convert heatmaps to landmarks
	landmark_preds = []
	for k in range(heatmap_preds.shape[2]):
		heatmap_pred = heatmap_preds[:, :, k]
		ind = np.unravel_index(np.argmax(heatmap_pred, axis=None), heatmap_pred.shape)
		landmark_preds.append(list(ind))
	landmark_preds = np.array(landmark_preds)
	# print(landmark_preds.shape)

	# On 64 x 64 scale
	return landmark_preds

def main():
	# Task 1
	model = init_model()

	# Task 2 (see load_data.py)
	results = load_data()

	# Task 3
	# Prepare data for training
	(X_train, Y_train, ind_train) = batch_preprocess(results['images_train'], results['landmarks_train'], 16)
	(X_val, Y_val, ind_val) = batch_preprocess(results['images_val'], results['landmarks_val'], 2)
	(X_test, Y_test, ind_test) = batch_preprocess(results['images_test'], results['landmarks_test'], 1)
	#print(X_train.shape, Y_train.shape)
	#print(X_val.shape, Y_val.shape)
	#print(X_test.shape, Y_test.shape)

	# Task 4
	model = train(model, X_train, Y_train, X_val, Y_val)

	# Task 5
	landmark_preds = test(model, X_test, Y_test)

	orig_img = results['images_test'][ind_test[0]]
	orig_lms = results['landmarks_test'][ind_test[0]]

	# Get bounding box size for test image for rescaling up
	face_detector = dlib.get_frontal_face_detector()
	detections = face_detector(cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY))
	d = detections[0]

	# Plot predicted landmarks on original image
	resize_ratio = d.width()/64
	translate = np.array([d.left(), d.top()])
	resized_landmarks = np.round((resize_ratio * landmark_preds) + translate)
	plot_visual(orig_img, resized_landmarks[:, 0], resized_landmarks[:, 1], 'test_predlm_plot.png')

	# Optional ground truth landmark plot
	plot_visual(orig_img, orig_lms[:, 0], orig_lms[:, 1], 'test_gtlm_plot.png')

	# Sanity check on 256 x 256 ground truth landmark plot and pred landmark plot
	# (resized_img, resized_lms, heatmaps) = preprocess(orig_img, orig_lms)
	# resize_ratio = 256/64
	# resized_landmarks = resize_ratio * landmark_preds
	# plot_visual(resized_img, resized_lms[:, 0], resized_lms[:, 1], 'test_smallgtlm_plot.png')
	# plot_visual(X_test[0, :, :, :], resized_landmarks[:, 0], resized_landmarks[:, 1], 'test_smallpredlm_plot.png')

	# Extra Credit
	sanders_img = cv2.imread('sanders.png')
	sanders_test = np.array([preprocess(sanders_img, None)[0]])

	sanders_landmark_preds = test(model, sanders_test)

	# Get bounding box size for sanders image for rescaling up
	face_detector = dlib.get_frontal_face_detector()
	detections = face_detector(cv2.cvtColor(sanders_img, cv2.COLOR_BGR2GRAY))
	d = detections[0]

	# Plot predicted landmarks on original image
	resize_ratio = d.width()/64
	translate = np.array([d.left(), d.top()])
	resized_landmarks = np.round((resize_ratio * sanders_landmark_preds) + translate)
	plot_visual(sanders_img, resized_landmarks[:, 0], resized_landmarks[:, 1], 'sanders_predlm_plot.png')

if __name__ == "__main__":
	main()