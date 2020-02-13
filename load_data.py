import numpy as np
import os
import scipy # For loading the .mat files
from scipy import io
import cv2

def load_data():
	np.random.seed(42)
	images_path = '../AFW/images/'
	landmarks_path = '../AFW/landmarks/'
	fnames = sorted(os.listdir(images_path))

	np.random.shuffle(fnames)

	images = [] 
	landmarks = [] 

	for filename in fnames[:100]:
	# TODO: Read in images and landmarks together
	# Create training/validation/test splits
		images.append(cv2.imread(images_path + filename))
		landmarkfilename = filename.replace(".jpg", "_pts.mat")
		landmarks.append(scipy.io.loadmat(landmarks_path + landmarkfilename)['pts_2d'])

	images = np.array(images)
	landmarks = np.array(landmarks)
	N = 100

	results = {}
	results['images_train'] = images[:int(.8*N)] # First 80% of data
	results['landmarks_train'] = landmarks[:int(.8*N)]
	results['images_val'] = images[int(.8*N):int(0.9*N)]
	results['landmarks_val'] = landmarks[int(.8*N):int(0.9*N)]
	results['images_test'] = images[int(0.9*N):]
	results['landmarks_test'] = landmarks[int(0.9*N):]
	return results

# results = load_data()
# print(results['images_train'].shape)