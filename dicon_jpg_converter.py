import numpy as np
import pydicom
from pydicom import dcmread
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import glob
import os

dicon_files = glob.glob('.\\dicon\\CG2A070680_20181127_203739.dcm')

def preprocessing(arr):
	threshold = 0.25
	upperbound = (1 - threshold)*65535
	lowerbound = threshold*65535

	for index in np.nditer(arr, op_flags=['readwrite']):
		if index > upperbound:
			index[...] = 65535
		elif index < lowerbound:
			index[...] = 0

		index[...] = np.uint(index[...] * 256/65536)
	return arr

for i in dicon_files:
	write_path = os.path.splitext(os.path.abspath(i.replace("dicon", "png")))[0] + "test" + ".png"
	ds = dcmread(i)
	arr = ds.pixel_array

	print(arr.shape[0])
	print(arr.shape[1])
	if arr.shape[0] > arr.shape[1]:
		arr = np.rot90(arr)
	
	img = preprocessing(arr.copy())
	plt.imsave(write_path, img, cmap='Greys')
	print('iteration')
	exit()