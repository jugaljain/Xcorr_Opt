
import numpy as np
import tensorflow as tf
from keras import backend as K
import time

input_dims = (12,37,1)

a = K.random_normal(input_dims)
b = K.random_normal(input_dims)

def Xcorr(p):
	a = p[0]
	b = p[1]

	#with tf.device('/device:GPU:0'):
		#make array of all possible filters
	a = tf.pad(a, [[2,2],[2,2],[0,0]])
	b = tf.pad(b, [[2,2],[4,4],[0,0]])
	filt_box = []

	for c in range(input_dims[2]):
		filt_list = []
		#across a row
		for i in range(0, input_dims[1]+4):
			for j in range(0,input_dims[0]):
				window = b[j:j+5, i:i+5, c]
				window = tf.reshape(window, [5,5,1])
				filt_list.append(window)
		filt_box.append(filt_list)


start = time.time()
Xcorr([a,b])
end = time.time()
print("Execution time: " + str(end - start) + " Seconds")
