import numpy as np
import tensorflow as tf
from keras import backend as K
import time
import itertools

input_dims = [25,37,12]

a = K.random_normal(input_dims)
b = K.random_normal(input_dims)

def fn2(p):
	reg = p[0]
	filt = p[1]
	mr,vr = tf.nn.moments(reg, axes=[0,1])
	mf, vf = tf.nn.moments(filt, axes=[0,1])

	reg_norm = tf.divide(tf.subtract(reg, mr), vr)
	filt_norm = tf.divide(tf.subtract(filt, mf), vf)

	fsum = tf.reduce_sum(tf.divide(tf.multiply(reg_norm, filt_norm), 24))
	return fsum 



def fn(p):
	a = p[0]
	b = p[1]
	windows = tf.zeros([1,5,5])
	ra = [range(5), range(5)]
	window = list(itertools.product(*ra))
	#b = tf.pad(b, [[0,0],[2,2],[4,4]])

	for i in range(0, input_dims[1]+4):
		for j in range(0, input_dims[2]+1):
			win = [(x+i,y+j) for x,y in window]
			w = tf.convert_to_tensor(win)
			w = tf.reshape(w, [5, 5, 2])
			filt = tf.gather_nd(b, w)
			count = ((input_dims[1]+4)*i) + j
			filt = tf.reshape(filt, [1,5,5])
			windows = tf.concat([windows, filt], axis=0)

	windows = windows[1:,:,:]

	filt_id = [i for i in range(60)]
	corr_block = tf.zeros([1,13,60])
	for i in range(0, input_dims[1]+2):
		idxs = [x+(12*i) for x in filt_id]
		filts = tf.gather(windows, idxs)
		corrs = tf.zeros([1,60])
		for j in range(0, input_dims[2]+1):
			win = [(x+i,y+j) for x,y in window]
			w = tf.convert_to_tensor(win)
			w = tf.reshape(w, [5, 5, 2])
			reg = tf.gather_nd(a, w)
			reg = tf.reshape(reg, [1,5,5])
			regs = tf.tile(reg, [60,1,1])
			corr = tf.map_fn(fn2,(regs, filts), dtype=tf.float32, parallel_iterations=60)
			corr = tf.reshape(corr, [1,60])
			corrs = tf.concat([corrs, corr], axis=0)
		corrs = corrs[1:,:]
		corrs = tf.reshape(corrs, [1,13,60])
		corr_block = tf.concat([corr_block, corrs], axis=0)
	print("done")
	return corr_block

def Xcorr(p):
	
	# with tf.device('/device:GPU:0'):
	# 	#make array of all possible filters
	# 	a = tf.pad(a, [[0,0],[2,2],[2,2]])
	# 	b = tf.pad(b, [[0,0],[2,2],[4,4]])
	# 	#filt_box = tf.zeros([5,5,input_dims[0]*(input_dims[1]+4), 1])

	# 	filt_list = tf.zeros([1,5,5])
	# 	for c in range(input_dims[0]):
	# 		#across a row
	# 		for i in range(0, input_dims[2]+4):
	# 			for j in range(0,input_dims[1]):
	# 				window = b[c, j:j+5, i:i+5]
	# 				window = tf.reshape(window,[1,5,5])
	# 				filt_list = tf.concat([filt_list, window], axis=0)
	# 	sess = tf.Session()
	# 	print(sess.run(filt_list))
	a = p[0]
	b = p[1]
	sess = tf.Session()
	a = tf.pad(a, [[0,0],[2,2],[2,2]])
	b = tf.pad(b, [[0,0],[4,4],[2,2]])
	m = tf.map_fn(fn, (a,b), dtype=tf.float32, parallel_iterations=25)
	out = sess.run(m)
	print(out[0,0,0])

	

start = time.time()
Xcorr([a,b])
end = time.time()
print("Execution time: " + str(end - start) + " Seconds")

