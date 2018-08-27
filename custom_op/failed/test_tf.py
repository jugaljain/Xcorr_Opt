import tensorflow as tf
import itertools 
import time

def fn(b):
	ra = [range(5), range(5)]
	window = list(itertools.product(*ra))
	out = []
	windows = tf.zeros([1,5,5])
	for i in range(0, 42):
		for j in range(0, 12):
			win = [(x+i,y+j) for x,y in window]
			w = tf.convert_to_tensor(win)
			w = tf.reshape(w, [5, 5, 2])
			filt = tf.gather_nd(b, w)
			filt = tf.reshape(filt, [1,5,5])
			windows = tf.concat([windows, filt], axis=0)
			if (i == 0 & j == 1):
				windows = windows[1:,:,:]
			if i > 4:
				out.append(windows[(12*(i-5)):(12*(i)),:,:])
	o = tf.stack(out)
	o = tf.reshape(o,[37,12,60,5,5])
	return(o)

def fn2(a):
	ra = [range(5), range(5)]
	window = list(itertools.product(*ra))
	out = []
	for i in range(0, 37):
		for j in range(0, 12):
			win = [(x+i,y+j) for x,y in window]
			w = tf.convert_to_tensor(win)
			w = tf.reshape(w, [5, 5, 2])
			filt = tf.gather_nd(a, w)
			filt = tf.reshape(filt, [1,5,5])
			out.append(tf.tile(filt, [60,1,1]))

	o = tf.stack(out)
	o = tf.reshape(o,[37,12,60,5,5])


a = tf.random_normal((25,37,12))
b = tf.random_normal((25,37,12))
sess = tf.Session()

b = tf.pad(b, [[0,0],[4,4],[2,2]])
m1 = tf.map_fn(fn, b, dtype=tf.float32, parallel_iterations=25)
start = time.time()
out = sess.run(m1)
end = time.time()
print("Op1: " + str(end-start))
print(out[0,0,0])

a = tf.pad(a, [[0,0],[2,2],[2,2]])
m2 = tf.map_fn(fn2, a, dtype=tf.float32, parallel_iterations=25)
start = time.time()
out = sess.run(m2)
end = time.time()
print("Op2: " + str(end-start))
print(out[0,0,0])

