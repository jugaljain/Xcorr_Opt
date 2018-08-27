import tensorflow as tf
import numpy
example_lib = tf.load_op_library('unroll_op.so')


f = open("randomA.txt", "r")
a = []
l = f.readline()
while l:
	a.append(float(l.strip("\n")))
	l = f.readline()

a = numpy.asarray(a)
a = numpy.reshape(a, (25,37,12)).astype(numpy.float32)
f.close()

f = open("randomB.txt", "r")
b = []
l = f.readline()
while l:
	b.append(float(l.strip("\n")))
	l = f.readline()

b = numpy.asarray(b)
b = numpy.reshape(b, (25,37,12)).astype(numpy.float32)
f.close()

a = tf.convert_to_tensor(a);
b = tf.convert_to_tensor(b);


with tf.device('/gpu:0'):
    out = example_lib.norm_x_corr(a, b, name='out')
sess = tf.Session()
output = sess.run(out)
print(output[0,0])