import numpy as np
#import cupy as cp
#import tensorflow as tf
#from keras import backend as K
import time
import matplotlib.pyplot as plt

#tf.test.gpu_device_name()

# Numpy implementation

# Function to find normalized correlation between two tensors

def normxcorr(x, y):
    N = 25
    e = 0.01
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    sd_x = np.std(x)+e
    sd_y = np.std(y)+e
    numerator = np.sum((x-mean_x)*(y-mean_y))
    denominator = (N-1)*sd_x*sd_y
    normxcorr = numerator/ denominator
    return normxcorr
    
def dist(x,y):
  dist = cp.linalg.norm(x-y)
  return dist
  
# Normalized cross correlation    
    
def Xcorr(p):

    out = []
    a = p[0]
    b = p[1]
    
    # converting tf tensors to numpy array
    #init = tf.global_variables_initializer()
    
    #with sess.as_default():
    #    a = a.eval(sess.run(init))
    #with sess.as_default():
    #    b = b.eval(sess.run(init))
    #with cp.cuda.Device(0):
        # Padding a, b with zeros of width 2 and 4 respectively
    a = np.pad(a, [(0,), (2,), (2,)], 'constant')
    b = np.pad(b, [(0,), (4,), (2,)], 'constant')

    for c in range(25): # Iterating through the channels of a,b
        # Selecting pixels from a
        for i in range(2, 39): # i is vertical
            for j in range(2, 14): # j is horizontal
                E = a[c, i-2:i+3, j-2:j+3]
                E = E.flatten()
                # Selecting pixels from b
                for k in range(i-2, i+3):
                    for l in range(2, 14):
                        F = b[c, k:k+5, l-2:l+3]
                        F = F.flatten()
                        distance = normxcorr(E, F)
                        out.append(distance)
    return out

print("Starting program.")

f = open("randomA.txt", "r")
a = []
l = f.readline()
while l:
    a.append(float(l.strip("\n")))
    l = f.readline()
a = np.asarray(a)
a = np.reshape(a, (25,37,12)).astype(np.float32)
f.close()

f = open("randomB.txt", "r")
b = []
l = f.readline()
while l:
    b.append(float(l.strip("\n")))
    l = f.readline()
b = np.asarray(b)
b = np.reshape(b, (25,37,12)).astype(np.float32)
f.close()


start = time.time()
print("Start time = " + str(start))
output = Xcorr([a,b])
end = time.time()
print("Execution time: " + str(end - start) + " Seconds")

output = np.asarray(output)
output = np.reshape(output, (25,444,60))
print(output[0,0])
print(a[0])