

import numpy as np


a = np.random.randn(25,37,12).astype(np.float32)
a = a.flatten()
b = np.random.randn(25,37,12).astype(np.float32)
b = b.flatten()

f = open("randomA.txt", "w")
for i in a:
	f.write(str(i) + "\n")
f.close()

f = open("randomB.txt", "w")
for i in b:
	f.write(str(i) + "\n")
f.close()

