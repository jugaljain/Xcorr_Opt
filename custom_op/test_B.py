import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

mod = SourceModule(""" 

	__global__ void UnrollBKernel(float *in, float *out) {
		int cx = blockIdx.x;
		int cy = blockIdx.y;
		int z = blockIdx.z;

		for(int k = -2; k < 3; k++){
			for(int l = 0; l < 12; l++){
				for(int i = -2; i < 3; i++){
					for(int j = -2; j < 3; j++){
						int x = k + cx + i;
						int y = l + j;

						int gidx = y + (x * 12) + (z * 12 * 37);
						int idx = (j + 2) + ((i + 2) * 5) + ((((k+2)*12)+l) * 5 * 5) + (cy * 60 * 5 * 5) + (cx * 12 * 60 * 5 * 5) + (z * 37 * 12 * 60 * 5 * 5);

						if(x < 0 || y < 0 || x > 36 || y > 11){
							out[idx] = 0;
						}
						else{
							out[idx] = in[gidx];
						}
					}
				}
			}
		}
	}

""")

b = np.random.randn(25,37,12).astype(np.float32)
out = np.random.randn(25,37,12,60,5,5).astype(np.float32)
b_gpu = cuda.mem_alloc(b.nbytes)
out_gpu = cuda.mem_alloc(out.nbytes)
cuda.memcpy_htod(b_gpu, b)

start = time.time()
func = mod.get_function("UnrollBKernel")
func(b_gpu, out_gpu,grid=(37,12,25),block=(1,1,1))
end = time.time()


a_doubled = np.empty_like(out)
cuda.memcpy_dtoh(a_doubled, out_gpu)
x = 0

# for i in a_doubled[0,0,0]:
# 	print(i)
# 	print(x)
# 	x = x + 1

# x = 0

# for i in a_doubled[0,1,0]:
# 	print(i)
# 	print(x)
# 	x = x + 1

print("Execution time: " + str(end - start) + " Seconds")
