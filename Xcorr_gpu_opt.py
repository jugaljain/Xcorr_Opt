import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import numpy
import curses
#stdscr = curses.initscr()

NUM_MAPS = 25
NUM_ROWS = 37
NUM_COLS = 12
WIN_SIZE = 5
NUM_NEIGH = 60
NUM_PAD = 492
NUM_PIX = 444
WIN_PIX = 25

mod = SourceModule("""
	#include <math.h>

	#define NUM_MAPS 25
	#define NUM_ROWS 37
	#define NUM_COLS 12
	#define WIN_SIZE 5
	#define NUM_NEIGH 60
	#define WIN_PIX 25
	#define NUM_PIX 444
	#define NUM_PAD 492

	__global__ void stencil_2db(float *in, float *out) {
		int cx = (blockIdx.x / NUM_COLS) - 2; 
		int cy = blockIdx.x % NUM_COLS;
		int x = cx + (threadIdx.x - 2);
		int y = cy + (threadIdx.y - 2);
		int z = threadIdx.z;

		int idx = threadIdx.y + (threadIdx.x * WIN_SIZE) + (blockIdx.x * WIN_PIX) + (threadIdx.z * NUM_PAD * WIN_PIX);
		int gidx = (z * NUM_ROWS * NUM_COLS) + (x * NUM_COLS) + y;

		if(x < 0 || y < 0 || x >= NUM_ROWS || y >= NUM_COLS){
			out[idx] = 0;
		}
		else{
			out[idx] = in[gidx];
		}
	}

	__global__ void stencil_2da(float *in, float *out){
		int cx = (blockIdx.x / NUM_COLS); 
		int cy = blockIdx.x % NUM_ROWS;
		int x = cx + (threadIdx.x - 2);
		int y = cy + (threadIdx.y - 2);
		int z = threadIdx.z;

		int idx = threadIdx.y + (threadIdx.x * WIN_SIZE) + (blockIdx.y * WIN_PIX) + (blockIdx.x * NUM_NEIGH * WIN_PIX) + (threadIdx.z * NUM_PIX * WIN_PIX * NUM_NEIGH);
		int gidx = (z * NUM_ROWS * NUM_COLS) + (x * NUM_COLS) + y;

		if(x < 0 || y < 0 || x >= NUM_ROWS || y >= NUM_COLS){
			out[idx] = 0;
		}
		else{
			out[idx] = in[gidx];
		}
	}

	__global__ void stencil_2dc(float *in, float *out){
		int r = blockIdx.x / NUM_COLS;
		int n = (r * NUM_COLS) + blockIdx.y;
		int z = threadIdx.z;
		int x = threadIdx.x;
		int y = threadIdx.y;
		
		int idx = threadIdx.y + (threadIdx.x * WIN_SIZE) + (blockIdx.y * WIN_PIX) + (blockIdx.x * NUM_NEIGH * WIN_PIX) + (threadIdx.z * NUM_PIX * NUM_NEIGH * WIN_PIX);
		int gidx = y + (x * WIN_SIZE) + (n * WIN_PIX) + (z * NUM_PAD * WIN_SIZE);

		out[idx] = in[gidx];

	}

	__global__ void mean(float *in, float *out){
		
		int idx = blockIdx.y + (blockIdx.x * NUM_NEIGH) + (threadIdx.z * NUM_PIX * NUM_NEIGH);
		for(int x = 0; x < WIN_PIX; x++){
			int gidx = x + (blockIdx.y * WIN_PIX) + (blockIdx.x * NUM_NEIGH * WIN_PIX) + (threadIdx.z * NUM_PIX * NUM_NEIGH * WIN_PIX);
			out[idx] += (in[gidx]);
		}
		out[idx] = out[idx] / WIN_PIX;
	}

	__global__ void stddev(float *in, float *means, float *out){
		int idx = blockIdx.y + (blockIdx.x * NUM_NEIGH) + (threadIdx.z * NUM_PIX * NUM_NEIGH);
		for(int x = 0; x < WIN_PIX; x++){
			int gidx = x + (blockIdx.y * WIN_PIX) + (blockIdx.x * NUM_NEIGH * WIN_PIX) + (threadIdx.z * NUM_PIX * NUM_NEIGH * WIN_PIX);
			out[idx] += (in[gidx] - means[idx]) * (in[gidx] - means[idx]);
		}
		out[idx] = out[idx] / (WIN_PIX - 1);
		out[idx] = sqrt(out[idx]);
	}

	__global__ void corr(float *in1, float *in2, float *ma, float *sda, float *mb, float *sdb, float *out){
		int idx = blockIdx.y + (blockIdx.x * NUM_NEIGH) + (threadIdx.z * NUM_PIX * NUM_NEIGH);
		for(int x = 0; x < WIN_PIX; x++){
			int gidx = x + (blockIdx.y * WIN_PIX) + (blockIdx.x * NUM_NEIGH * WIN_PIX) + (threadIdx.z * NUM_PIX * NUM_NEIGH * WIN_PIX);
			out[idx] = (in1[gidx] - ma[idx]) * (in2[gidx] - mb[idx]) / ((WIN_PIX - 1) * sdb[idx] * sda[idx]);
		}
	}
""")

f = open("test/randomA.txt", "r")
a = []
l = f.readline()
while l:
	a.append(float(l.strip("\n")))
	l = f.readline()

a = numpy.asarray(a)
a = numpy.reshape(a, (25,37,12)).astype(numpy.float32)
f.close()

f = open("test/randomB.txt", "r")
b = []
l = f.readline()
while l:
	b.append(float(l.strip("\n")))
	l = f.readline()

b = numpy.asarray(b)
b = numpy.reshape(b, (25,37,12)).astype(numpy.float32)
f.close()


start = time.time()

# a = numpy.random.randn(25,37,12).astype(numpy.float32)
# b = numpy.random.randn(25,37,12).astype(numpy.float32)
c = numpy.random.randn(NUM_MAPS,NUM_PAD,WIN_SIZE,WIN_SIZE).astype(numpy.float32)  #unrolled second
ma = numpy.zeros((NUM_MAPS,NUM_PIX,NUM_NEIGH), dtype=numpy.float32)		#mean values
mb = numpy.zeros((NUM_MAPS,NUM_PIX,NUM_NEIGH), dtype=numpy.float32)		#mean values
sda = numpy.zeros((NUM_MAPS,NUM_PIX,NUM_NEIGH), dtype=numpy.float32)		#std dev values
sdb = numpy.zeros((NUM_MAPS,NUM_PIX,NUM_NEIGH), dtype=numpy.float32)		#std dev values
aint = numpy.random.randn(NUM_MAPS,NUM_PIX,NUM_NEIGH,WIN_SIZE,WIN_SIZE).astype(numpy.float32)	#unrolled first
caint = numpy.random.randn(NUM_MAPS,NUM_PIX,NUM_NEIGH,WIN_SIZE,WIN_SIZE).astype(numpy.float32)	#further unrolled second
final = numpy.random.randn(NUM_MAPS,NUM_PIX,NUM_NEIGH).astype(numpy.float32)


a_gpu = cuda.mem_alloc(a.nbytes)
print(a_gpu)		
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)
ma_gpu = cuda.mem_alloc(ma.nbytes)
mb_gpu = cuda.mem_alloc(mb.nbytes)
sda_gpu = cuda.mem_alloc(sda.nbytes)
sdb_gpu = cuda.mem_alloc(sdb.nbytes)
aint_gpu = cuda.mem_alloc(aint.nbytes)
caint_gpu = cuda.mem_alloc(caint.nbytes)
final_gpu = cuda.mem_alloc(final.nbytes)


cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)
cuda.memcpy_htod(c_gpu, c)
cuda.memcpy_htod(ma_gpu, ma)
cuda.memcpy_htod(mb_gpu, mb)
cuda.memcpy_htod(sda_gpu, sda)
cuda.memcpy_htod(sdb_gpu, sdb)
cuda.memcpy_htod(aint_gpu, aint)
cuda.memcpy_htod(caint_gpu, caint)
cuda.memcpy_htod(final_gpu, final)


func = mod.get_function("stencil_2db")
func(b_gpu, c_gpu,grid=(NUM_PAD,1,1),block=(WIN_SIZE,WIN_SIZE,NUM_MAPS))
func2 = mod.get_function("stencil_2da")
func2(a_gpu, aint_gpu, grid=(NUM_PIX,NUM_NEIGH,1), block=(WIN_SIZE,WIN_SIZE,NUM_MAPS))
func3 = mod.get_function("stencil_2dc")
func3(c_gpu, caint_gpu, grid=(NUM_PIX,NUM_NEIGH,1), block=(WIN_SIZE,WIN_SIZE,NUM_MAPS))
func4 = mod.get_function("mean")

func4(caint_gpu, mb_gpu, grid=(NUM_PIX,NUM_NEIGH,1), block=(1,1,NUM_MAPS))
func4(aint_gpu, ma_gpu, grid=(NUM_PIX,NUM_NEIGH,1), block=(1,1,NUM_MAPS))

func5 = mod.get_function("stddev")
func5(caint_gpu, mb_gpu, sdb_gpu, grid=(NUM_PIX,NUM_NEIGH,1), block=(1,1,NUM_MAPS))
func5(aint_gpu, ma_gpu, sda_gpu, grid=(NUM_PIX,NUM_NEIGH,1), block=(1,1,NUM_MAPS))

func6 = mod.get_function("corr")
func6(aint_gpu, caint_gpu, ma_gpu, sda_gpu, mb_gpu, sdb_gpu, final_gpu, grid=(NUM_PIX,NUM_NEIGH,1), block=(1,1,NUM_MAPS))
end = time.time()

a_doubled = numpy.empty_like(final)
cuda.memcpy_dtoh(a_doubled, final_gpu)
i = 0
print(a_doubled[0,0])
print(a[0])
print("Execution time: " + str(end - start) + " Seconds")

	

# try:
# 	for x in a_doubled[0,0]:
# 		stdscr.addstr(0, 0, "{0}".format(x))
# 		stdscr.addstr(5, 0, "{0}".format(i))
# 		stdscr.refresh()
# 		time.sleep(1)

# 		#print(i, end='\r')
# 		i += 1
# finally:
# 	curses.echo()
# 	curses.nocbreak()
# 	curses.endwin()
# for x in a_doubled[1,0]:
# 	print(x)
# 	print(i)
# 	i += 1
# print(a_doubled.shape)
# print(b[0])
# print(b.shape)

