#include <stdio.h>
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void UnrollAKernel(const float* in, float* out) {
	int cx = (blockIdx.x / 12); 
	int cy = blockIdx.x % 12;
	int x = cx + (threadIdx.x - 2);
	int y = cy + (threadIdx.y - 2);
	int z = threadIdx.z;

	int idx = threadIdx.y + (threadIdx.x * 5) + (blockIdx.y * 5 * 5) + (blockIdx.x * 60 * 5 * 5) + (threadIdx.z * 444 * 60 * 5 * 5);
	int gidx = (z * 37 * 12) + (x * 12) + y;

	if(x < 0 || y < 0 || x >= 37 || y >= 12){
		out[idx] = 0;
	}
	else{
		out[idx] = in[gidx];
	}
}

__global__ void UnrollBKernel(const float *in, float *out) {
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

__global__ void mean(float *in, float *out){
		
		int idx = blockIdx.y + (blockIdx.x * 60) + (threadIdx.z * 444 * 60);
		out[idx] = 0;
		for(int x = 0; x < 25; x++){
			int gidx = x + (blockIdx.y * 25) + (blockIdx.x * 60 * 25) + (threadIdx.z * 444 * 60 * 25);
			out[idx] += (in[gidx]);
		}
		out[idx] = out[idx] / 25;
	}

__global__ void stddev(float *in, float *means, float *out){
	int idx = blockIdx.y + (blockIdx.x * 60) + (threadIdx.z * 444 * 60);
	out[idx] = 0;
	for(int x = 0; x < 25; x++){
		int gidx = x + (blockIdx.y * 25) + (blockIdx.x * 60 * 25) + (threadIdx.z * 444 * 60 * 25);
		out[idx] += (in[gidx] - means[idx]) * (in[gidx] - means[idx]);
	}
	out[idx] /= (24);
	out[idx] = sqrt(out[idx]);
}

__global__ void corr(const float *in1, const float *in2, float *ma, float *sda, float *mb, float *sdb, float *out){
	int idx = blockIdx.y + (blockIdx.x * 60) + (threadIdx.z * 444 * 60);
	out[idx] = 0;
	for(int x = 0; x < 25; x++){
		int gidx = x + (blockIdx.y * 25) + (blockIdx.x * 60 * 25) + (threadIdx.z * 444 * 60 * 25);
		out[idx] += (in1[gidx] - ma[idx]) * (in2[gidx] - mb[idx]);
	}

	out[idx] /= ((24) * sdb[idx] * sda[idx]);
}

void NormXCorrKernelLauncher(const float* in1, const float* in2, float* out) {
	dim3 grid1(444,60);
	dim3 block1(5,5,25);
	float* out1 = 0;
	float* out2 = 0;
	float* mean1 = 0;
	float* mean2 = 0;
	float* sd1 = 0;
	float* sd2 = 0;

	int numB1 = 25*444*60*5*5*sizeof(float);
	int numB2 = 25*444*60*sizeof(float);
	int numB3 = 25*492*5*5*sizeof(float);

	cudaMalloc((void**)&out1, numB1);
	cudaMalloc((void**)&out2, numB1);
	cudaMalloc((void**)&mean1, numB2);
	cudaMalloc((void**)&mean2, numB2);
	cudaMalloc((void**)&sd1, numB2);
	cudaMalloc((void**)&sd2, numB2);


	UnrollAKernel<<<grid1, block1>>>(in1, out1);

	dim3 grid2(37,12,25);
	UnrollBKernel<<<grid2, 1>>>(in2, out2);
	dim3 grid3(492,1,1);
	// stencil_2db<<<grid3, block1>>>(in2, intrmdte);
	// stencil_2dc<<<grid1, block1>>>(intrmdte, out2);

	dim3 block2(1,1,25);
	mean<<<grid1, block2>>>(out1, mean1);
	mean<<<grid1, block2>>>(out2, mean2);
	stddev<<<grid1, block2>>>(out1, mean1, sd1);
	stddev<<<grid1, block2>>>(out2, mean2, sd2);

	corr<<<grid1, block2>>>(out1, out2, mean1, sd1, mean2, sd2, out);

	cudaFree(out1);
	cudaFree(out2);
	cudaFree(mean1);
	cudaFree(mean2);
	cudaFree(sd1);
	cudaFree(sd2);
}

#endif