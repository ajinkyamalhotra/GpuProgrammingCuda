#include <wb.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Math.h>
#define NUM_BINS 4096
#define BLOCK_SIZE 512 
#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

__global__ void histogram(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins) {

	//@@ Using privitization technique
	__shared__ unsigned int hist[NUM_BINS];
	
	int numOfElementsPerThread = NUM_BINS / BLOCK_SIZE;
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	for (int j = 0; j < numOfElementsPerThread; ++j)  
		hist[threadIdx.x + blockDim.x*j] = 0;
	
	__syncthreads();
	
	if (i < num_elements)
		atomicAdd(&hist[input[i]], 1);
	__syncthreads();
	
	for (int k = 0; k < numOfElementsPerThread; ++k) 
		atomicAdd(&bins[threadIdx.x + blockDim.x*k], hist[threadIdx.x+blockDim.x*k]);
}

__global__ void saturate(unsigned int *bins, unsigned int num_bins) {
	
	//@@If the bin value is more than 127, make it equal to 127
	for (int i = 0; i < NUM_BINS / BLOCK_SIZE; ++i)
		
		if (bins[threadIdx.x + blockDim.x*i] >= 128)
		
			bins[threadIdx.x + blockDim.x*i]  = 127;
}
