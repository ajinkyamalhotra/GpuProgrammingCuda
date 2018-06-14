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

	//@@ Make sure to use the privitization technique
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
	
	//@@ Write the kernel that applies saturtion to counters (i.e., if the bin value is more than 127, make it equal to 127)
	for (int i = 0; i < NUM_BINS / BLOCK_SIZE; ++i)
		
		if (bins[threadIdx.x + blockDim.x*i] >= 128)
		
			bins[threadIdx.x + blockDim.x*i]  = 127;
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  
  int inputLength;
  
  unsigned int *hostInput;
  
  unsigned int *hostBins;
  
  unsigned int *deviceInput;
  
  unsigned int *deviceBins;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0), &inputLength, "Integer");
  
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  wbTime_start(GPU, "Allocating device memory");

  //@@ Allocate device memory here
  int size = inputLength * sizeof(float);
  
  int binSize = NUM_BINS * sizeof(float);
  
  cudaMalloc((void **) &deviceInput, size);
  
  cudaMalloc((void **) &deviceBins, binSize);

  CUDA_CHECK(cudaDeviceSynchronize());
  
  wbTime_stop(GPU, "Allocating device memory");

  wbTime_start(GPU, "Copying input host memory to device");
  
  //@@ Copy input host memory to device
  cudaMemcpy(deviceInput, hostInput, size, cudaMemcpyHostToDevice);
  
  CUDA_CHECK(cudaDeviceSynchronize());
  
  wbTime_stop(GPU, "Copying input host memory to device");
	
  wbTime_start(GPU, "Clearing the bins on device");
  
  //@@ zero out the deviceBins using cudaMemset() 
  //@@ initialize all the values in deviceBins array to "0".
  cudaMemset(deviceBins, 0, binSize);

  wbTime_stop(GPU, "Clearing the bins on device");
  
  //@@ Initialize the grid and block dimensions here
  int gridSize = (inputLength-1 / BLOCK_SIZE) + 1;
  
  dim3 DimGridHistogram(gridSize, 1, 1);
  
  dim3 DimGridSaturate(1, 1, 1);
  
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  wbLog(TRACE, "Launching kernel");
  
  wbTime_start(Compute, "Performing CUDA computation");
  
  //@@ Invoke kernels: first call histogram kernel and then call saturate kernel
  histogram << <DimGridHistogram, DimBlock >> > (deviceInput, deviceBins, inputLength, NUM_BINS);
  
  saturate << <DimGridSaturate, DimBlock >> > (deviceBins, NUM_BINS);

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output device memory to host");
  
  //@@ Copy output device memory to host
  cudaMemcpy(hostBins, deviceBins, binSize, cudaMemcpyDeviceToHost);

  CUDA_CHECK(cudaDeviceSynchronize());
  
  wbTime_stop(Copy, "Copying output device memory to host");

  wbTime_start(GPU, "Freeing device memory");
  
  //@@ Free the device memory here
  cudaFree(deviceInput);
  
  cudaFree(deviceBins);
 
  wbTime_stop(GPU, "Freeing device memory");

  wbSolution(args, hostBins, NUM_BINS);

  free(hostBins);
  
  free(hostInput);
  
  return 0;
}
