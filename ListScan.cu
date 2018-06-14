#include <wb.h>

#include <cuda.h>

#include <cuda_runtime.h>

#include <device_launch_parameters.h>

#include <Math.h>

#define BLOCK_SIZE 512

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float *aux, int len) {
    
	//@@ Modify the body of this kernel to generate the scanned blocks
    //@@ Make sure to use the workefficient version of the parallel scan
    //@@ Also make sure to store the block sum to the aux array 
	__shared__ float XY[2 * BLOCK_SIZE];

	int bx = blockIdx.x; 
	
	int tx = threadIdx.x;

	int i = 2 * bx * blockDim.x + tx;
	
	if (i<len) XY[tx] = input[i];
	
	if (i + blockDim.x<len) XY[tx + blockDim.x] = input[i + blockDim.x];
	__syncthreads();

	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2){
		__syncthreads();
	
		int index = (tx + 1)*stride * 2 - 1;
		
		if (index < 2 * BLOCK_SIZE) XY[index] += XY[index - stride];
	}

	for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		__syncthreads();
		
		int index = (tx + 1)*stride * 2 - 1;
		
		if (index + stride < 2 * BLOCK_SIZE) XY[index + stride] += XY[index];
	}
	
	__syncthreads();
	if (i < len) output[i] = XY[tx];
	
	if (i + blockDim.x < len) output[i + blockDim.x] = XY[tx + blockDim.x];

	if (aux != NULL && tx == 0) aux[bx] = XY[2 * blockDim.x - 1];
}

__global__ void addScannedBlockSums(float *input, float *aux, int len) {
	
	//@@ Modify the body of this kernel to add scanned block sums to
	int tx = threadIdx.x;
	
	int bx = blockIdx.x;

	int dx = blockDim.x;
	
	int i = 2 * bx * dx + tx;
	
	if (bx > 0) {
	
		if (i < len) aux[i] += input[bx-1];
		
		if (i + dx < len) aux[i + dx] += input[blockIdx.x - 1];
	}
}

int main(int argc, char **argv) {
  
  wbArg_t args;
  
  float *hostInput;  // The input 1D list
  
  float *hostOutput; // The output 1D list
  
  float *deviceInput;
  
  float *deviceOutput;
  
  float *deviceAuxArray, *deviceAuxScannedArray;
  
  int numElements; // number of elements in the input/output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  
  hostOutput = (float *)malloc(numElements * sizeof(float));
  
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ", numElements);

  wbTime_start(GPU, "Allocating device memory.");
  
  //@@ Allocate device memory
  //allocating input and output memory
  int I_O_size = (numElements) * sizeof(float);
  
  cudaMalloc((void **) &deviceInput, I_O_size);
  
  cudaMalloc((void **) &deviceOutput, I_O_size);

  //allcating memory for auxillary arrays
  int auxArraySize = 2*BLOCK_SIZE * sizeof(float);
  
  cudaMalloc((void **) &deviceAuxArray, auxArraySize);
  
  cudaMalloc((void **) &deviceAuxScannedArray, auxArraySize);

  wbTime_stop(GPU, "Allocating device memory.");

  wbTime_start(GPU, "Clearing output device memory.");
  
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  
  wbTime_stop(GPU, "Clearing output device memory.");

  wbTime_start(GPU, "Copying input host memory to device.");
  
  //@@ Copy input host memory to device	
  cudaMemcpy(deviceInput, hostInput, I_O_size, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input host memory to device.");
  
  //@@ Initialize the grid and block dimensions here 
  int gridSize = (numElements-1 / BLOCK_SIZE)+1;
  
  dim3 DimGrid1(gridSize, 1, 1);
  
  dim3 DimGrid2(1, 1, 1);
  
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");

  //@@ Modify this to complete the functionality of the scan on the deivce. You need to launch scan kernel twice: 
  //@@ 1) for generating scanned blocks (hint: pass deviceAuxArray to the aux parameter)
  //@@ 2) for generating scanned aux array that has the scanned block sums. (hint: pass NULL to the aux parameter)
  //@@ Then you should call addScannedBlockSums kernel.
  scan << <DimGrid1, DimBlock >> > (deviceInput, deviceOutput, deviceAuxArray, numElements);
  cudaDeviceSynchronize();
  
  scan << <DimGrid2, DimBlock >> > (deviceAuxArray, deviceAuxScannedArray, NULL, 2*BLOCK_SIZE);
  cudaDeviceSynchronize();
  
  addScannedBlockSums << <DimGrid1, DimBlock >> > (deviceAuxScannedArray, deviceOutput, numElements);
  cudaDeviceSynchronize();
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output device memory to host");
  
  //@@ Copy results from device to host	
  cudaMemcpy(hostOutput, deviceOutput, I_O_size, cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying output device memory to host");

  wbTime_start(GPU, "Freeing device memory");
  
  //@@ Deallocate device memory
  cudaFree(deviceAuxScannedArray);
  
  cudaFree(deviceAuxArray);
  
  cudaFree(deviceOutput);
  
  cudaFree(deviceInput);
  
  wbTime_stop(GPU, "Freeing device memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  
  free(hostOutput);

  return 0;
}
