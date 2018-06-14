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

//@@ storing the sum of the elements in the aux array step 1
__global__ void scan(float *input, float *output, float *aux, int len) {
 
	//@@declaring shared memeory of size 2*inputSize
    	__shared__ float XY[2 * BLOCK_SIZE];
	
	//@@X-axis block id
	int bx = blockIdx.x; 
	
	//@@X-axis thread id
	int tx = threadIdx.x;

	int i = 2 * bx * blockDim.x + tx;
	
	//@@ loading data from global memory to shared memory stage 1
	if (i<len)
		XY[tx] = input[i];
	
	//@@ loading data from global memory to shared memory stage 2
	if (i + blockDim.x<len)
		XY[tx + blockDim.x] = input[i + blockDim.x];
	
	//@@making sure that all threads in a block are done with loading data from global memory to shared memory
	//@@before proceeding to the calculations phase
	__syncthreads();

	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2){
		//@@making sure that all threads in a block are done with previous step before starting the next
		__syncthreads();
	
		int index = (tx + 1)*stride * 2 - 1;
		
		if (index < 2 * BLOCK_SIZE) 
			XY[index] += XY[index - stride];
	}

	for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		//@@making sure that all threads in a block are done with previous step before starting the next
		__syncthreads();
		
		int index = (tx + 1)*stride * 2 - 1;
		
		if (index + stride < 2 * BLOCK_SIZE)
			XY[index + stride] += XY[index];
	}
	
	//@@making sure that all threads in a block are done with previous step before starting the next
	__syncthreads();
	
	if (i < len)
		output[i] = XY[tx];
	
	if (i + blockDim.x < len)
		output[i + blockDim.x] = XY[tx + blockDim.x];

	//@@storing the block sum to the aux array 
	if (aux != NULL && tx == 0)
		aux[bx] = XY[2 * blockDim.x - 1];
}

//@@adding the sums stored in aux array to get the final values
__global__ void addScannedBlockSums(float *input, float *aux, int len) {
	
	int tx = threadIdx.x;
	
	int bx = blockIdx.x;

	int dx = blockDim.x;
	
	int i = 2 * bx * dx + tx;
	
	if (bx > 0) {
	
		if (i < len) 
			aux[i] += input[bx-1];
		
		if (i + dx < len) 
			aux[i + dx] += input[blockIdx.x - 1];
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

  //@@allocating input and output memory
  int I_O_size = (numElements) * sizeof(float);
  
  cudaMalloc((void **) &deviceInput, I_O_size);
  
  cudaMalloc((void **) &deviceOutput, I_O_size);

  //@@allocating memory for auxillary arrays
  int auxArraySize = 2*BLOCK_SIZE * sizeof(float);
  
  cudaMalloc((void **) &deviceAuxArray, auxArraySize);
  
  cudaMalloc((void **) &deviceAuxScannedArray, auxArraySize);

  wbTime_stop(GPU, "Allocating device memory.");

  wbTime_start(GPU, "Clearing output device memory.");
  
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  
  wbTime_stop(GPU, "Clearing output device memory.");

  wbTime_start(GPU, "Copying input host memory to device.");
  
  //@@ Copying input host memory to device	
  cudaMemcpy(deviceInput, hostInput, I_O_size, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input host memory to device.");
  
  //@@ Initializing the grid and block dimensions here 
  int gridSize = (numElements-1 / BLOCK_SIZE)+1;
  
  dim3 DimGrid1(gridSize, 1, 1);
  
  dim3 DimGrid2(1, 1, 1);
  
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");

  //@@ for generating scanned blocks
  scan << <DimGrid1, DimBlock >> > (deviceInput, deviceOutput, deviceAuxArray, numElements);
  cudaDeviceSynchronize();
  
  //@@ for generating scanned aux array
  scan << <DimGrid2, DimBlock >> > (deviceAuxArray, deviceAuxScannedArray, NULL, 2*BLOCK_SIZE);
  cudaDeviceSynchronize();
  
  //@@ for adding the scanned aux array
  addScannedBlockSums << <DimGrid1, DimBlock >> > (deviceAuxScannedArray, deviceOutput, numElements);
  cudaDeviceSynchronize();
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output device memory to host");
  
  //@@ Copying results from device to host	
  cudaMemcpy(hostOutput, deviceOutput, I_O_size, cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying output device memory to host");

  wbTime_start(GPU, "Freeing device memory");
  
  //@@ Freeing device memory
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
