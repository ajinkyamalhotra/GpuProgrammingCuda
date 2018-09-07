#include <wb.h>
#include <wb.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

//@@ Vector Addition kernel
__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
	
	int i = threadIdx.x + (blockDim.x * blockIdx.x);
	
	//@@checking boundary condition and adding vectors
	if (i < len)
		out[i] = in1[i] + in2[i];
}

//@@ Host Code
int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
	
  //@@ Allocating GPU memory
  int size = inputLength * sizeof(float);
  cudaMalloc((void **) &deviceInput1, size);
  cudaMalloc((void **) &deviceInput2, size);
  cudaMalloc((void **) &deviceOutput, size);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  
  //@@ Copying memory to the GPU
  cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");
	
  //@@ Initializing the grid and block dimensions
  dim3 DimGrid(inputLength/256 + 1, 1, 1);
  dim3 DimBlock(256, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
	
  //@@ Launching the GPU Kernel
  vecAdd << <DimGrid, DimBlock >> > (deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
	
  //@@ Copying the GPU memory back to the CPU
  cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  
  //@@ Freeing the GPU memory
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
