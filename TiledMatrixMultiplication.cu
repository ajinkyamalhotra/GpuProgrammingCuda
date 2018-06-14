
#include <wb.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#define TILE_WIDTH 4
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@Tiled matrix multiplication kernel
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
 	
	//@@declaring the shared memory
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
	
	//@@X-axis and Y-axis block id
	int bx = blockIdx.x; int by = blockIdx.y;
	
	//@@X-axis and Y-axis thread id
	int tx = threadIdx.x; int ty = threadIdx.y;
	
	//@@Y-axis matrix dimension
	int row = by * blockDim.y + ty;
	
	//@@X-axis matrix dimension
	int col = bx * blockDim.x + tx;
	
	//@@Initilaizing final value to add in the output matrix
	float pValue = 0;
	
	//@@initializing shared memory
	for (int p = 0; p < numAColumns / TILE_WIDTH + 1 ; p++) {
		
		//@@checking boundary conditions
		//@@loading the input data from global memory matrix A to shared memory matrix ds_A.
		if (row < numARows && p*TILE_WIDTH + tx < numAColumns) 
			ds_A[ty][tx] = A[row*numAColumns + p*TILE_WIDTH + tx];
		else ds_A[ty][tx] = 0.0;
		
		//@@checking boundary conditions
		//@@loading the input data from global memory matrix B to shared memory matrix ds_B.
		if (p*TILE_WIDTH + ty < numBRows && col < numBColumns) 
			ds_B[ty][tx] = B[(p*TILE_WIDTH + ty)*numBColumns + col];
		else ds_B[ty][tx] = 0.0;
		
		//@@making sure that all threads in a block are done with the loading data into shared memory
		//@@before proceeding to the calculations phase
		__syncthreads();
		
		//@@Checking boundary condition and multiplication values from matrix ds_A and ds_B and adding to pValue
		if (row < numARows  && col < numBColumns) 
			for (int i = 0; i < TILE_WIDTH; i++) pValue += ds_A[ty][i] * ds_B[i][tx];

		//@@making sure that all threads in a block are done with the loading data into shared memory
		//@@before proceeding to the calculations phase
		__syncthreads();
	}
	
	//@@ checking boundary condition
	if (row < numARows  && col < numBColumns)
		//@@ writing the computed value to matrix C in global memory
		C[row*numCColumns + col] = pValue;
	
	pValue = 0;
	
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  
  hostC = NULL;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  
  //@@ Setting numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;
  
  //@@ Allocating the hostC matrix
  int sizeC = (numCRows * numCColumns) * sizeof(float);
  hostC = (float *)malloc(sizeC);

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  
  //@@ Allocating GPU memory
  int sizeA = (numARows * numAColumns) * sizeof(float);
  int sizeB = (numBRows * numBColumns) * sizeof(float);
  cudaMalloc((void **) &deviceA, sizeA);
  cudaMalloc((void **) &deviceB, sizeB);
  cudaMalloc((void **) &deviceC, sizeC);
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
	
  //@@ Copying memory to the GPU
  cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initializing the grid and block dimensions
  int width = numBColumns;
  int height = numARows;
  int block_size = TILE_WIDTH;
  dim3 DimGrid((width/block_size)+1, (height/block_size)+1, 1);
  dim3 DimBlock(block_size, block_size, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  
  //@@ Launching the GPU Kernel
  matrixMultiplyShared<<<DimGrid, DimBlock>>> (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");
  wbTime_start(Copy, "Copying output memory to the CPU");
	
  //@@ Copying the GPU memory back to the CPU
  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  
  //@@ Freeing the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
	
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
