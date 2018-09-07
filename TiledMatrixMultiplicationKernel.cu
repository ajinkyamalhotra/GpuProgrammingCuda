
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
		
		//@@making sure that all threads in a block are done with loading data from global memory to shared memory
		//@@before proceeding to the calculations phase
		__syncthreads();
		
		//@@Checking boundary condition and multiplication values from matrix ds_A and ds_B and adding to pValue
		if (row < numARows  && col < numBColumns) 
			for (int i = 0; i < TILE_WIDTH; i++) pValue += ds_A[ty][i] * ds_B[i][tx];

		//@@making sure that all threads in a block are done with loading data from global memory to shared memory
		//@@before proceeding to the calculations phase
		__syncthreads();
	}
	
	//@@ checking boundary condition
	if (row < numARows  && col < numBColumns)
		//@@ writing the computed value to matrix C in global memory
		C[row*numCColumns + col] = pValue;
	
	pValue = 0;
	
}
