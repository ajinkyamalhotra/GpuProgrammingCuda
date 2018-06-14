
#include <wb.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#define TILE_WIDTH = 16;
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


//Matrix multiplication kernel
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
	
	//Y-axis Thread Dimension
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	//X-axis Thread Dimension
	int columns = blockIdx.x*blockDim.x + threadIdx.x;

	// Initilaizing final value to add in the output matrix
	float pValue = 0;

	//checking for boundary condition 
	if (row < numARows  && columns < numBColumns) {
		
		//adding values from 0 to matrix A width and from 0 to martrix B Height
		for (int k = 0; k < numAColumns; k++) {
		
			//Summation of the rows from matrix A and columns from matrix B to pValue
			pValue += A[row*numAColumns + k] * B[k*numBColumns + columns];
		
		}
		
		//add the final value to the output matrix
		C[row*numCColumns + columns] = pValue;
	
	}

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
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  
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
  
  //@@ Allocating GPU memory here
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
  //@@ width = numBColumns, Height = numARows
  dim3 DimGrid((numBColumns/16)+1, (numARows/16)+1, 1);
  dim3 DimBlock(16, 16, 1);

  wbTime_start(Compute, "Performing CUDA computation");
	
  //@@ Launching the GPU Kernel 
  matrixMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
	
  //@@ Copying the GPU memory back to the CPU
  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
	
  //@@ Freeing the GPU memory
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
