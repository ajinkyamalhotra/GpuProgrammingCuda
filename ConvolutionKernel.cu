#include <wb.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Math.h>

#define MASK_WIDTH 5
#define O_TILE_WIDTH 16
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH-1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@tiled 2D convolution kernel
__global__ void convolutional2D(float* in, const float* __restrict__ mask, float* out, int maskWidth, int width, int height, int channels) {
	
	//@@ declaring shared variable for the input
	__shared__ float in_S[BLOCK_WIDTH][BLOCK_WIDTH];

	int tx = threadIdx.x; int ty = threadIdx.y;
	
	//@@If BlockDim is used we will get ghost elements as input
	int row_o = blockIdx.y * O_TILE_WIDTH + ty;
	int col_o = blockIdx.x * O_TILE_WIDTH + tx;

	//@@the start point calculations on the actual input
	int row_i = row_o - (MASK_WIDTH / 2);
	int col_i = col_o - (MASK_WIDTH / 2);

	//@@We have 3 channels, one thread will process 3 values//
	for (int k = 0; k < channels; ++k) {

		//@@loading the input data from global memory to shared memory.
		if (row_i > -1 && row_i < height && col_i > -1 && col_i < width)
			in_S[ty][tx] = in[(row_i *width + col_i) * channels + k];
		
		else
			in_S[ty][tx] = 0.0f;

		//@@making sure all the threads in a block are done with data loading phase
		//@@before proceeding to the calculation step
		__syncthreads();

		float pixVal = 0.0f;
		
		//@@checking for out of bound error
		if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
			int i = 0; int j = 0;
			for (i = 0; i < MASK_WIDTH; ++i) {
				for (j = 0; j < MASK_WIDTH; ++j) {
					int currRow = row_i + i;
					int currCol = col_i + j;
					if (currRow > -1 && currRow < height && currCol > -1 && currCol < width) {
						pixVal += in_S[i + ty][j + tx] * mask[i*MASK_WIDTH + j];
					}
				}
			}
			if (col_o < width && row_o < height) {
				out[(row_o*width + col_o) * channels + k] = clamp(pixVal);
			}
		}
	}
}
