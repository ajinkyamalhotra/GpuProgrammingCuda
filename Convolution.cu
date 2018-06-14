#include <wb.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Math.h>

#define MASK_WIDTH 5
#define O_TILE_WIDTH 16
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH-1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE 
//implement the tiled 2D convolution kernel with adjustments for channels
__global__ void convolutional2D(float* in, const float* __restrict__ mask, float* out, int maskWidth, int width, int height, int channels) {
	
	//@@ declaring shared variable for the input
	__shared__ float in_S[BLOCK_WIDTH][BLOCK_WIDTH];

	int tx = threadIdx.x; int ty = threadIdx.y;
	//@@Since if BlockDim is used we will get ghost elements as input
	int row_o = blockIdx.y * O_TILE_WIDTH + ty;
	int col_o = blockIdx.x * O_TILE_WIDTH + tx;

	//@@the start point calculations on the actual input
	int row_i = row_o - (MASK_WIDTH / 2);
	int col_i = col_o - (MASK_WIDTH / 2);

	//@@Since we have 3 channels, one thread will process 3 values//
	for (int k = 0; k < channels; ++k) {

		//@@loading the input data from global memory to shared memory.
		if (row_i > -1 && row_i < height && col_i > -1 && col_i < width)
			in_S[ty][tx] = in[(row_i *width + col_i) * channels + k];
		else in_S[ty][tx] = 0.0f;

		//@@making sure all the threads in a block have arrived with the input data.
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

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile  = wbArg_getInputFile(arg, 1);

  inputImage   = wbImport(inputImageFile);
  hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == MASK_WIDTH);    /* mask height is fixed to 5 */
  assert(maskColumns == MASK_WIDTH); /* mask width is fixed to 5 */

  imageWidth    = wbImage_getWidth(inputImage);
  imageHeight   = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  //allocate device memory
  int imageSize = (imageWidth * imageHeight * imageChannels) * sizeof(float);
  int maskSize = (maskRows * maskColumns) * sizeof(float);
  cudaMalloc((void **) &deviceInputImageData, imageSize);
  cudaMalloc((void **) &deviceOutputImageData, imageSize);
  cudaMalloc((void **) &deviceMaskData, maskSize);

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  //copy host memory to device
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, maskSize, cudaMemcpyHostToDevice);
  
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  
  //Grid height and width
  int gridWidth  = (imageWidth-1)/O_TILE_WIDTH +1;
  int gridHeight = (imageHeight-1)/O_TILE_WIDTH +1;
  dim3 DimGrid(gridWidth, gridHeight, 1);
  
  //Using Design2: 
  dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
 
  //invoking the kernel
  convolutional2D<< <DimGrid, DimBlock >> > (deviceInputImageData, deviceMaskData, deviceOutputImageData, MASK_WIDTH, imageWidth, imageHeight, imageChannels);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  //copy results from device to host	
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  //@@ INSERT CODE HERE
  //deallocate device memory	
  cudaFree(deviceInputImageData);
  cudaFree(deviceMaskData);
  cudaFree(deviceOutputImageData);

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
