#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#define var auto
#define fa(i,op,n) for(int i=op;i<=n;i++)
#define fb(i,op,n) for(int i=op;i>=n;i--)

#define X_IDX blockIdx.x * blockDim.x + threadIdx.x
#define Y_IDX blockIdx.y * blockDim.y + threadIdx.y
#define Z_IDX blockIdx.z * blockDim.z + threadIdx.z
#define BLOCK_Id_11 blockIdx.x
#define BLOCK_Id_12 blockIdx.x
#define BLOCK_Id_13 blockIdx.x
#define BLOCK_Id_21 blockIdx.y * gridDim.x + blockIdx.x
#define BLOCK_Id_22 blockIdx.y * gridDim.x + blockIdx.x
#define BLOCK_Id_23 blockIdx.y * gridDim.x + blockIdx.x
#define BLOCK_Id_31 blockIdx.z * gridDim.y * gridDim.x * blockIdx.y * gridDim.x * blockIdx.x
#define BLOCK_Id_32 blockIdx.z * gridDim.y * gridDim.x * blockIdx.y * gridDim.x * blockIdx.x
#define BLOCK_Id_33 blockIdx.z * gridDim.y * gridDim.x * blockIdx.y * gridDim.x * blockIdx.x
#define GLOBAL_INDEX_11 blockIdx.x * blockDim.x + threadIdx.x
#define GLOBAL_INDEX_12 blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
#define GLOBAL_INDEX_13 blockIdx.x * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x
#define GLOBAL_INDEX_21 (BLOCK_Id_21) * blockDim.x + threadIdx.x
#define GLOBAL_INDEX_22 (BLOCK_Id_22) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x
#define GLOBAL_INDEX_23 (BLOCK_Id_23) * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x
#define GLOBAL_INDEX_31 (BLOCK_Id_31) * blockDim.x + threadIdx.x
#define GLOBAL_INDEX_32 (BLOCK_Id_23) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x
#define GLOBAL_INDEX_33 (BLOCK_Id_33) * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x

#define CUDA_ERROR(s) if (cudaStatus != cudaSuccess) { std::cerr << "CUDA 发生错误：" << s << std::endl; goto Error; }

using uchar = unsigned char;


__global__ void downSampleKernel(uchar* old_image, int rows, int cols, int channels, uchar* new_image) {
	int x = X_IDX, y = Y_IDX;
	uchar sum = 0;
	int ori_x = x * 2;
	int ori_y = y * 2;
	fa(c, 0, 2) {
		uchar sum = 0;
		fa(dx, 0, 1)
			fa(dy, 0, 1) {
			if (ori_x + dx >= rows || ori_y + dy >= cols)continue;
			int nx = ori_x + dx, ny = ori_y + dy;
			sum += old_image[nx * cols * channels + ny * channels + c] / (2 * 2);
		}
		new_image[x * cols / 2 * channels + y * channels + c] = sum;
	}
}


cudaError_t gpu_down_sample(cv::Mat& old_image, std::shared_ptr<uchar[]> new_image)
{
	cudaError_t cudaStatus;

	int rows = old_image.rows;
	int cols = old_image.cols;
	int out_rows = rows / 2;
	int out_cols = cols / 2;
	int channels = old_image.channels();
	int old_size = rows * cols * 3;
	int new_size = rows / 2 * cols / 2 * 3;

	uchar* dev_old_image = 0;
	uchar* dev_new_image = 0;


	// block 大小：最好用 32 的倍数，且总大小 <= 1024
	// gird  大小：直接使用公式 
	dim3 block(32, 32);
	dim3 gird((out_rows + block.x - 1) / block.x, (out_cols + block.y - 1) / block.y);


	cudaStatus = cudaMalloc((void**)&dev_old_image, old_size * sizeof(uchar));
	CUDA_ERROR(1);

	cudaStatus = cudaMalloc((void**)&dev_new_image, new_size * sizeof(uchar));
	CUDA_ERROR(2);

	cudaStatus = cudaMemcpy(dev_old_image, old_image.data, old_size * sizeof(uchar), cudaMemcpyHostToDevice);
	CUDA_ERROR(3);

	downSampleKernel<<<gird, block>>>(dev_old_image, rows, cols, channels, dev_new_image);

	cudaStatus = cudaGetLastError();
	CUDA_ERROR(4);

	cudaStatus = cudaDeviceSynchronize();
	CUDA_ERROR(5);

	cudaStatus = cudaMemcpy(new_image.get(), dev_new_image, new_size * sizeof(uchar), cudaMemcpyDeviceToHost);
	CUDA_ERROR(6);


Error:
	// 释放GPU内存
	cudaFree(dev_old_image);
	cudaFree(dev_new_image);

	return cudaStatus;
}