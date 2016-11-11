#include <algorithm>

#include "caffe/common.hpp" 
#include "caffe/util/winograd.hpp"
#include "caffe/util/math_functions.hpp"

#define BLOCK_SIZE 32

namespace caffe{

template <typename Dtype> 
__global__ void padSrc_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int outH, int outW, int inputs, int batchs, int pad, float pData, int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int cIdx = idx / (batchs * outH * outW);
		int bIdx = idx / (outH * outW) % batchs;
		int yIdx = idx / outW % outH - pad;
		int xIdx = idx % outW - pad;

		if(xIdx < 0 || xIdx >= dataW || yIdx < 0 || yIdx >= dataH)
			dst[idx] = pData; 
		else
			dst[idx] = src[((bIdx * inputs + cIdx) * dataH + yIdx) * dataW + xIdx]; 
	}
}

template <typename Dtype> 
__global__ void winoWeight_gpu_kernel(const Dtype *src, Dtype *dst,  int inputs, int outputs, int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int outputIdx = idx / inputs;
		int inputIdx = idx % inputs;
		int gap = inputs * outputs;
		int kIdx = outputIdx * inputs * 9 + inputIdx * 9;
		int gIdx = idx % gap;

        dst[gIdx + 0 * gap] = + 1./1. * ( + src[kIdx + 0]);
        dst[gIdx + 1 * gap] = + 1./2. * ( + src[kIdx + 0] + src[kIdx + 1] + src[kIdx + 2]);
        dst[gIdx + 2 * gap] = + 1./2. * ( + src[kIdx + 0] - src[kIdx + 1] + src[kIdx + 2]);
        dst[gIdx + 3 * gap] = + 1./1. * ( + src[kIdx + 2]);
        dst[gIdx + 4 * gap] = + 1./2. * ( + src[kIdx + 0] + src[kIdx + 3] + src[kIdx + 6]);
        dst[gIdx + 5 * gap] = + 1./4. * ( + src[kIdx + 0] + src[kIdx + 1] + src[kIdx + 2] + src[kIdx + 3] + src[kIdx + 4] + src[kIdx + 5] + src[kIdx + 6] + src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 6 * gap] = + 1./4. * ( + src[kIdx + 0] - src[kIdx + 1] + src[kIdx + 2] + src[kIdx + 3] - src[kIdx + 4] + src[kIdx + 5] + src[kIdx + 6] - src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 7 * gap] = + 1./2. * ( + src[kIdx + 2] + src[kIdx + 5] + src[kIdx + 8]);
        dst[gIdx + 8 * gap] = + 1./2. * ( + src[kIdx + 0] - src[kIdx + 3] + src[kIdx + 6]);
        dst[gIdx + 9 * gap] = + 1./4. * ( + src[kIdx + 0] + src[kIdx + 1] + src[kIdx + 2] - src[kIdx + 3] - src[kIdx + 4] - src[kIdx + 5] + src[kIdx + 6] + src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 10 * gap] = + 1./4. * ( + src[kIdx + 0] - src[kIdx + 1] + src[kIdx + 2] - src[kIdx + 3] + src[kIdx + 4] - src[kIdx + 5] + src[kIdx + 6] - src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 11 * gap] = + 1./2. * ( + src[kIdx + 2] - src[kIdx + 5] + src[kIdx + 8]);
        dst[gIdx + 12 * gap] = + 1./1. * ( + src[kIdx + 6]);
        dst[gIdx + 13 * gap] = + 1./2. * ( + src[kIdx + 6] + src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 14 * gap] = + 1./2. * ( + src[kIdx + 6] - src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 15 * gap] = + 1./1. * ( + src[kIdx + 8]);
	}
}

template <typename Dtype> 
__global__ void wino4x4Weight_gpu_kernel(const Dtype *src, Dtype *dst,  int inputs, int outputs, int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int outputIdx = idx / inputs;
		int inputIdx = idx % inputs;
		int gap = inputs * outputs;
		int kIdx = outputIdx * inputs * 9 + inputIdx * 9;
		int gIdx = idx % gap;

        dst[gIdx + 0 * gap] = + 1./16. * ( + src[kIdx + 0]);
        dst[gIdx + 1 * gap] = + 1./24. * ( - src[kIdx + 0] - src[kIdx + 1] - src[kIdx + 2]);
        dst[gIdx + 2 * gap] = + 1./24. * ( - src[kIdx + 0] + src[kIdx + 1] - src[kIdx + 2]);
        dst[gIdx + 3 * gap] = + 1./96. * ( + src[kIdx + 0]) + 1./48. * ( + src[kIdx + 1]) + 1./24. * ( + src[kIdx + 2]);
        dst[gIdx + 4 * gap] = + 1./96. * ( + src[kIdx + 0]) + 1./48. * ( - src[kIdx + 1]) + 1./24. * ( + src[kIdx + 2]);
        dst[gIdx + 5 * gap] = + 1./4. * ( + src[kIdx + 2]);
        dst[gIdx + 6 * gap] = + 1./24. * ( - src[kIdx + 0] - src[kIdx + 3] - src[kIdx + 6]);
        dst[gIdx + 7 * gap] = + 1./36. * ( + src[kIdx + 0] + src[kIdx + 1] + src[kIdx + 2] + src[kIdx + 3] + src[kIdx + 4] + src[kIdx + 5] + src[kIdx + 6] + src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 8 * gap] = + 1./36. * ( + src[kIdx + 0] - src[kIdx + 1] + src[kIdx + 2] + src[kIdx + 3] - src[kIdx + 4] + src[kIdx + 5] + src[kIdx + 6] - src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 9 * gap] = + 1./144. * ( - src[kIdx + 0] - src[kIdx + 3] - src[kIdx + 6]) + 1./72. * ( - src[kIdx + 1] - src[kIdx + 4] - src[kIdx + 7]) + 1./36. * ( - src[kIdx + 2] - src[kIdx + 5] - src[kIdx + 8]);
        dst[gIdx + 10 * gap] = + 1./144. * ( - src[kIdx + 0] - src[kIdx + 3] - src[kIdx + 6]) + 1./72. * ( + src[kIdx + 1] + src[kIdx + 4] + src[kIdx + 7]) + 1./36. * ( - src[kIdx + 2] - src[kIdx + 5] - src[kIdx + 8]);
        dst[gIdx + 11 * gap] = + 1./6. * ( - src[kIdx + 2] - src[kIdx + 5] - src[kIdx + 8]);
        dst[gIdx + 12 * gap] = + 1./24. * ( - src[kIdx + 0] + src[kIdx + 3] - src[kIdx + 6]);
        dst[gIdx + 13 * gap] = + 1./36. * ( + src[kIdx + 0] + src[kIdx + 1] + src[kIdx + 2] - src[kIdx + 3] - src[kIdx + 4] - src[kIdx + 5] + src[kIdx + 6] + src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 14 * gap] = + 1./36. * ( + src[kIdx + 0] - src[kIdx + 1] + src[kIdx + 2] - src[kIdx + 3] + src[kIdx + 4] - src[kIdx + 5] + src[kIdx + 6] - src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 15 * gap] = + 1./144. * ( - src[kIdx + 0] + src[kIdx + 3] - src[kIdx + 6]) + 1./72. * ( - src[kIdx + 1] + src[kIdx + 4] - src[kIdx + 7]) + 1./36. * ( - src[kIdx + 2] + src[kIdx + 5] - src[kIdx + 8]);
        dst[gIdx + 16 * gap] = + 1./144. * ( - src[kIdx + 0] + src[kIdx + 3] - src[kIdx + 6]) + 1./72. * ( + src[kIdx + 1] - src[kIdx + 4] + src[kIdx + 7]) + 1./36. * ( - src[kIdx + 2] + src[kIdx + 5] - src[kIdx + 8]);
        dst[gIdx + 17 * gap] = + 1./6. * ( - src[kIdx + 2] + src[kIdx + 5] - src[kIdx + 8]);
        dst[gIdx + 18 * gap] = + 1./96. * ( + src[kIdx + 0]) + 1./48. * ( + src[kIdx + 3]) + 1./24. * ( + src[kIdx + 6]);
        dst[gIdx + 19 * gap] = + 1./144. * ( - src[kIdx + 0] - src[kIdx + 1] - src[kIdx + 2]) + 1./72. * ( - src[kIdx + 3] - src[kIdx + 4] - src[kIdx + 5]) + 1./36. * ( - src[kIdx + 6] - src[kIdx + 7] - src[kIdx + 8]);
        dst[gIdx + 20 * gap] = + 1./144. * ( - src[kIdx + 0] + src[kIdx + 1] - src[kIdx + 2]) + 1./72. * ( - src[kIdx + 3] + src[kIdx + 4] - src[kIdx + 5]) + 1./36. * ( - src[kIdx + 6] + src[kIdx + 7] - src[kIdx + 8]);
        dst[gIdx + 21 * gap] = + 1./576. * ( + src[kIdx + 0]) + 1./288. * ( + src[kIdx + 1] + src[kIdx + 3]) + 1./72. * ( + src[kIdx + 5] + src[kIdx + 7]) + 1./144. * ( + src[kIdx + 2] + src[kIdx + 4] + src[kIdx + 6]) + 1./36. * ( + src[kIdx + 8]);
        dst[gIdx + 22 * gap] = + 1./576. * ( + src[kIdx + 0]) + 1./288. * ( - src[kIdx + 1] + src[kIdx + 3]) + 1./72. * ( + src[kIdx + 5] - src[kIdx + 7]) + 1./144. * ( + src[kIdx + 2] - src[kIdx + 4] + src[kIdx + 6]) + 1./36. * ( + src[kIdx + 8]);
        dst[gIdx + 23 * gap] = + 1./24. * ( + src[kIdx + 2]) + 1./12. * ( + src[kIdx + 5]) + 1./6. * ( + src[kIdx + 8]);
        dst[gIdx + 24 * gap] = + 1./96. * ( + src[kIdx + 0]) + 1./48. * ( - src[kIdx + 3]) + 1./24. * ( + src[kIdx + 6]);
        dst[gIdx + 25 * gap] = + 1./144. * ( - src[kIdx + 0] - src[kIdx + 1] - src[kIdx + 2]) + 1./72. * ( + src[kIdx + 3] + src[kIdx + 4] + src[kIdx + 5]) + 1./36. * ( - src[kIdx + 6] - src[kIdx + 7] - src[kIdx + 8]);
        dst[gIdx + 26 * gap] = + 1./144. * ( - src[kIdx + 0] + src[kIdx + 1] - src[kIdx + 2]) + 1./72. * ( + src[kIdx + 3] - src[kIdx + 4] + src[kIdx + 5]) + 1./36. * ( - src[kIdx + 6] + src[kIdx + 7] - src[kIdx + 8]);
        dst[gIdx + 27 * gap] = + 1./576. * ( + src[kIdx + 0]) + 1./288. * ( + src[kIdx + 1] - src[kIdx + 3]) + 1./72. * ( - src[kIdx + 5] + src[kIdx + 7]) + 1./144. * ( + src[kIdx + 2] - src[kIdx + 4] + src[kIdx + 6]) + 1./36. * ( + src[kIdx + 8]);
        dst[gIdx + 28 * gap] = + 1./576. * ( + src[kIdx + 0]) + 1./288. * ( - src[kIdx + 1] - src[kIdx + 3]) + 1./72. * ( - src[kIdx + 5] - src[kIdx + 7]) + 1./144. * ( + src[kIdx + 2] + src[kIdx + 4] + src[kIdx + 6]) + 1./36. * ( + src[kIdx + 8]);
        dst[gIdx + 29 * gap] = + 1./24. * ( + src[kIdx + 2]) + 1./12. * ( - src[kIdx + 5]) + 1./6. * ( + src[kIdx + 8]);
        dst[gIdx + 30 * gap] = + 1./4. * ( + src[kIdx + 6]);
        dst[gIdx + 31 * gap] = + 1./6. * ( - src[kIdx + 6] - src[kIdx + 7] - src[kIdx + 8]);
        dst[gIdx + 32 * gap] = + 1./6. * ( - src[kIdx + 6] + src[kIdx + 7] - src[kIdx + 8]);
        dst[gIdx + 33 * gap] = + 1./24. * ( + src[kIdx + 6]) + 1./12. * ( + src[kIdx + 7]) + 1./6. * ( + src[kIdx + 8]);
        dst[gIdx + 34 * gap] = + 1./24. * ( + src[kIdx + 6]) + 1./12. * ( - src[kIdx + 7]) + 1./6. * ( + src[kIdx + 8]);
        dst[gIdx + 35 * gap] = + 1./1. * ( + src[kIdx + 8]);
	}
}

template <typename Dtype> 
__global__ void wino6x6Weight_gpu_kernel(const Dtype *src, Dtype *dst,  int inputs, int outputs, int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int outputIdx = idx / inputs;
		int inputIdx = idx % inputs;
		int gap = inputs * outputs;
		int kIdx = outputIdx * inputs * 9 + inputIdx * 9;
		int gIdx = idx % gap;

        dst[gIdx + 0 * gap] = + 1./1. * ( + src[kIdx + 0]);
        dst[gIdx + 1 * gap] = + 2./9. * ( - src[kIdx + 0] - src[kIdx + 1] - src[kIdx + 2]);
        dst[gIdx + 2 * gap] = + 2./9. * ( - src[kIdx + 0] + src[kIdx + 1] - src[kIdx + 2]);
        dst[gIdx + 3 * gap] = + 2./45. * ( + src[kIdx + 2]) + 1./90. * ( + src[kIdx + 0]) + 1./45. * ( + src[kIdx + 1]);
        dst[gIdx + 4 * gap] = + 2./45. * ( + src[kIdx + 2]) + 1./90. * ( + src[kIdx + 0]) + 1./45. * ( - src[kIdx + 1]);
        dst[gIdx + 5 * gap] = + 16./45. * ( + src[kIdx + 1]) + 32./45. * ( + src[kIdx + 0]) + 8./45. * ( + src[kIdx + 2]);
        dst[gIdx + 6 * gap] = + 16./45. * ( - src[kIdx + 1]) + 32./45. * ( + src[kIdx + 0]) + 8./45. * ( + src[kIdx + 2]);
        dst[gIdx + 7 * gap] = + 1./1. * ( + src[kIdx + 2]);
        dst[gIdx + 8 * gap] = + 2./9. * ( - src[kIdx + 0] - src[kIdx + 3] - src[kIdx + 6]);
        dst[gIdx + 9 * gap] = + 4./81. * ( + src[kIdx + 0] + src[kIdx + 1] + src[kIdx + 2] + src[kIdx + 3] + src[kIdx + 4] + src[kIdx + 5] + src[kIdx + 6] + src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 10 * gap] = + 4./81. * ( + src[kIdx + 0] - src[kIdx + 1] + src[kIdx + 2] + src[kIdx + 3] - src[kIdx + 4] + src[kIdx + 5] + src[kIdx + 6] - src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 11 * gap] = + 2./405. * ( - src[kIdx + 1] - src[kIdx + 4] - src[kIdx + 7]) + 4./405. * ( - src[kIdx + 2] - src[kIdx + 5] - src[kIdx + 8]) + 1./405. * ( - src[kIdx + 0] - src[kIdx + 3] - src[kIdx + 6]);
        dst[gIdx + 12 * gap] = + 2./405. * ( + src[kIdx + 1] + src[kIdx + 4] + src[kIdx + 7]) + 4./405. * ( - src[kIdx + 2] - src[kIdx + 5] - src[kIdx + 8]) + 1./405. * ( - src[kIdx + 0] - src[kIdx + 3] - src[kIdx + 6]);
        dst[gIdx + 13 * gap] = + 16./405. * ( - src[kIdx + 2] - src[kIdx + 5] - src[kIdx + 8]) + 32./405. * ( - src[kIdx + 1] - src[kIdx + 4] - src[kIdx + 7]) + 64./405. * ( - src[kIdx + 0] - src[kIdx + 3] - src[kIdx + 6]);
        dst[gIdx + 14 * gap] = + 16./405. * ( - src[kIdx + 2] - src[kIdx + 5] - src[kIdx + 8]) + 32./405. * ( + src[kIdx + 1] + src[kIdx + 4] + src[kIdx + 7]) + 64./405. * ( - src[kIdx + 0] - src[kIdx + 3] - src[kIdx + 6]);
        dst[gIdx + 15 * gap] = + 2./9. * ( - src[kIdx + 2] - src[kIdx + 5] - src[kIdx + 8]);
        dst[gIdx + 16 * gap] = + 2./9. * ( - src[kIdx + 0] + src[kIdx + 3] - src[kIdx + 6]);
        dst[gIdx + 17 * gap] = + 4./81. * ( + src[kIdx + 0] + src[kIdx + 1] + src[kIdx + 2] - src[kIdx + 3] - src[kIdx + 4] - src[kIdx + 5] + src[kIdx + 6] + src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 18 * gap] = + 4./81. * ( + src[kIdx + 0] - src[kIdx + 1] + src[kIdx + 2] - src[kIdx + 3] + src[kIdx + 4] - src[kIdx + 5] + src[kIdx + 6] - src[kIdx + 7] + src[kIdx + 8]);
        dst[gIdx + 19 * gap] = + 2./405. * ( - src[kIdx + 1] + src[kIdx + 4] - src[kIdx + 7]) + 4./405. * ( - src[kIdx + 2] + src[kIdx + 5] - src[kIdx + 8]) + 1./405. * ( - src[kIdx + 0] + src[kIdx + 3] - src[kIdx + 6]);
        dst[gIdx + 20 * gap] = + 2./405. * ( + src[kIdx + 1] - src[kIdx + 4] + src[kIdx + 7]) + 4./405. * ( - src[kIdx + 2] + src[kIdx + 5] - src[kIdx + 8]) + 1./405. * ( - src[kIdx + 0] + src[kIdx + 3] - src[kIdx + 6]);
        dst[gIdx + 21 * gap] = + 16./405. * ( - src[kIdx + 2] + src[kIdx + 5] - src[kIdx + 8]) + 32./405. * ( - src[kIdx + 1] + src[kIdx + 4] - src[kIdx + 7]) + 64./405. * ( - src[kIdx + 0] + src[kIdx + 3] - src[kIdx + 6]);
        dst[gIdx + 22 * gap] = + 16./405. * ( - src[kIdx + 2] + src[kIdx + 5] - src[kIdx + 8]) + 32./405. * ( + src[kIdx + 1] - src[kIdx + 4] + src[kIdx + 7]) + 64./405. * ( - src[kIdx + 0] + src[kIdx + 3] - src[kIdx + 6]);
        dst[gIdx + 23 * gap] = + 2./9. * ( - src[kIdx + 2] + src[kIdx + 5] - src[kIdx + 8]);
        dst[gIdx + 24 * gap] = + 2./45. * ( + src[kIdx + 6]) + 1./90. * ( + src[kIdx + 0]) + 1./45. * ( + src[kIdx + 3]);
        dst[gIdx + 25 * gap] = + 2./405. * ( - src[kIdx + 3] - src[kIdx + 4] - src[kIdx + 5]) + 4./405. * ( - src[kIdx + 6] - src[kIdx + 7] - src[kIdx + 8]) + 1./405. * ( - src[kIdx + 0] - src[kIdx + 1] - src[kIdx + 2]);
        dst[gIdx + 26 * gap] = + 2./405. * ( - src[kIdx + 3] + src[kIdx + 4] - src[kIdx + 5]) + 4./405. * ( - src[kIdx + 6] + src[kIdx + 7] - src[kIdx + 8]) + 1./405. * ( - src[kIdx + 0] + src[kIdx + 1] - src[kIdx + 2]);
        dst[gIdx + 27 * gap] = + 2./2025. * ( + src[kIdx + 5] + src[kIdx + 7]) + 1./4050. * ( + src[kIdx + 1] + src[kIdx + 3]) + 1./8100. * ( + src[kIdx + 0]) + 4./2025. * ( + src[kIdx + 8]) + 1./2025. * ( + src[kIdx + 2] + src[kIdx + 4] + src[kIdx + 6]);
        dst[gIdx + 28 * gap] = + 2./2025. * ( + src[kIdx + 5] - src[kIdx + 7]) + 1./4050. * ( - src[kIdx + 1] + src[kIdx + 3]) + 1./8100. * ( + src[kIdx + 0]) + 4./2025. * ( + src[kIdx + 8]) + 1./2025. * ( + src[kIdx + 2] - src[kIdx + 4] + src[kIdx + 6]);
        dst[gIdx + 29 * gap] = + 32./2025. * ( + src[kIdx + 3] + src[kIdx + 7]) + 16./2025. * ( + src[kIdx + 0] + src[kIdx + 4] + src[kIdx + 8]) + 64./2025. * ( + src[kIdx + 6]) + 8./2025. * ( + src[kIdx + 1] + src[kIdx + 5]) + 4./2025. * ( + src[kIdx + 2]);
        dst[gIdx + 30 * gap] = + 32./2025. * ( + src[kIdx + 3] - src[kIdx + 7]) + 16./2025. * ( + src[kIdx + 0] - src[kIdx + 4] + src[kIdx + 8]) + 64./2025. * ( + src[kIdx + 6]) + 8./2025. * ( - src[kIdx + 1] + src[kIdx + 5]) + 4./2025. * ( + src[kIdx + 2]);
        dst[gIdx + 31 * gap] = + 2./45. * ( + src[kIdx + 8]) + 1./90. * ( + src[kIdx + 2]) + 1./45. * ( + src[kIdx + 5]);
        dst[gIdx + 32 * gap] = + 2./45. * ( + src[kIdx + 6]) + 1./90. * ( + src[kIdx + 0]) + 1./45. * ( - src[kIdx + 3]);
        dst[gIdx + 33 * gap] = + 2./405. * ( + src[kIdx + 3] + src[kIdx + 4] + src[kIdx + 5]) + 4./405. * ( - src[kIdx + 6] - src[kIdx + 7] - src[kIdx + 8]) + 1./405. * ( - src[kIdx + 0] - src[kIdx + 1] - src[kIdx + 2]);
        dst[gIdx + 34 * gap] = + 2./405. * ( + src[kIdx + 3] - src[kIdx + 4] + src[kIdx + 5]) + 4./405. * ( - src[kIdx + 6] + src[kIdx + 7] - src[kIdx + 8]) + 1./405. * ( - src[kIdx + 0] + src[kIdx + 1] - src[kIdx + 2]);
        dst[gIdx + 35 * gap] = + 2./2025. * ( - src[kIdx + 5] + src[kIdx + 7]) + 1./4050. * ( + src[kIdx + 1] - src[kIdx + 3]) + 1./8100. * ( + src[kIdx + 0]) + 4./2025. * ( + src[kIdx + 8]) + 1./2025. * ( + src[kIdx + 2] - src[kIdx + 4] + src[kIdx + 6]);
        dst[gIdx + 36 * gap] = + 2./2025. * ( - src[kIdx + 5] - src[kIdx + 7]) + 1./4050. * ( - src[kIdx + 1] - src[kIdx + 3]) + 1./8100. * ( + src[kIdx + 0]) + 4./2025. * ( + src[kIdx + 8]) + 1./2025. * ( + src[kIdx + 2] + src[kIdx + 4] + src[kIdx + 6]);
        dst[gIdx + 37 * gap] = + 32./2025. * ( - src[kIdx + 3] + src[kIdx + 7]) + 16./2025. * ( + src[kIdx + 0] - src[kIdx + 4] + src[kIdx + 8]) + 64./2025. * ( + src[kIdx + 6]) + 8./2025. * ( + src[kIdx + 1] - src[kIdx + 5]) + 4./2025. * ( + src[kIdx + 2]);
        dst[gIdx + 38 * gap] = + 32./2025. * ( - src[kIdx + 3] - src[kIdx + 7]) + 16./2025. * ( + src[kIdx + 0] + src[kIdx + 4] + src[kIdx + 8]) + 64./2025. * ( + src[kIdx + 6]) + 8./2025. * ( - src[kIdx + 1] - src[kIdx + 5]) + 4./2025. * ( + src[kIdx + 2]);
        dst[gIdx + 39 * gap] = + 2./45. * ( + src[kIdx + 8]) + 1./90. * ( + src[kIdx + 2]) + 1./45. * ( - src[kIdx + 5]);
        dst[gIdx + 40 * gap] = + 16./45. * ( + src[kIdx + 3]) + 32./45. * ( + src[kIdx + 0]) + 8./45. * ( + src[kIdx + 6]);
        dst[gIdx + 41 * gap] = + 16./405. * ( - src[kIdx + 6] - src[kIdx + 7] - src[kIdx + 8]) + 32./405. * ( - src[kIdx + 3] - src[kIdx + 4] - src[kIdx + 5]) + 64./405. * ( - src[kIdx + 0] - src[kIdx + 1] - src[kIdx + 2]);
        dst[gIdx + 42 * gap] = + 16./405. * ( - src[kIdx + 6] + src[kIdx + 7] - src[kIdx + 8]) + 32./405. * ( - src[kIdx + 3] + src[kIdx + 4] - src[kIdx + 5]) + 64./405. * ( - src[kIdx + 0] + src[kIdx + 1] - src[kIdx + 2]);
        dst[gIdx + 43 * gap] = + 8./2025. * ( + src[kIdx + 3] + src[kIdx + 7]) + 16./2025. * ( + src[kIdx + 0] + src[kIdx + 4] + src[kIdx + 8]) + 64./2025. * ( + src[kIdx + 2]) + 32./2025. * ( + src[kIdx + 1] + src[kIdx + 5]) + 4./2025. * ( + src[kIdx + 6]);
        dst[gIdx + 44 * gap] = + 8./2025. * ( + src[kIdx + 3] - src[kIdx + 7]) + 16./2025. * ( + src[kIdx + 0] - src[kIdx + 4] + src[kIdx + 8]) + 64./2025. * ( + src[kIdx + 2]) + 32./2025. * ( - src[kIdx + 1] + src[kIdx + 5]) + 4./2025. * ( + src[kIdx + 6]);
        dst[gIdx + 45 * gap] = + 1024./2025. * ( + src[kIdx + 0]) + 128./2025. * ( + src[kIdx + 5] + src[kIdx + 7]) + 64./2025. * ( + src[kIdx + 8]) + 512./2025. * ( + src[kIdx + 1] + src[kIdx + 3]) + 256./2025. * ( + src[kIdx + 2] + src[kIdx + 4] + src[kIdx + 6]);
        dst[gIdx + 46 * gap] = + 1024./2025. * ( + src[kIdx + 0]) + 128./2025. * ( + src[kIdx + 5] - src[kIdx + 7]) + 64./2025. * ( + src[kIdx + 8]) + 512./2025. * ( - src[kIdx + 1] + src[kIdx + 3]) + 256./2025. * ( + src[kIdx + 2] - src[kIdx + 4] + src[kIdx + 6]);
        dst[gIdx + 47 * gap] = + 16./45. * ( + src[kIdx + 5]) + 32./45. * ( + src[kIdx + 2]) + 8./45. * ( + src[kIdx + 8]);
        dst[gIdx + 48 * gap] = + 16./45. * ( - src[kIdx + 3]) + 32./45. * ( + src[kIdx + 0]) + 8./45. * ( + src[kIdx + 6]);
        dst[gIdx + 49 * gap] = + 16./405. * ( - src[kIdx + 6] - src[kIdx + 7] - src[kIdx + 8]) + 32./405. * ( + src[kIdx + 3] + src[kIdx + 4] + src[kIdx + 5]) + 64./405. * ( - src[kIdx + 0] - src[kIdx + 1] - src[kIdx + 2]);
        dst[gIdx + 50 * gap] = + 16./405. * ( - src[kIdx + 6] + src[kIdx + 7] - src[kIdx + 8]) + 32./405. * ( + src[kIdx + 3] - src[kIdx + 4] + src[kIdx + 5]) + 64./405. * ( - src[kIdx + 0] + src[kIdx + 1] - src[kIdx + 2]);
        dst[gIdx + 51 * gap] = + 8./2025. * ( - src[kIdx + 3] + src[kIdx + 7]) + 16./2025. * ( + src[kIdx + 0] - src[kIdx + 4] + src[kIdx + 8]) + 64./2025. * ( + src[kIdx + 2]) + 32./2025. * ( + src[kIdx + 1] - src[kIdx + 5]) + 4./2025. * ( + src[kIdx + 6]);
        dst[gIdx + 52 * gap] = + 8./2025. * ( - src[kIdx + 3] - src[kIdx + 7]) + 16./2025. * ( + src[kIdx + 0] + src[kIdx + 4] + src[kIdx + 8]) + 64./2025. * ( + src[kIdx + 2]) + 32./2025. * ( - src[kIdx + 1] - src[kIdx + 5]) + 4./2025. * ( + src[kIdx + 6]);
        dst[gIdx + 53 * gap] = + 1024./2025. * ( + src[kIdx + 0]) + 128./2025. * ( - src[kIdx + 5] + src[kIdx + 7]) + 64./2025. * ( + src[kIdx + 8]) + 512./2025. * ( + src[kIdx + 1] - src[kIdx + 3]) + 256./2025. * ( + src[kIdx + 2] - src[kIdx + 4] + src[kIdx + 6]);
        dst[gIdx + 54 * gap] = + 1024./2025. * ( + src[kIdx + 0]) + 128./2025. * ( - src[kIdx + 5] - src[kIdx + 7]) + 64./2025. * ( + src[kIdx + 8]) + 512./2025. * ( - src[kIdx + 1] - src[kIdx + 3]) + 256./2025. * ( + src[kIdx + 2] + src[kIdx + 4] + src[kIdx + 6]);
        dst[gIdx + 55 * gap] = + 16./45. * ( - src[kIdx + 5]) + 32./45. * ( + src[kIdx + 2]) + 8./45. * ( + src[kIdx + 8]);
        dst[gIdx + 56 * gap] = + 1./1. * ( + src[kIdx + 6]);
        dst[gIdx + 57 * gap] = + 2./9. * ( - src[kIdx + 6] - src[kIdx + 7] - src[kIdx + 8]);
        dst[gIdx + 58 * gap] = + 2./9. * ( - src[kIdx + 6] + src[kIdx + 7] - src[kIdx + 8]);
        dst[gIdx + 59 * gap] = + 2./45. * ( + src[kIdx + 8]) + 1./90. * ( + src[kIdx + 6]) + 1./45. * ( + src[kIdx + 7]);
        dst[gIdx + 60 * gap] = + 2./45. * ( + src[kIdx + 8]) + 1./90. * ( + src[kIdx + 6]) + 1./45. * ( - src[kIdx + 7]);
        dst[gIdx + 61 * gap] = + 16./45. * ( + src[kIdx + 7]) + 32./45. * ( + src[kIdx + 6]) + 8./45. * ( + src[kIdx + 8]);
        dst[gIdx + 62 * gap] = + 16./45. * ( - src[kIdx + 7]) + 32./45. * ( + src[kIdx + 6]) + 8./45. * ( + src[kIdx + 8]);
        dst[gIdx + 63 * gap] = + 1./1. * ( + src[kIdx + 8]);
	}
}

template <typename Dtype> 
__global__ void winoSrc_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int tileH, int tileW, int inputs, int batchs, int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int gap = inputs * batchs * tileH * tileW;
		int highIdx = idx / (tileH * tileW);
		int yIdx = idx / tileW % tileH;
		int xIdx = idx % tileW;
		int bIdx = idx;
		int sIdx = highIdx * dataW * dataH + yIdx * dataW * 2 + xIdx * 2;

        dst[bIdx + 0 * gap] = + 1./1. * ( + src[sIdx + 0 * dataW + 0] - src[sIdx + 0 * dataW + 2] - src[sIdx + 2 * dataW + 0] + src[sIdx + 2 * dataW + 2]);
        dst[bIdx + 1 * gap] = + 1./1. * ( + src[sIdx + 0 * dataW + 1] + src[sIdx + 0 * dataW + 2] - src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 2]);
        dst[bIdx + 2 * gap] = + 1./1. * ( - src[sIdx + 0 * dataW + 1] + src[sIdx + 0 * dataW + 2] + src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 2]);
        dst[bIdx + 3 * gap] = + 1./1. * ( - src[sIdx + 0 * dataW + 1] + src[sIdx + 0 * dataW + 3] + src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 3]);
        dst[bIdx + 4 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 0] - src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 0] - src[sIdx + 2 * dataW + 2]);
        dst[bIdx + 5 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2]);
        dst[bIdx + 6 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2]);
        dst[bIdx + 7 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 3]);
        dst[bIdx + 8 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 0] + src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 0] - src[sIdx + 2 * dataW + 2]);
        dst[bIdx + 9 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2]);
        dst[bIdx + 10 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2]);
        dst[bIdx + 11 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 3]);
        dst[bIdx + 12 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 0] + src[sIdx + 1 * dataW + 2] + src[sIdx + 3 * dataW + 0] - src[sIdx + 3 * dataW + 2]);
        dst[bIdx + 13 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] + src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2]);
        dst[bIdx + 14 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2]);
        dst[bIdx + 15 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 3] - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 3]);
	}
}


template <typename Dtype> 
__global__ void wino4x4Src_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int tileH, int tileW, int inputs, int batchs, int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int gap = inputs * batchs * tileH * tileW;
		int highIdx = idx / (tileH * tileW);
		int yIdx = idx / tileW % tileH;
		int xIdx = idx % tileW;
		int bIdx = idx;
		int sIdx = highIdx * dataW * dataH + yIdx * dataW * 4 + xIdx * 4;

        dst[bIdx + 0 * gap] = + 1./1. * ( + src[sIdx + 4 * dataW + 4]) + 4./1. * ( + src[sIdx + 0 * dataW + 4] + src[sIdx + 4 * dataW + 0]) + 5./1. * ( - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2]) + 16./1. * ( + src[sIdx + 0 * dataW + 0]) + 20./1. * ( - src[sIdx + 0 * dataW + 2] - src[sIdx + 2 * dataW + 0]) + 25./1. * ( + src[sIdx + 2 * dataW + 2]);
        dst[bIdx + 1 * gap] = + 16./1. * ( - src[sIdx + 0 * dataW + 1] - src[sIdx + 0 * dataW + 2]) + 20./1. * ( + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2]) + 4./1. * ( + src[sIdx + 0 * dataW + 3] + src[sIdx + 0 * dataW + 4] - src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2]) + 5./1. * ( - src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4]) + 1./1. * ( + src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]);
        dst[bIdx + 2 * gap] = + 16./1. * ( + src[sIdx + 0 * dataW + 1] - src[sIdx + 0 * dataW + 2]) + 20./1. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2]) + 4./1. * ( - src[sIdx + 0 * dataW + 3] + src[sIdx + 0 * dataW + 4] + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2]) + 5./1. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4]) + 1./1. * ( - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]);
        dst[bIdx + 3 * gap] = + 1./1. * ( - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( - src[sIdx + 4 * dataW + 1] + src[sIdx + 4 * dataW + 3]) + 4./1. * ( - src[sIdx + 0 * dataW + 2] + src[sIdx + 0 * dataW + 4]) + 5./1. * ( + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 4]) + 8./1. * ( - src[sIdx + 0 * dataW + 1] + src[sIdx + 0 * dataW + 3]) + 10./1. * ( + src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 3]);
        dst[bIdx + 4 * gap] = + 1./1. * ( - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 3]) + 4./1. * ( - src[sIdx + 0 * dataW + 2] + src[sIdx + 0 * dataW + 4]) + 5./1. * ( + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 4]) + 8./1. * ( + src[sIdx + 0 * dataW + 1] - src[sIdx + 0 * dataW + 3]) + 10./1. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 3]);
        dst[bIdx + 5 * gap] = + 1./1. * ( + src[sIdx + 4 * dataW + 5]) + 4./1. * ( + src[sIdx + 0 * dataW + 5] + src[sIdx + 4 * dataW + 1]) + 5./1. * ( - src[sIdx + 2 * dataW + 5] - src[sIdx + 4 * dataW + 3]) + 16./1. * ( + src[sIdx + 0 * dataW + 1]) + 20./1. * ( - src[sIdx + 0 * dataW + 3] - src[sIdx + 2 * dataW + 1]) + 25./1. * ( + src[sIdx + 2 * dataW + 3]);
        dst[bIdx + 6 * gap] = + 16./1. * ( - src[sIdx + 1 * dataW + 0] - src[sIdx + 2 * dataW + 0]) + 4./1. * ( - src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 4] + src[sIdx + 3 * dataW + 0] + src[sIdx + 4 * dataW + 0]) + 20./1. * ( + src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 2]) + 5./1. * ( - src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 1./1. * ( + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4]);
        dst[bIdx + 7 * gap] = + 16./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2]) + 1./1. * ( + src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 4./1. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4] - src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2]);
        dst[bIdx + 8 * gap] = + 16./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2]) + 1./1. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 4./1. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4] + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 2] + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2]);
        dst[bIdx + 9 * gap] = + 8./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 3] + src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 3]) + 1./1. * ( - src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 3] - src[sIdx + 4 * dataW + 1] + src[sIdx + 4 * dataW + 3]) + 4./1. * ( + src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 4]);
        dst[bIdx + 10 * gap] = + 8./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 3]) + 1./1. * ( - src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 3]) + 4./1. * ( + src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 4]);
        dst[bIdx + 11 * gap] = + 16./1. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 2 * dataW + 1]) + 4./1. * ( - src[sIdx + 1 * dataW + 5] - src[sIdx + 2 * dataW + 5] + src[sIdx + 3 * dataW + 1] + src[sIdx + 4 * dataW + 1]) + 20./1. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 2 * dataW + 3]) + 5./1. * ( - src[sIdx + 3 * dataW + 3] - src[sIdx + 4 * dataW + 3]) + 1./1. * ( + src[sIdx + 3 * dataW + 5] + src[sIdx + 4 * dataW + 5]);
        dst[bIdx + 12 * gap] = + 16./1. * ( + src[sIdx + 1 * dataW + 0] - src[sIdx + 2 * dataW + 0]) + 4./1. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 4] - src[sIdx + 3 * dataW + 0] + src[sIdx + 4 * dataW + 0]) + 20./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 2]) + 5./1. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 1./1. * ( - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4]);
        dst[bIdx + 13 * gap] = + 16./1. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2]) + 1./1. * ( - src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 4./1. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4] + src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2]);
        dst[bIdx + 14 * gap] = + 16./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2]) + 1./1. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 4./1. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4] - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2] + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2]);
        dst[bIdx + 15 * gap] = + 8./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 3] + src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 3]) + 1./1. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 3] - src[sIdx + 4 * dataW + 1] + src[sIdx + 4 * dataW + 3]) + 4./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 4]);
        dst[bIdx + 16 * gap] = + 8./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 3]) + 1./1. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 3]) + 4./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 4]);
        dst[bIdx + 17 * gap] = + 16./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 2 * dataW + 1]) + 4./1. * ( + src[sIdx + 1 * dataW + 5] - src[sIdx + 2 * dataW + 5] - src[sIdx + 3 * dataW + 1] + src[sIdx + 4 * dataW + 1]) + 20./1. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 2 * dataW + 3]) + 5./1. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 4 * dataW + 3]) + 1./1. * ( - src[sIdx + 3 * dataW + 5] + src[sIdx + 4 * dataW + 5]);
        dst[bIdx + 18 * gap] = + 1./1. * ( - src[sIdx + 2 * dataW + 4] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( - src[sIdx + 1 * dataW + 4] + src[sIdx + 3 * dataW + 4]) + 4./1. * ( - src[sIdx + 2 * dataW + 0] + src[sIdx + 4 * dataW + 0]) + 5./1. * ( + src[sIdx + 2 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 8./1. * ( - src[sIdx + 1 * dataW + 0] + src[sIdx + 3 * dataW + 0]) + 10./1. * ( + src[sIdx + 1 * dataW + 2] - src[sIdx + 3 * dataW + 2]);
        dst[bIdx + 19 * gap] = + 8./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] - src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 2]) + 1./1. * ( - src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4] + src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 4] + src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 4]) + 4./1. * ( + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] - src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2]);
        dst[bIdx + 20 * gap] = + 8./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 2]) + 1./1. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 4] - src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 4]) + 4./1. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2]);
        dst[bIdx + 21 * gap] = + 1./1. * ( + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( + src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 3] - src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 1] + src[sIdx + 4 * dataW + 3]) + 4./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 3] - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 22 * gap] = + 1./1. * ( + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( + src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 3] - src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 3]) + 4./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 3] + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 23 * gap] = + 1./1. * ( - src[sIdx + 2 * dataW + 5] + src[sIdx + 4 * dataW + 5]) + 2./1. * ( - src[sIdx + 1 * dataW + 5] + src[sIdx + 3 * dataW + 5]) + 4./1. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 4 * dataW + 1]) + 5./1. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 4 * dataW + 3]) + 8./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 3 * dataW + 1]) + 10./1. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 24 * gap] = + 1./1. * ( - src[sIdx + 2 * dataW + 4] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 3 * dataW + 4]) + 4./1. * ( - src[sIdx + 2 * dataW + 0] + src[sIdx + 4 * dataW + 0]) + 5./1. * ( + src[sIdx + 2 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 8./1. * ( + src[sIdx + 1 * dataW + 0] - src[sIdx + 3 * dataW + 0]) + 10./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 3 * dataW + 2]);
        dst[bIdx + 25 * gap] = + 8./1. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] + src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2]) + 1./1. * ( - src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4] + src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4] - src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4]) + 4./1. * ( + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] - src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2]);
        dst[bIdx + 26 * gap] = + 8./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2]) + 1./1. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4] + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4]) + 4./1. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2]);
        dst[bIdx + 27 * gap] = + 1./1. * ( + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 3] + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 1] + src[sIdx + 4 * dataW + 3]) + 4./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 3] + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 28 * gap] = + 1./1. * ( + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 4]) + 2./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 3] + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 3]) + 4./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 3] - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 29 * gap] = + 1./1. * ( - src[sIdx + 2 * dataW + 5] + src[sIdx + 4 * dataW + 5]) + 2./1. * ( + src[sIdx + 1 * dataW + 5] - src[sIdx + 3 * dataW + 5]) + 4./1. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 4 * dataW + 1]) + 5./1. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 4 * dataW + 3]) + 8./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 3 * dataW + 1]) + 10./1. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 30 * gap] = + 1./1. * ( + src[sIdx + 5 * dataW + 4]) + 4./1. * ( + src[sIdx + 1 * dataW + 4] + src[sIdx + 5 * dataW + 0]) + 5./1. * ( - src[sIdx + 3 * dataW + 4] - src[sIdx + 5 * dataW + 2]) + 16./1. * ( + src[sIdx + 1 * dataW + 0]) + 20./1. * ( - src[sIdx + 1 * dataW + 2] - src[sIdx + 3 * dataW + 0]) + 25./1. * ( + src[sIdx + 3 * dataW + 2]);
        dst[bIdx + 31 * gap] = + 16./1. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2]) + 20./1. * ( + src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2]) + 4./1. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4] - src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 2]) + 5./1. * ( - src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4]) + 1./1. * ( + src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4]);
        dst[bIdx + 32 * gap] = + 16./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2]) + 20./1. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2]) + 4./1. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4] + src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 2]) + 5./1. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4]) + 1./1. * ( - src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4]);
        dst[bIdx + 33 * gap] = + 1./1. * ( - src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 4]) + 2./1. * ( - src[sIdx + 5 * dataW + 1] + src[sIdx + 5 * dataW + 3]) + 4./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 4]) + 5./1. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 4]) + 8./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 3]) + 10./1. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 34 * gap] = + 1./1. * ( - src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 4]) + 2./1. * ( + src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 3]) + 4./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 4]) + 5./1. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 4]) + 8./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 3]) + 10./1. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 35 * gap] = + 1./1. * ( + src[sIdx + 5 * dataW + 5]) + 4./1. * ( + src[sIdx + 1 * dataW + 5] + src[sIdx + 5 * dataW + 1]) + 5./1. * ( - src[sIdx + 3 * dataW + 5] - src[sIdx + 5 * dataW + 3]) + 16./1. * ( + src[sIdx + 1 * dataW + 1]) + 20./1. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 3 * dataW + 1]) + 25./1. * ( + src[sIdx + 3 * dataW + 3]);
	}
}

template <typename Dtype> 
__global__ void wino6x6Src_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int tileH, int tileW, int inputs, int batchs, int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int gap = inputs * batchs * tileH * tileW;
		int highIdx = idx / (tileH * tileW);
		int yIdx = idx / tileW % tileH;
		int xIdx = idx % tileW;
		int bIdx = idx;
		int sIdx = highIdx * dataW * dataH + yIdx * dataW * 6 + xIdx * 6;

        dst[bIdx + 0 * gap] = + 21./4. * ( - src[sIdx + 0 * dataW + 2] + src[sIdx + 0 * dataW + 4] - src[sIdx + 2 * dataW + 0] + src[sIdx + 2 * dataW + 6] + src[sIdx + 4 * dataW + 0] - src[sIdx + 4 * dataW + 6] + src[sIdx + 6 * dataW + 2] - src[sIdx + 6 * dataW + 4]) + 1./1. * ( + src[sIdx + 0 * dataW + 0] - src[sIdx + 0 * dataW + 6] - src[sIdx + 6 * dataW + 0] + src[sIdx + 6 * dataW + 6]) + 441./16. * ( + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 4]);
        dst[bIdx + 1 * gap] = + 21./4. * ( - src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 5] - src[sIdx + 2 * dataW + 6] + src[sIdx + 4 * dataW + 1] + src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 5] + src[sIdx + 4 * dataW + 6]) + 1./1. * ( + src[sIdx + 0 * dataW + 1] + src[sIdx + 0 * dataW + 2] + src[sIdx + 0 * dataW + 5] + src[sIdx + 0 * dataW + 6] - src[sIdx + 6 * dataW + 1] - src[sIdx + 6 * dataW + 2] - src[sIdx + 6 * dataW + 5] - src[sIdx + 6 * dataW + 6]) + 357./16. * ( + src[sIdx + 2 * dataW + 3] + src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 3] - src[sIdx + 4 * dataW + 4]) + 17./4. * ( - src[sIdx + 0 * dataW + 3] - src[sIdx + 0 * dataW + 4] + src[sIdx + 6 * dataW + 3] + src[sIdx + 6 * dataW + 4]);
        dst[bIdx + 2 * gap] = + 21./4. * ( + src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 5] - src[sIdx + 2 * dataW + 6] - src[sIdx + 4 * dataW + 1] + src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 5] + src[sIdx + 4 * dataW + 6]) + 1./1. * ( - src[sIdx + 0 * dataW + 1] + src[sIdx + 0 * dataW + 2] - src[sIdx + 0 * dataW + 5] + src[sIdx + 0 * dataW + 6] + src[sIdx + 6 * dataW + 1] - src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 5] - src[sIdx + 6 * dataW + 6]) + 357./16. * ( - src[sIdx + 2 * dataW + 3] + src[sIdx + 2 * dataW + 4] + src[sIdx + 4 * dataW + 3] - src[sIdx + 4 * dataW + 4]) + 17./4. * ( + src[sIdx + 0 * dataW + 3] - src[sIdx + 0 * dataW + 4] - src[sIdx + 6 * dataW + 3] + src[sIdx + 6 * dataW + 4]);
        dst[bIdx + 3 * gap] = + 105./16. * ( + src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 4]) + 1./2. * ( + src[sIdx + 0 * dataW + 1] - src[sIdx + 6 * dataW + 1]) + 2./1. * ( + src[sIdx + 0 * dataW + 5] - src[sIdx + 6 * dataW + 5]) + 1./4. * ( + src[sIdx + 0 * dataW + 2] - src[sIdx + 6 * dataW + 2]) + 21./2. * ( - src[sIdx + 2 * dataW + 5] + src[sIdx + 4 * dataW + 5]) + 1./1. * ( + src[sIdx + 0 * dataW + 6] - src[sIdx + 6 * dataW + 6]) + 5./4. * ( - src[sIdx + 0 * dataW + 4] + src[sIdx + 6 * dataW + 4]) + 105./8. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 4 * dataW + 3]) + 5./2. * ( - src[sIdx + 0 * dataW + 3] + src[sIdx + 6 * dataW + 3]) + 21./4. * ( - src[sIdx + 2 * dataW + 6] + src[sIdx + 4 * dataW + 6]) + 21./16. * ( - src[sIdx + 2 * dataW + 2] + src[sIdx + 4 * dataW + 2]) + 21./8. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 4 * dataW + 1]);
        dst[bIdx + 4 * gap] = + 105./16. * ( + src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 4]) + 1./2. * ( - src[sIdx + 0 * dataW + 1] + src[sIdx + 6 * dataW + 1]) + 2./1. * ( - src[sIdx + 0 * dataW + 5] + src[sIdx + 6 * dataW + 5]) + 1./4. * ( + src[sIdx + 0 * dataW + 2] - src[sIdx + 6 * dataW + 2]) + 21./2. * ( + src[sIdx + 2 * dataW + 5] - src[sIdx + 4 * dataW + 5]) + 1./1. * ( + src[sIdx + 0 * dataW + 6] - src[sIdx + 6 * dataW + 6]) + 5./4. * ( - src[sIdx + 0 * dataW + 4] + src[sIdx + 6 * dataW + 4]) + 105./8. * ( - src[sIdx + 2 * dataW + 3] + src[sIdx + 4 * dataW + 3]) + 5./2. * ( + src[sIdx + 0 * dataW + 3] - src[sIdx + 6 * dataW + 3]) + 21./4. * ( - src[sIdx + 2 * dataW + 6] + src[sIdx + 4 * dataW + 6]) + 21./16. * ( - src[sIdx + 2 * dataW + 2] + src[sIdx + 4 * dataW + 2]) + 21./8. * ( + src[sIdx + 2 * dataW + 1] - src[sIdx + 4 * dataW + 1]);
        dst[bIdx + 5 * gap] = + 1./2. * ( + src[sIdx + 0 * dataW + 5] - src[sIdx + 6 * dataW + 5]) + 2./1. * ( + src[sIdx + 0 * dataW + 1] - src[sIdx + 6 * dataW + 1]) + 4./1. * ( + src[sIdx + 0 * dataW + 2] - src[sIdx + 6 * dataW + 2]) + 5./1. * ( - src[sIdx + 0 * dataW + 4] + src[sIdx + 6 * dataW + 4]) + 21./2. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 4 * dataW + 1]) + 1./1. * ( + src[sIdx + 0 * dataW + 6] - src[sIdx + 6 * dataW + 6]) + 105./8. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 4 * dataW + 3]) + 105./4. * ( + src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 4]) + 21./1. * ( - src[sIdx + 2 * dataW + 2] + src[sIdx + 4 * dataW + 2]) + 5./2. * ( - src[sIdx + 0 * dataW + 3] + src[sIdx + 6 * dataW + 3]) + 21./4. * ( - src[sIdx + 2 * dataW + 6] + src[sIdx + 4 * dataW + 6]) + 21./8. * ( - src[sIdx + 2 * dataW + 5] + src[sIdx + 4 * dataW + 5]);
        dst[bIdx + 6 * gap] = + 1./2. * ( - src[sIdx + 0 * dataW + 5] + src[sIdx + 6 * dataW + 5]) + 2./1. * ( - src[sIdx + 0 * dataW + 1] + src[sIdx + 6 * dataW + 1]) + 4./1. * ( + src[sIdx + 0 * dataW + 2] - src[sIdx + 6 * dataW + 2]) + 5./1. * ( - src[sIdx + 0 * dataW + 4] + src[sIdx + 6 * dataW + 4]) + 21./2. * ( + src[sIdx + 2 * dataW + 1] - src[sIdx + 4 * dataW + 1]) + 1./1. * ( + src[sIdx + 0 * dataW + 6] - src[sIdx + 6 * dataW + 6]) + 105./8. * ( - src[sIdx + 2 * dataW + 3] + src[sIdx + 4 * dataW + 3]) + 105./4. * ( + src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 4]) + 21./1. * ( - src[sIdx + 2 * dataW + 2] + src[sIdx + 4 * dataW + 2]) + 5./2. * ( + src[sIdx + 0 * dataW + 3] - src[sIdx + 6 * dataW + 3]) + 21./4. * ( - src[sIdx + 2 * dataW + 6] + src[sIdx + 4 * dataW + 6]) + 21./8. * ( + src[sIdx + 2 * dataW + 5] - src[sIdx + 4 * dataW + 5]);
        dst[bIdx + 7 * gap] = + 21./4. * ( + src[sIdx + 0 * dataW + 3] - src[sIdx + 0 * dataW + 5] + src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 7] - src[sIdx + 4 * dataW + 1] + src[sIdx + 4 * dataW + 7] - src[sIdx + 6 * dataW + 3] + src[sIdx + 6 * dataW + 5]) + 1./1. * ( - src[sIdx + 0 * dataW + 1] + src[sIdx + 0 * dataW + 7] + src[sIdx + 6 * dataW + 1] - src[sIdx + 6 * dataW + 7]) + 441./16. * ( - src[sIdx + 2 * dataW + 3] + src[sIdx + 2 * dataW + 5] + src[sIdx + 4 * dataW + 3] - src[sIdx + 4 * dataW + 5]);
        dst[bIdx + 8 * gap] = + 21./4. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 4] - src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 4]) + 1./1. * ( + src[sIdx + 1 * dataW + 0] - src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 0] - src[sIdx + 2 * dataW + 6] + src[sIdx + 5 * dataW + 0] - src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 0] - src[sIdx + 6 * dataW + 6]) + 357./16. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 4]) + 17./4. * ( - src[sIdx + 3 * dataW + 0] + src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 0] + src[sIdx + 4 * dataW + 6]);
        dst[bIdx + 9 * gap] = + 289./16. * ( + src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 1./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 5] + src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 5] + src[sIdx + 2 * dataW + 6] + src[sIdx + 5 * dataW + 1] + src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 5] + src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 17./4. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4] - src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 5] - src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 5] - src[sIdx + 4 * dataW + 6] - src[sIdx + 5 * dataW + 3] - src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4]);
        dst[bIdx + 10 * gap] = + 289./16. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 1./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 5] + src[sIdx + 1 * dataW + 6] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 5] + src[sIdx + 2 * dataW + 6] - src[sIdx + 5 * dataW + 1] + src[sIdx + 5 * dataW + 2] - src[sIdx + 5 * dataW + 5] + src[sIdx + 5 * dataW + 6] - src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 2] - src[sIdx + 6 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 17./4. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4] + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 5] - src[sIdx + 3 * dataW + 6] + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 5] - src[sIdx + 4 * dataW + 6] + src[sIdx + 5 * dataW + 3] - src[sIdx + 5 * dataW + 4] + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4]);
        dst[bIdx + 11 * gap] = + 17./8. * ( - src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 1]) + 1./2. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 1] + src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 1]) + 2./1. * ( + src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 5] + src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 5]) + 1./4. * ( + src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 2] + src[sIdx + 5 * dataW + 2] + src[sIdx + 6 * dataW + 2]) + 1./1. * ( + src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 6] + src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 6]) + 5./4. * ( - src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 4] - src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 4]) + 17./2. * ( - src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 5]) + 17./16. * ( - src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 17./4. * ( - src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 6]) + 5./2. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 3] - src[sIdx + 5 * dataW + 3] - src[sIdx + 6 * dataW + 3]) + 85./16. * ( + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( + src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 3]);
        dst[bIdx + 12 * gap] = + 17./8. * ( + src[sIdx + 3 * dataW + 1] + src[sIdx + 4 * dataW + 1]) + 1./2. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 2 * dataW + 1] - src[sIdx + 5 * dataW + 1] - src[sIdx + 6 * dataW + 1]) + 2./1. * ( - src[sIdx + 1 * dataW + 5] - src[sIdx + 2 * dataW + 5] - src[sIdx + 5 * dataW + 5] - src[sIdx + 6 * dataW + 5]) + 1./4. * ( + src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 2] + src[sIdx + 5 * dataW + 2] + src[sIdx + 6 * dataW + 2]) + 1./1. * ( + src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 6] + src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 6]) + 5./4. * ( - src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 4] - src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 4]) + 17./2. * ( + src[sIdx + 3 * dataW + 5] + src[sIdx + 4 * dataW + 5]) + 17./16. * ( - src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 17./4. * ( - src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 6]) + 5./2. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 2 * dataW + 3] + src[sIdx + 5 * dataW + 3] + src[sIdx + 6 * dataW + 3]) + 85./16. * ( + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( - src[sIdx + 3 * dataW + 3] - src[sIdx + 4 * dataW + 3]);
        dst[bIdx + 13 * gap] = + 17./8. * ( - src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 5]) + 1./2. * ( + src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 5] + src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 5]) + 2./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 1] + src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 1]) + 4./1. * ( + src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 2] + src[sIdx + 5 * dataW + 2] + src[sIdx + 6 * dataW + 2]) + 5./1. * ( - src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 4] - src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 4]) + 1./1. * ( + src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 6] + src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 6]) + 17./2. * ( - src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 1]) + 17./1. * ( - src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 17./4. * ( - src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 6]) + 5./2. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 3] - src[sIdx + 5 * dataW + 3] - src[sIdx + 6 * dataW + 3]) + 85./4. * ( + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( + src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 3]);
        dst[bIdx + 14 * gap] = + 17./8. * ( + src[sIdx + 3 * dataW + 5] + src[sIdx + 4 * dataW + 5]) + 1./2. * ( - src[sIdx + 1 * dataW + 5] - src[sIdx + 2 * dataW + 5] - src[sIdx + 5 * dataW + 5] - src[sIdx + 6 * dataW + 5]) + 2./1. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 2 * dataW + 1] - src[sIdx + 5 * dataW + 1] - src[sIdx + 6 * dataW + 1]) + 4./1. * ( + src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 2] + src[sIdx + 5 * dataW + 2] + src[sIdx + 6 * dataW + 2]) + 5./1. * ( - src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 4] - src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 4]) + 1./1. * ( + src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 6] + src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 6]) + 17./2. * ( + src[sIdx + 3 * dataW + 1] + src[sIdx + 4 * dataW + 1]) + 17./1. * ( - src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 17./4. * ( - src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 6]) + 5./2. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 2 * dataW + 3] + src[sIdx + 5 * dataW + 3] + src[sIdx + 6 * dataW + 3]) + 85./4. * ( + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( - src[sIdx + 3 * dataW + 3] - src[sIdx + 4 * dataW + 3]);
        dst[bIdx + 15 * gap] = + 21./4. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 5] + src[sIdx + 5 * dataW + 3] - src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 5]) + 1./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 7] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 7] - src[sIdx + 5 * dataW + 1] + src[sIdx + 5 * dataW + 7] - src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 7]) + 357./16. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 5]) + 17./4. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 7] + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 7]);
        dst[bIdx + 16 * gap] = + 21./4. * ( + src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 4] + src[sIdx + 5 * dataW + 2] - src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 4]) + 1./1. * ( - src[sIdx + 1 * dataW + 0] + src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 0] - src[sIdx + 2 * dataW + 6] - src[sIdx + 5 * dataW + 0] + src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 0] - src[sIdx + 6 * dataW + 6]) + 357./16. * ( - src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 4]) + 17./4. * ( + src[sIdx + 3 * dataW + 0] - src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 0] + src[sIdx + 4 * dataW + 6]);
        dst[bIdx + 17 * gap] = + 289./16. * ( - src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 1./1. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 5] - src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 5] + src[sIdx + 2 * dataW + 6] - src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 2] - src[sIdx + 5 * dataW + 5] - src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 17./4. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4] + src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 5] + src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 5] - src[sIdx + 4 * dataW + 6] + src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4]);
        dst[bIdx + 18 * gap] = + 289./16. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 1./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 5] - src[sIdx + 1 * dataW + 6] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 5] + src[sIdx + 2 * dataW + 6] + src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 5] - src[sIdx + 5 * dataW + 6] - src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 2] - src[sIdx + 6 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 17./4. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4] - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 5] + src[sIdx + 3 * dataW + 6] + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 5] - src[sIdx + 4 * dataW + 6] - src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4] + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4]);
        dst[bIdx + 19 * gap] = + 17./8. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 1]) + 1./2. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 1] - src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 1]) + 2./1. * ( - src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 5] - src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 5]) + 1./4. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 2] - src[sIdx + 5 * dataW + 2] + src[sIdx + 6 * dataW + 2]) + 1./1. * ( - src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 6] - src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 6]) + 5./4. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 4] + src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 4]) + 17./2. * ( + src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 5]) + 17./16. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 17./4. * ( + src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 6]) + 5./2. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 3] + src[sIdx + 5 * dataW + 3] - src[sIdx + 6 * dataW + 3]) + 85./16. * ( - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 3]);
        dst[bIdx + 20 * gap] = + 17./8. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 4 * dataW + 1]) + 1./2. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 2 * dataW + 1] + src[sIdx + 5 * dataW + 1] - src[sIdx + 6 * dataW + 1]) + 2./1. * ( + src[sIdx + 1 * dataW + 5] - src[sIdx + 2 * dataW + 5] + src[sIdx + 5 * dataW + 5] - src[sIdx + 6 * dataW + 5]) + 1./4. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 2] - src[sIdx + 5 * dataW + 2] + src[sIdx + 6 * dataW + 2]) + 1./1. * ( - src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 6] - src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 6]) + 5./4. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 4] + src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 4]) + 17./2. * ( - src[sIdx + 3 * dataW + 5] + src[sIdx + 4 * dataW + 5]) + 17./16. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 17./4. * ( + src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 6]) + 5./2. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 2 * dataW + 3] - src[sIdx + 5 * dataW + 3] + src[sIdx + 6 * dataW + 3]) + 85./16. * ( - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 4 * dataW + 3]);
        dst[bIdx + 21 * gap] = + 17./8. * ( + src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 5]) + 1./2. * ( - src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 5] - src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 5]) + 2./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 1] - src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 1]) + 4./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 2] - src[sIdx + 5 * dataW + 2] + src[sIdx + 6 * dataW + 2]) + 5./1. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 4] + src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 4]) + 1./1. * ( - src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 6] - src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 6]) + 17./2. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 1]) + 17./1. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 17./4. * ( + src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 6]) + 5./2. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 3] + src[sIdx + 5 * dataW + 3] - src[sIdx + 6 * dataW + 3]) + 85./4. * ( - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 3]);
        dst[bIdx + 22 * gap] = + 17./8. * ( - src[sIdx + 3 * dataW + 5] + src[sIdx + 4 * dataW + 5]) + 1./2. * ( + src[sIdx + 1 * dataW + 5] - src[sIdx + 2 * dataW + 5] + src[sIdx + 5 * dataW + 5] - src[sIdx + 6 * dataW + 5]) + 2./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 2 * dataW + 1] + src[sIdx + 5 * dataW + 1] - src[sIdx + 6 * dataW + 1]) + 4./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 2] - src[sIdx + 5 * dataW + 2] + src[sIdx + 6 * dataW + 2]) + 5./1. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 4] + src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 4]) + 1./1. * ( - src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 6] - src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 6]) + 17./2. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 4 * dataW + 1]) + 17./1. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 17./4. * ( + src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 6]) + 5./2. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 2 * dataW + 3] - src[sIdx + 5 * dataW + 3] + src[sIdx + 6 * dataW + 3]) + 85./4. * ( - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 4 * dataW + 3]);
        dst[bIdx + 23 * gap] = + 21./4. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 5] - src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 5]) + 1./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 7] - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 7] + src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 7] - src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 7]) + 357./16. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 5]) + 17./4. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 7] + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 7]);
        dst[bIdx + 24 * gap] = + 105./16. * ( + src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 4]) + 1./2. * ( + src[sIdx + 1 * dataW + 0] - src[sIdx + 1 * dataW + 6]) + 2./1. * ( + src[sIdx + 5 * dataW + 0] - src[sIdx + 5 * dataW + 6]) + 1./4. * ( + src[sIdx + 2 * dataW + 0] - src[sIdx + 2 * dataW + 6]) + 21./2. * ( - src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 4]) + 1./1. * ( + src[sIdx + 6 * dataW + 0] - src[sIdx + 6 * dataW + 6]) + 105./8. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 4]) + 5./4. * ( - src[sIdx + 4 * dataW + 0] + src[sIdx + 4 * dataW + 6]) + 5./2. * ( - src[sIdx + 3 * dataW + 0] + src[sIdx + 3 * dataW + 6]) + 21./4. * ( - src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 4]) + 21./16. * ( - src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 4]) + 21./8. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 4]);
        dst[bIdx + 25 * gap] = + 17./8. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 4]) + 1./2. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 5] + src[sIdx + 1 * dataW + 6]) + 2./1. * ( + src[sIdx + 5 * dataW + 1] + src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 5] + src[sIdx + 5 * dataW + 6]) + 1./4. * ( + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 1./1. * ( + src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 17./16. * ( - src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4]) + 17./2. * ( - src[sIdx + 5 * dataW + 3] - src[sIdx + 5 * dataW + 4]) + 5./4. * ( - src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 17./4. * ( - src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 5./2. * ( - src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 5] - src[sIdx + 3 * dataW + 6]) + 85./16. * ( + src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( + src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 4]);
        dst[bIdx + 26 * gap] = + 17./8. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 4]) + 1./2. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 5] + src[sIdx + 1 * dataW + 6]) + 2./1. * ( - src[sIdx + 5 * dataW + 1] + src[sIdx + 5 * dataW + 2] - src[sIdx + 5 * dataW + 5] + src[sIdx + 5 * dataW + 6]) + 1./4. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 1./1. * ( - src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 2] - src[sIdx + 6 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 17./16. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4]) + 17./2. * ( + src[sIdx + 5 * dataW + 3] - src[sIdx + 5 * dataW + 4]) + 5./4. * ( + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 17./4. * ( + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 5./2. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 5] - src[sIdx + 3 * dataW + 6]) + 85./16. * ( - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 4]);
        dst[bIdx + 27 * gap] = + 1./2. * ( + src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 5] + src[sIdx + 5 * dataW + 2] + src[sIdx + 6 * dataW + 1]) + 2./1. * ( + src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 5]) + 1./4. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 6] + src[sIdx + 6 * dataW + 2]) + 5./1. * ( - src[sIdx + 3 * dataW + 5] - src[sIdx + 5 * dataW + 3]) + 1./1. * ( + src[sIdx + 1 * dataW + 5] + src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 6]) + 5./4. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 6] - src[sIdx + 6 * dataW + 4]) + 5./8. * ( - src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 3] - src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 1]) + 25./16. * ( + src[sIdx + 4 * dataW + 4]) + 1./8. * ( + src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 1]) + 25./8. * ( + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 3]) + 5./16. * ( - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2]) + 5./2. * ( - src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 5] - src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 3]) + 1./16. * ( + src[sIdx + 2 * dataW + 2]) + 4./1. * ( + src[sIdx + 5 * dataW + 5]) + 25./4. * ( + src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 28 * gap] = + 1./2. * ( + src[sIdx + 1 * dataW + 6] - src[sIdx + 2 * dataW + 5] + src[sIdx + 5 * dataW + 2] - src[sIdx + 6 * dataW + 1]) + 2./1. * ( + src[sIdx + 5 * dataW + 6] - src[sIdx + 6 * dataW + 5]) + 1./4. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 6] + src[sIdx + 6 * dataW + 2]) + 5./1. * ( + src[sIdx + 3 * dataW + 5] + src[sIdx + 5 * dataW + 3]) + 1./1. * ( - src[sIdx + 1 * dataW + 5] - src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 6]) + 5./4. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 6] - src[sIdx + 6 * dataW + 4]) + 5./8. * ( - src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 3] - src[sIdx + 3 * dataW + 2] + src[sIdx + 4 * dataW + 1]) + 25./16. * ( + src[sIdx + 4 * dataW + 4]) + 1./8. * ( + src[sIdx + 1 * dataW + 2] - src[sIdx + 2 * dataW + 1]) + 25./8. * ( + src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 3]) + 5./16. * ( - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2]) + 5./2. * ( - src[sIdx + 3 * dataW + 6] + src[sIdx + 4 * dataW + 5] - src[sIdx + 5 * dataW + 4] + src[sIdx + 6 * dataW + 3]) + 1./16. * ( + src[sIdx + 2 * dataW + 2]) + 4./1. * ( - src[sIdx + 5 * dataW + 5]) + 25./4. * ( - src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 29 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 2] + src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 2./1. * ( + src[sIdx + 1 * dataW + 2] + src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 1]) + 1./4. * ( + src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 5./1. * ( - src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 2] - src[sIdx + 5 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 1./2. * ( + src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 1] + src[sIdx + 6 * dataW + 5]) + 5./4. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 4] - src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 10./1. * ( - src[sIdx + 3 * dataW + 2] - src[sIdx + 5 * dataW + 4]) + 5./8. * ( - src[sIdx + 2 * dataW + 3] - src[sIdx + 4 * dataW + 5]) + 1./8. * ( + src[sIdx + 2 * dataW + 5]) + 8./1. * ( + src[sIdx + 5 * dataW + 2]) + 25./2. * ( + src[sIdx + 3 * dataW + 4]) + 5./2. * ( - src[sIdx + 1 * dataW + 4] - src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 1] - src[sIdx + 6 * dataW + 3]) + 25./8. * ( + src[sIdx + 4 * dataW + 3]) + 4./1. * ( + src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 2]) + 25./4. * ( + src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 4]);
        dst[bIdx + 30 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 2] - src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 2./1. * ( + src[sIdx + 1 * dataW + 2] + src[sIdx + 5 * dataW + 6] - src[sIdx + 6 * dataW + 1]) + 1./4. * ( - src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 5./1. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 2] + src[sIdx + 5 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 1./2. * ( + src[sIdx + 1 * dataW + 6] - src[sIdx + 2 * dataW + 1] - src[sIdx + 6 * dataW + 5]) + 5./4. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 4] + src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 10./1. * ( - src[sIdx + 3 * dataW + 2] - src[sIdx + 5 * dataW + 4]) + 5./8. * ( + src[sIdx + 2 * dataW + 3] + src[sIdx + 4 * dataW + 5]) + 1./8. * ( - src[sIdx + 2 * dataW + 5]) + 8./1. * ( + src[sIdx + 5 * dataW + 2]) + 25./2. * ( + src[sIdx + 3 * dataW + 4]) + 5./2. * ( - src[sIdx + 1 * dataW + 4] - src[sIdx + 3 * dataW + 6] + src[sIdx + 4 * dataW + 1] + src[sIdx + 6 * dataW + 3]) + 25./8. * ( - src[sIdx + 4 * dataW + 3]) + 4./1. * ( - src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 2]) + 25./4. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 4]);
        dst[bIdx + 31 * gap] = + 105./16. * ( - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 5]) + 1./2. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 7]) + 2./1. * ( - src[sIdx + 5 * dataW + 1] + src[sIdx + 5 * dataW + 7]) + 1./4. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 7]) + 21./2. * ( + src[sIdx + 5 * dataW + 3] - src[sIdx + 5 * dataW + 5]) + 1./1. * ( - src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 7]) + 105./8. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 5]) + 5./4. * ( + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 7]) + 5./2. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 7]) + 21./4. * ( + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 5]) + 21./16. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 5]) + 21./8. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 5]);
        dst[bIdx + 32 * gap] = + 105./16. * ( + src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 4]) + 1./2. * ( - src[sIdx + 1 * dataW + 0] + src[sIdx + 1 * dataW + 6]) + 2./1. * ( - src[sIdx + 5 * dataW + 0] + src[sIdx + 5 * dataW + 6]) + 1./4. * ( + src[sIdx + 2 * dataW + 0] - src[sIdx + 2 * dataW + 6]) + 21./2. * ( + src[sIdx + 5 * dataW + 2] - src[sIdx + 5 * dataW + 4]) + 1./1. * ( + src[sIdx + 6 * dataW + 0] - src[sIdx + 6 * dataW + 6]) + 105./8. * ( - src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 4]) + 5./4. * ( - src[sIdx + 4 * dataW + 0] + src[sIdx + 4 * dataW + 6]) + 5./2. * ( + src[sIdx + 3 * dataW + 0] - src[sIdx + 3 * dataW + 6]) + 21./4. * ( - src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 4]) + 21./16. * ( - src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 4]) + 21./8. * ( + src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 4]);
        dst[bIdx + 33 * gap] = + 17./8. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4]) + 1./2. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 5] - src[sIdx + 1 * dataW + 6]) + 2./1. * ( - src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 2] - src[sIdx + 5 * dataW + 5] - src[sIdx + 5 * dataW + 6]) + 1./4. * ( + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 1./1. * ( + src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 17./16. * ( - src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4]) + 17./2. * ( + src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4]) + 5./4. * ( - src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 17./4. * ( - src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 5./2. * ( + src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 5] + src[sIdx + 3 * dataW + 6]) + 85./16. * ( + src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( - src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4]);
        dst[bIdx + 34 * gap] = + 17./8. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4]) + 1./2. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 5] - src[sIdx + 1 * dataW + 6]) + 2./1. * ( + src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 5] - src[sIdx + 5 * dataW + 6]) + 1./4. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 1./1. * ( - src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 2] - src[sIdx + 6 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 17./16. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4]) + 17./2. * ( - src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4]) + 5./4. * ( + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 17./4. * ( + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 5./2. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 5] + src[sIdx + 3 * dataW + 6]) + 85./16. * ( - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4]);
        dst[bIdx + 35 * gap] = + 1./2. * ( - src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 5] - src[sIdx + 5 * dataW + 2] + src[sIdx + 6 * dataW + 1]) + 2./1. * ( - src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 5]) + 1./4. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 6] + src[sIdx + 6 * dataW + 2]) + 5./1. * ( + src[sIdx + 3 * dataW + 5] + src[sIdx + 5 * dataW + 3]) + 1./1. * ( - src[sIdx + 1 * dataW + 5] - src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 6]) + 5./4. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 6] - src[sIdx + 6 * dataW + 4]) + 5./8. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 3] + src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 1]) + 25./16. * ( + src[sIdx + 4 * dataW + 4]) + 1./8. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 1]) + 25./8. * ( - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 3]) + 5./16. * ( - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2]) + 5./2. * ( + src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 5] + src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 3]) + 1./16. * ( + src[sIdx + 2 * dataW + 2]) + 4./1. * ( - src[sIdx + 5 * dataW + 5]) + 25./4. * ( - src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 36 * gap] = + 1./2. * ( - src[sIdx + 1 * dataW + 6] - src[sIdx + 2 * dataW + 5] - src[sIdx + 5 * dataW + 2] - src[sIdx + 6 * dataW + 1]) + 2./1. * ( - src[sIdx + 5 * dataW + 6] - src[sIdx + 6 * dataW + 5]) + 1./4. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 6] + src[sIdx + 6 * dataW + 2]) + 5./1. * ( - src[sIdx + 3 * dataW + 5] - src[sIdx + 5 * dataW + 3]) + 1./1. * ( + src[sIdx + 1 * dataW + 5] + src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 6]) + 5./4. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 6] - src[sIdx + 6 * dataW + 4]) + 5./8. * ( + src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 3] + src[sIdx + 3 * dataW + 2] + src[sIdx + 4 * dataW + 1]) + 25./16. * ( + src[sIdx + 4 * dataW + 4]) + 1./8. * ( - src[sIdx + 1 * dataW + 2] - src[sIdx + 2 * dataW + 1]) + 25./8. * ( - src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 3]) + 5./16. * ( - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2]) + 5./2. * ( + src[sIdx + 3 * dataW + 6] + src[sIdx + 4 * dataW + 5] + src[sIdx + 5 * dataW + 4] + src[sIdx + 6 * dataW + 3]) + 1./16. * ( + src[sIdx + 2 * dataW + 2]) + 4./1. * ( + src[sIdx + 5 * dataW + 5]) + 25./4. * ( + src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 37 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 2] - src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 2./1. * ( - src[sIdx + 1 * dataW + 2] - src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 1]) + 1./4. * ( - src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 5./1. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 2] + src[sIdx + 5 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 1./2. * ( - src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 1] + src[sIdx + 6 * dataW + 5]) + 5./4. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 4] + src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 10./1. * ( + src[sIdx + 3 * dataW + 2] + src[sIdx + 5 * dataW + 4]) + 5./8. * ( - src[sIdx + 2 * dataW + 3] - src[sIdx + 4 * dataW + 5]) + 1./8. * ( + src[sIdx + 2 * dataW + 5]) + 8./1. * ( - src[sIdx + 5 * dataW + 2]) + 25./2. * ( - src[sIdx + 3 * dataW + 4]) + 5./2. * ( + src[sIdx + 1 * dataW + 4] + src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 1] - src[sIdx + 6 * dataW + 3]) + 25./8. * ( + src[sIdx + 4 * dataW + 3]) + 4./1. * ( - src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 2]) + 25./4. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 4]);
        dst[bIdx + 38 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 2] + src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 2./1. * ( - src[sIdx + 1 * dataW + 2] - src[sIdx + 5 * dataW + 6] - src[sIdx + 6 * dataW + 1]) + 1./4. * ( + src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 5./1. * ( - src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 2] - src[sIdx + 5 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 1./2. * ( - src[sIdx + 1 * dataW + 6] - src[sIdx + 2 * dataW + 1] - src[sIdx + 6 * dataW + 5]) + 5./4. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 4] - src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 10./1. * ( + src[sIdx + 3 * dataW + 2] + src[sIdx + 5 * dataW + 4]) + 5./8. * ( + src[sIdx + 2 * dataW + 3] + src[sIdx + 4 * dataW + 5]) + 1./8. * ( - src[sIdx + 2 * dataW + 5]) + 8./1. * ( - src[sIdx + 5 * dataW + 2]) + 25./2. * ( - src[sIdx + 3 * dataW + 4]) + 5./2. * ( + src[sIdx + 1 * dataW + 4] + src[sIdx + 3 * dataW + 6] + src[sIdx + 4 * dataW + 1] + src[sIdx + 6 * dataW + 3]) + 25./8. * ( - src[sIdx + 4 * dataW + 3]) + 4./1. * ( + src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 2]) + 25./4. * ( + src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 4]);
        dst[bIdx + 39 * gap] = + 105./16. * ( - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 5]) + 1./2. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 7]) + 2./1. * ( + src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 7]) + 1./4. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 7]) + 21./2. * ( - src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 5]) + 1./1. * ( - src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 7]) + 105./8. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 5]) + 5./4. * ( + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 7]) + 5./2. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 7]) + 21./4. * ( + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 5]) + 21./16. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 5]) + 21./8. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 5]);
        dst[bIdx + 40 * gap] = + 1./2. * ( + src[sIdx + 5 * dataW + 0] - src[sIdx + 5 * dataW + 6]) + 2./1. * ( + src[sIdx + 1 * dataW + 0] - src[sIdx + 1 * dataW + 6]) + 4./1. * ( + src[sIdx + 2 * dataW + 0] - src[sIdx + 2 * dataW + 6]) + 5./1. * ( - src[sIdx + 4 * dataW + 0] + src[sIdx + 4 * dataW + 6]) + 21./2. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 4]) + 1./1. * ( + src[sIdx + 6 * dataW + 0] - src[sIdx + 6 * dataW + 6]) + 105./8. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 4]) + 105./4. * ( + src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 4]) + 21./1. * ( - src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 4]) + 5./2. * ( - src[sIdx + 3 * dataW + 0] + src[sIdx + 3 * dataW + 6]) + 21./4. * ( - src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 4]) + 21./8. * ( - src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 4]);
        dst[bIdx + 41 * gap] = + 17./8. * ( - src[sIdx + 5 * dataW + 3] - src[sIdx + 5 * dataW + 4]) + 1./2. * ( + src[sIdx + 5 * dataW + 1] + src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 5] + src[sIdx + 5 * dataW + 6]) + 2./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 5] + src[sIdx + 1 * dataW + 6]) + 4./1. * ( + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 5./1. * ( - src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 1./1. * ( + src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 17./2. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 4]) + 17./1. * ( - src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4]) + 17./4. * ( - src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 5./2. * ( - src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 5] - src[sIdx + 3 * dataW + 6]) + 85./4. * ( + src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( + src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 4]);
        dst[bIdx + 42 * gap] = + 17./8. * ( + src[sIdx + 5 * dataW + 3] - src[sIdx + 5 * dataW + 4]) + 1./2. * ( - src[sIdx + 5 * dataW + 1] + src[sIdx + 5 * dataW + 2] - src[sIdx + 5 * dataW + 5] + src[sIdx + 5 * dataW + 6]) + 2./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 5] + src[sIdx + 1 * dataW + 6]) + 4./1. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 5./1. * ( + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 1./1. * ( - src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 2] - src[sIdx + 6 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 17./2. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 4]) + 17./1. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4]) + 17./4. * ( + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 5./2. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 5] - src[sIdx + 3 * dataW + 6]) + 85./4. * ( - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 4]);
        dst[bIdx + 43 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 2] + src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 2./1. * ( + src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 1] + src[sIdx + 6 * dataW + 5]) + 4./1. * ( + src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 5./1. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 4] - src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 1./2. * ( + src[sIdx + 1 * dataW + 2] + src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 1]) + 8./1. * ( + src[sIdx + 2 * dataW + 5]) + 10./1. * ( - src[sIdx + 2 * dataW + 3] - src[sIdx + 4 * dataW + 5]) + 5./8. * ( - src[sIdx + 3 * dataW + 2] - src[sIdx + 5 * dataW + 4]) + 1./8. * ( + src[sIdx + 5 * dataW + 2]) + 5./4. * ( - src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 2] - src[sIdx + 5 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 25./2. * ( + src[sIdx + 4 * dataW + 3]) + 5./2. * ( - src[sIdx + 1 * dataW + 4] - src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 1] - src[sIdx + 6 * dataW + 3]) + 25./8. * ( + src[sIdx + 3 * dataW + 4]) + 1./4. * ( + src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 2]) + 25./4. * ( + src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 4]);
        dst[bIdx + 44 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 2] - src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 2./1. * ( + src[sIdx + 1 * dataW + 6] - src[sIdx + 2 * dataW + 1] - src[sIdx + 6 * dataW + 5]) + 4./1. * ( - src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 5./1. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 4] + src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 1./2. * ( + src[sIdx + 1 * dataW + 2] + src[sIdx + 5 * dataW + 6] - src[sIdx + 6 * dataW + 1]) + 8./1. * ( - src[sIdx + 2 * dataW + 5]) + 10./1. * ( + src[sIdx + 2 * dataW + 3] + src[sIdx + 4 * dataW + 5]) + 5./8. * ( - src[sIdx + 3 * dataW + 2] - src[sIdx + 5 * dataW + 4]) + 1./8. * ( + src[sIdx + 5 * dataW + 2]) + 5./4. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 2] + src[sIdx + 5 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 25./2. * ( - src[sIdx + 4 * dataW + 3]) + 5./2. * ( - src[sIdx + 1 * dataW + 4] - src[sIdx + 3 * dataW + 6] + src[sIdx + 4 * dataW + 1] + src[sIdx + 6 * dataW + 3]) + 25./8. * ( + src[sIdx + 3 * dataW + 4]) + 1./4. * ( - src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 2]) + 25./4. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 4]);
        dst[bIdx + 45 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 5] + src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 6]) + 2./1. * ( + src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 5] + src[sIdx + 5 * dataW + 2] + src[sIdx + 6 * dataW + 1]) + 4./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 6] + src[sIdx + 6 * dataW + 2]) + 5./1. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 6] - src[sIdx + 6 * dataW + 4]) + 1./2. * ( + src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 5]) + 8./1. * ( + src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 1]) + 10./1. * ( - src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 3] - src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 1]) + 16./1. * ( + src[sIdx + 2 * dataW + 2]) + 5./4. * ( - src[sIdx + 3 * dataW + 5] - src[sIdx + 5 * dataW + 3]) + 25./2. * ( + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 3]) + 20./1. * ( - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2]) + 5./2. * ( - src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 5] - src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 3]) + 25./1. * ( + src[sIdx + 4 * dataW + 4]) + 1./4. * ( + src[sIdx + 5 * dataW + 5]) + 25./4. * ( + src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 46 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 5] - src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 6]) + 2./1. * ( + src[sIdx + 1 * dataW + 6] - src[sIdx + 2 * dataW + 5] + src[sIdx + 5 * dataW + 2] - src[sIdx + 6 * dataW + 1]) + 4./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 6] + src[sIdx + 6 * dataW + 2]) + 5./1. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 6] - src[sIdx + 6 * dataW + 4]) + 1./2. * ( + src[sIdx + 5 * dataW + 6] - src[sIdx + 6 * dataW + 5]) + 8./1. * ( + src[sIdx + 1 * dataW + 2] - src[sIdx + 2 * dataW + 1]) + 10./1. * ( - src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 3] - src[sIdx + 3 * dataW + 2] + src[sIdx + 4 * dataW + 1]) + 16./1. * ( + src[sIdx + 2 * dataW + 2]) + 5./4. * ( + src[sIdx + 3 * dataW + 5] + src[sIdx + 5 * dataW + 3]) + 25./2. * ( + src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 3]) + 20./1. * ( - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2]) + 5./2. * ( - src[sIdx + 3 * dataW + 6] + src[sIdx + 4 * dataW + 5] - src[sIdx + 5 * dataW + 4] + src[sIdx + 6 * dataW + 3]) + 25./1. * ( + src[sIdx + 4 * dataW + 4]) + 1./4. * ( - src[sIdx + 5 * dataW + 5]) + 25./4. * ( - src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 47 * gap] = + 1./2. * ( - src[sIdx + 5 * dataW + 1] + src[sIdx + 5 * dataW + 7]) + 2./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 7]) + 4./1. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 7]) + 5./1. * ( + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 7]) + 21./2. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 5]) + 1./1. * ( - src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 7]) + 105./8. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 5]) + 105./4. * ( - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 5]) + 21./1. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 5]) + 5./2. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 7]) + 21./4. * ( + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 5]) + 21./8. * ( + src[sIdx + 5 * dataW + 3] - src[sIdx + 5 * dataW + 5]);
        dst[bIdx + 48 * gap] = + 1./2. * ( - src[sIdx + 5 * dataW + 0] + src[sIdx + 5 * dataW + 6]) + 2./1. * ( - src[sIdx + 1 * dataW + 0] + src[sIdx + 1 * dataW + 6]) + 4./1. * ( + src[sIdx + 2 * dataW + 0] - src[sIdx + 2 * dataW + 6]) + 5./1. * ( - src[sIdx + 4 * dataW + 0] + src[sIdx + 4 * dataW + 6]) + 21./2. * ( + src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 4]) + 1./1. * ( + src[sIdx + 6 * dataW + 0] - src[sIdx + 6 * dataW + 6]) + 105./8. * ( - src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 4]) + 105./4. * ( + src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 4]) + 21./1. * ( - src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 4]) + 5./2. * ( + src[sIdx + 3 * dataW + 0] - src[sIdx + 3 * dataW + 6]) + 21./4. * ( - src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 4]) + 21./8. * ( + src[sIdx + 5 * dataW + 2] - src[sIdx + 5 * dataW + 4]);
        dst[bIdx + 49 * gap] = + 17./8. * ( + src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4]) + 1./2. * ( - src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 2] - src[sIdx + 5 * dataW + 5] - src[sIdx + 5 * dataW + 6]) + 2./1. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 5] - src[sIdx + 1 * dataW + 6]) + 4./1. * ( + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 5./1. * ( - src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 1./1. * ( + src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 17./2. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4]) + 17./1. * ( - src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4]) + 17./4. * ( - src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 5./2. * ( + src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 5] + src[sIdx + 3 * dataW + 6]) + 85./4. * ( + src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( - src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4]);
        dst[bIdx + 50 * gap] = + 17./8. * ( - src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4]) + 1./2. * ( + src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 5] - src[sIdx + 5 * dataW + 6]) + 2./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 5] - src[sIdx + 1 * dataW + 6]) + 4./1. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 5./1. * ( + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 1./1. * ( - src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 2] - src[sIdx + 6 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 17./2. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4]) + 17./1. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4]) + 17./4. * ( + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 5./2. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 5] + src[sIdx + 3 * dataW + 6]) + 85./4. * ( - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4]) + 85./8. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4]);
        dst[bIdx + 51 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 2] - src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 2./1. * ( - src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 1] + src[sIdx + 6 * dataW + 5]) + 4./1. * ( - src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 5./1. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 4] + src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 1./2. * ( - src[sIdx + 1 * dataW + 2] - src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 1]) + 8./1. * ( + src[sIdx + 2 * dataW + 5]) + 10./1. * ( - src[sIdx + 2 * dataW + 3] - src[sIdx + 4 * dataW + 5]) + 5./8. * ( + src[sIdx + 3 * dataW + 2] + src[sIdx + 5 * dataW + 4]) + 1./8. * ( - src[sIdx + 5 * dataW + 2]) + 5./4. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 2] + src[sIdx + 5 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 25./2. * ( + src[sIdx + 4 * dataW + 3]) + 5./2. * ( + src[sIdx + 1 * dataW + 4] + src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 1] - src[sIdx + 6 * dataW + 3]) + 25./8. * ( - src[sIdx + 3 * dataW + 4]) + 1./4. * ( - src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 2]) + 25./4. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 4]);
        dst[bIdx + 52 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 2] + src[sIdx + 5 * dataW + 5] + src[sIdx + 6 * dataW + 6]) + 2./1. * ( - src[sIdx + 1 * dataW + 6] - src[sIdx + 2 * dataW + 1] - src[sIdx + 6 * dataW + 5]) + 4./1. * ( + src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 6]) + 5./1. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 2 * dataW + 4] - src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 6]) + 1./2. * ( - src[sIdx + 1 * dataW + 2] - src[sIdx + 5 * dataW + 6] - src[sIdx + 6 * dataW + 1]) + 8./1. * ( - src[sIdx + 2 * dataW + 5]) + 10./1. * ( + src[sIdx + 2 * dataW + 3] + src[sIdx + 4 * dataW + 5]) + 5./8. * ( + src[sIdx + 3 * dataW + 2] + src[sIdx + 5 * dataW + 4]) + 1./8. * ( - src[sIdx + 5 * dataW + 2]) + 5./4. * ( - src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 2] - src[sIdx + 5 * dataW + 3] - src[sIdx + 6 * dataW + 4]) + 25./2. * ( - src[sIdx + 4 * dataW + 3]) + 5./2. * ( + src[sIdx + 1 * dataW + 4] + src[sIdx + 3 * dataW + 6] + src[sIdx + 4 * dataW + 1] + src[sIdx + 6 * dataW + 3]) + 25./8. * ( - src[sIdx + 3 * dataW + 4]) + 1./4. * ( + src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 2]) + 25./4. * ( + src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 4]);
        dst[bIdx + 53 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 5] - src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 6]) + 2./1. * ( - src[sIdx + 1 * dataW + 6] + src[sIdx + 2 * dataW + 5] - src[sIdx + 5 * dataW + 2] + src[sIdx + 6 * dataW + 1]) + 4./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 6] + src[sIdx + 6 * dataW + 2]) + 5./1. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 6] - src[sIdx + 6 * dataW + 4]) + 1./2. * ( - src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 5]) + 8./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 1]) + 10./1. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 2 * dataW + 3] + src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 1]) + 16./1. * ( + src[sIdx + 2 * dataW + 2]) + 5./4. * ( + src[sIdx + 3 * dataW + 5] + src[sIdx + 5 * dataW + 3]) + 25./2. * ( - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 3]) + 20./1. * ( - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2]) + 5./2. * ( + src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 5] + src[sIdx + 5 * dataW + 4] - src[sIdx + 6 * dataW + 3]) + 25./1. * ( + src[sIdx + 4 * dataW + 4]) + 1./4. * ( - src[sIdx + 5 * dataW + 5]) + 25./4. * ( - src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 54 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 5] + src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 6]) + 2./1. * ( - src[sIdx + 1 * dataW + 6] - src[sIdx + 2 * dataW + 5] - src[sIdx + 5 * dataW + 2] - src[sIdx + 6 * dataW + 1]) + 4./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 6] + src[sIdx + 6 * dataW + 2]) + 5./1. * ( - src[sIdx + 1 * dataW + 3] - src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 6] - src[sIdx + 6 * dataW + 4]) + 1./2. * ( - src[sIdx + 5 * dataW + 6] - src[sIdx + 6 * dataW + 5]) + 8./1. * ( - src[sIdx + 1 * dataW + 2] - src[sIdx + 2 * dataW + 1]) + 10./1. * ( + src[sIdx + 1 * dataW + 4] + src[sIdx + 2 * dataW + 3] + src[sIdx + 3 * dataW + 2] + src[sIdx + 4 * dataW + 1]) + 16./1. * ( + src[sIdx + 2 * dataW + 2]) + 5./4. * ( - src[sIdx + 3 * dataW + 5] - src[sIdx + 5 * dataW + 3]) + 25./2. * ( - src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 3]) + 20./1. * ( - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2]) + 5./2. * ( + src[sIdx + 3 * dataW + 6] + src[sIdx + 4 * dataW + 5] + src[sIdx + 5 * dataW + 4] + src[sIdx + 6 * dataW + 3]) + 25./1. * ( + src[sIdx + 4 * dataW + 4]) + 1./4. * ( + src[sIdx + 5 * dataW + 5]) + 25./4. * ( + src[sIdx + 3 * dataW + 3]);
        dst[bIdx + 55 * gap] = + 1./2. * ( + src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 7]) + 2./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 7]) + 4./1. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 7]) + 5./1. * ( + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 7]) + 21./2. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 5]) + 1./1. * ( - src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 7]) + 105./8. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 5]) + 105./4. * ( - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 5]) + 21./1. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 5]) + 5./2. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 7]) + 21./4. * ( + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 5]) + 21./8. * ( - src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 5]);
        dst[bIdx + 56 * gap] = + 21./4. * ( + src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 4] + src[sIdx + 3 * dataW + 0] - src[sIdx + 3 * dataW + 6] - src[sIdx + 5 * dataW + 0] + src[sIdx + 5 * dataW + 6] - src[sIdx + 7 * dataW + 2] + src[sIdx + 7 * dataW + 4]) + 1./1. * ( - src[sIdx + 1 * dataW + 0] + src[sIdx + 1 * dataW + 6] + src[sIdx + 7 * dataW + 0] - src[sIdx + 7 * dataW + 6]) + 441./16. * ( - src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 4] + src[sIdx + 5 * dataW + 2] - src[sIdx + 5 * dataW + 4]);
        dst[bIdx + 57 * gap] = + 21./4. * ( + src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 5] + src[sIdx + 3 * dataW + 6] - src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 2] - src[sIdx + 5 * dataW + 5] - src[sIdx + 5 * dataW + 6]) + 1./1. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 5] - src[sIdx + 1 * dataW + 6] + src[sIdx + 7 * dataW + 1] + src[sIdx + 7 * dataW + 2] + src[sIdx + 7 * dataW + 5] + src[sIdx + 7 * dataW + 6]) + 357./16. * ( - src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4] + src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4]) + 17./4. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4] - src[sIdx + 7 * dataW + 3] - src[sIdx + 7 * dataW + 4]);
        dst[bIdx + 58 * gap] = + 21./4. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 5] + src[sIdx + 3 * dataW + 6] + src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 5] - src[sIdx + 5 * dataW + 6]) + 1./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 5] - src[sIdx + 1 * dataW + 6] - src[sIdx + 7 * dataW + 1] + src[sIdx + 7 * dataW + 2] - src[sIdx + 7 * dataW + 5] + src[sIdx + 7 * dataW + 6]) + 357./16. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4] - src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4]) + 17./4. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4] + src[sIdx + 7 * dataW + 3] - src[sIdx + 7 * dataW + 4]);
        dst[bIdx + 59 * gap] = + 105./16. * ( - src[sIdx + 3 * dataW + 4] + src[sIdx + 5 * dataW + 4]) + 1./2. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 7 * dataW + 1]) + 2./1. * ( - src[sIdx + 1 * dataW + 5] + src[sIdx + 7 * dataW + 5]) + 1./4. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 7 * dataW + 2]) + 21./2. * ( + src[sIdx + 3 * dataW + 5] - src[sIdx + 5 * dataW + 5]) + 1./1. * ( - src[sIdx + 1 * dataW + 6] + src[sIdx + 7 * dataW + 6]) + 5./4. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 7 * dataW + 4]) + 105./8. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 5 * dataW + 3]) + 5./2. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 7 * dataW + 3]) + 21./4. * ( + src[sIdx + 3 * dataW + 6] - src[sIdx + 5 * dataW + 6]) + 21./16. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 5 * dataW + 2]) + 21./8. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 5 * dataW + 1]);
        dst[bIdx + 60 * gap] = + 105./16. * ( - src[sIdx + 3 * dataW + 4] + src[sIdx + 5 * dataW + 4]) + 1./2. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 7 * dataW + 1]) + 2./1. * ( + src[sIdx + 1 * dataW + 5] - src[sIdx + 7 * dataW + 5]) + 1./4. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 7 * dataW + 2]) + 21./2. * ( - src[sIdx + 3 * dataW + 5] + src[sIdx + 5 * dataW + 5]) + 1./1. * ( - src[sIdx + 1 * dataW + 6] + src[sIdx + 7 * dataW + 6]) + 5./4. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 7 * dataW + 4]) + 105./8. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 5 * dataW + 3]) + 5./2. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 7 * dataW + 3]) + 21./4. * ( + src[sIdx + 3 * dataW + 6] - src[sIdx + 5 * dataW + 6]) + 21./16. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 5 * dataW + 2]) + 21./8. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 5 * dataW + 1]);
        dst[bIdx + 61 * gap] = + 1./2. * ( - src[sIdx + 1 * dataW + 5] + src[sIdx + 7 * dataW + 5]) + 2./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 7 * dataW + 1]) + 4./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 7 * dataW + 2]) + 5./1. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 7 * dataW + 4]) + 21./2. * ( + src[sIdx + 3 * dataW + 1] - src[sIdx + 5 * dataW + 1]) + 1./1. * ( - src[sIdx + 1 * dataW + 6] + src[sIdx + 7 * dataW + 6]) + 105./8. * ( - src[sIdx + 3 * dataW + 3] + src[sIdx + 5 * dataW + 3]) + 105./4. * ( - src[sIdx + 3 * dataW + 4] + src[sIdx + 5 * dataW + 4]) + 21./1. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 5 * dataW + 2]) + 5./2. * ( + src[sIdx + 1 * dataW + 3] - src[sIdx + 7 * dataW + 3]) + 21./4. * ( + src[sIdx + 3 * dataW + 6] - src[sIdx + 5 * dataW + 6]) + 21./8. * ( + src[sIdx + 3 * dataW + 5] - src[sIdx + 5 * dataW + 5]);
        dst[bIdx + 62 * gap] = + 1./2. * ( + src[sIdx + 1 * dataW + 5] - src[sIdx + 7 * dataW + 5]) + 2./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 7 * dataW + 1]) + 4./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 7 * dataW + 2]) + 5./1. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 7 * dataW + 4]) + 21./2. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 5 * dataW + 1]) + 1./1. * ( - src[sIdx + 1 * dataW + 6] + src[sIdx + 7 * dataW + 6]) + 105./8. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 5 * dataW + 3]) + 105./4. * ( - src[sIdx + 3 * dataW + 4] + src[sIdx + 5 * dataW + 4]) + 21./1. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 5 * dataW + 2]) + 5./2. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 7 * dataW + 3]) + 21./4. * ( + src[sIdx + 3 * dataW + 6] - src[sIdx + 5 * dataW + 6]) + 21./8. * ( - src[sIdx + 3 * dataW + 5] + src[sIdx + 5 * dataW + 5]);
        dst[bIdx + 63 * gap] = + 21./4. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 5] - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 7] + src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 7] + src[sIdx + 7 * dataW + 3] - src[sIdx + 7 * dataW + 5]) + 1./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 7] - src[sIdx + 7 * dataW + 1] + src[sIdx + 7 * dataW + 7]) + 441./16. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 5] - src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 5]);
	}
}


template <typename Dtype> 
__global__ void winoSrcAddOpt_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int tileH, int tileW, int inputs, int batchs, int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int gap = inputs * batchs * tileH * tileW;
		int highIdx = idx / (tileH * tileW);
		int yIdx = idx / tileW % tileH;
		int xIdx = idx % tileW;
		int bIdx = idx;
		int sIdx = highIdx * dataW * dataH + yIdx * dataW * 2 + xIdx * 2;

		Dtype t[5];
		dst[bIdx + 0 * gap] = + 1./1. * ( + src[sIdx + 0 * dataW + 0] - src[sIdx + 0 * dataW + 2] - src[sIdx + 2 * dataW + 0] + src[sIdx + 2 * dataW + 2]);
		t[0] = + src[sIdx + 0 * dataW + 2] - src[sIdx + 2 * dataW + 2];
		dst[bIdx + 1 * gap] = + 1./1. * ( + src[sIdx + 0 * dataW + 1] - src[sIdx + 2 * dataW + 1] + t[0]);
		t[1] = - src[sIdx + 0 * dataW + 1] + src[sIdx + 2 * dataW + 1];
		dst[bIdx + 2 * gap] = + 1./1. * ( + t[0] + t[1]);
		dst[bIdx + 3 * gap] = + 1./1. * ( + src[sIdx + 0 * dataW + 3] - src[sIdx + 2 * dataW + 3] + t[1]);
		t[0] = - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 3];
		dst[bIdx + 7 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 3] + t[0]);
		t[1] = + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 3];
		dst[bIdx + 11 * gap] = + 1./1. * ( + t[1] + t[0]);
		t[0] = + src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 2];
		dst[bIdx + 5 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 2 * dataW + 1] + t[0]);
		dst[bIdx + 6 * gap] = + 1./1. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 2 * dataW + 1] + t[0]);
		t[2] = - src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2];
		dst[bIdx + 9 * gap] = + 1./1. * ( + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] + t[2]);
		t[0] = + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2];
		dst[bIdx + 10 * gap] = + 1./1. * ( - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2] + t[0]);
		t[3] = + src[sIdx + 2 * dataW + 0] - src[sIdx + 2 * dataW + 2];
		dst[bIdx + 4 * gap] = + 1./1. * ( + src[sIdx + 1 * dataW + 0] - src[sIdx + 1 * dataW + 2] + t[3]);
		t[4] = - src[sIdx + 1 * dataW + 0] + src[sIdx + 1 * dataW + 2];
		dst[bIdx + 8 * gap] = + 1./1. * ( + t[3] + t[4]);
		dst[bIdx + 12 * gap] = + 1./1. * ( + src[sIdx + 3 * dataW + 0] - src[sIdx + 3 * dataW + 2] + t[4]);
		dst[bIdx + 13 * gap] = + 1./1. * ( + src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2] + t[2]);
		dst[bIdx + 14 * gap] = + 1./1. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2] + t[0]);
		dst[bIdx + 15 * gap] = + 1./1. * ( - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 3] + t[1]);
	}
}

template <typename Dtype> 
__global__ void wino4x4SrcAddOpt_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int tileH, int tileW, int inputs, int batchs, int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int gap = inputs * batchs * tileH * tileW;
		int highIdx = idx / (tileH * tileW);
		int yIdx = idx / tileW % tileH;
		int xIdx = idx % tileW;
		int bIdx = idx;
		int sIdx = highIdx * dataW * dataH + yIdx * dataW * 4 + xIdx * 4;

		Dtype t[30];
		t[3] = - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2];
		dst[bIdx + 0 * gap] = + 1./1. * ( + src[sIdx + 4 * dataW + 4]) + 4./1. * ( + src[sIdx + 0 * dataW + 4] + src[sIdx + 4 * dataW + 0]) + 5./1. * ( + t[3]) + 16./1. * ( + src[sIdx + 0 * dataW + 0]) + 20./1. * ( - src[sIdx + 0 * dataW + 2] - src[sIdx + 2 * dataW + 0]) + 25./1. * ( + src[sIdx + 2 * dataW + 2]);
		t[5] = + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2];
		t[0] = + src[sIdx + 0 * dataW + 4] - src[sIdx + 4 * dataW + 2];
		t[22] = + src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4];
		dst[bIdx + 1 * gap] = + 16./1. * ( - src[sIdx + 0 * dataW + 1] - src[sIdx + 0 * dataW + 2]) + 20./1. * ( + t[5]) + 4./1. * ( + src[sIdx + 0 * dataW + 3] - src[sIdx + 4 * dataW + 1] + t[0]) + 5./1. * ( - src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4]) + 1./1. * ( + t[22]);
		t[4] = - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 2];
		t[28] = - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4];
		dst[bIdx + 2 * gap] = + 16./1. * ( + src[sIdx + 0 * dataW + 1] - src[sIdx + 0 * dataW + 2]) + 20./1. * ( + t[4]) + 4./1. * ( - src[sIdx + 0 * dataW + 3] + src[sIdx + 4 * dataW + 1] + t[0]) + 5./1. * ( + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4]) + 1./1. * ( + t[28]);
		t[2] = - src[sIdx + 0 * dataW + 2] + src[sIdx + 0 * dataW + 4];
		t[1] = - src[sIdx + 4 * dataW + 2] + src[sIdx + 4 * dataW + 4];
		t[12] = - src[sIdx + 4 * dataW + 1] + src[sIdx + 4 * dataW + 3];
		t[0] = + src[sIdx + 2 * dataW + 2] - src[sIdx + 2 * dataW + 4];
		t[13] = + src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 3];
		dst[bIdx + 3 * gap] = + 1./1. * ( + t[1]) + 2./1. * ( + t[12]) + 4./1. * ( + t[2]) + 5./1. * ( + t[0]) + 8./1. * ( - src[sIdx + 0 * dataW + 1] + src[sIdx + 0 * dataW + 3]) + 10./1. * ( + t[13]);
		t[7] = - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 3];
		t[8] = + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 3];
		dst[bIdx + 4 * gap] = + 1./1. * ( + t[1]) + 2./1. * ( + t[8]) + 4./1. * ( + t[2]) + 5./1. * ( + t[0]) + 8./1. * ( + src[sIdx + 0 * dataW + 1] - src[sIdx + 0 * dataW + 3]) + 10./1. * ( + t[7]);
		dst[bIdx + 5 * gap] = + 1./1. * ( + src[sIdx + 4 * dataW + 5]) + 4./1. * ( + src[sIdx + 0 * dataW + 5] + src[sIdx + 4 * dataW + 1]) + 5./1. * ( - src[sIdx + 2 * dataW + 5] - src[sIdx + 4 * dataW + 3]) + 16./1. * ( + src[sIdx + 0 * dataW + 1]) + 20./1. * ( - src[sIdx + 0 * dataW + 3] - src[sIdx + 2 * dataW + 1]) + 25./1. * ( + src[sIdx + 2 * dataW + 3]);
		t[2] = - src[sIdx + 2 * dataW + 5] + src[sIdx + 4 * dataW + 1];
		t[9] = - src[sIdx + 3 * dataW + 3] - src[sIdx + 4 * dataW + 3];
		dst[bIdx + 11 * gap] = + 16./1. * ( - src[sIdx + 1 * dataW + 1] - src[sIdx + 2 * dataW + 1]) + 4./1. * ( - src[sIdx + 1 * dataW + 5] + src[sIdx + 3 * dataW + 1] + t[2]) + 20./1. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 2 * dataW + 3]) + 5./1. * ( + t[9]) + 1./1. * ( + src[sIdx + 3 * dataW + 5] + src[sIdx + 4 * dataW + 5]);
		dst[bIdx + 17 * gap] = + 16./1. * ( + src[sIdx + 1 * dataW + 1] - src[sIdx + 2 * dataW + 1]) + 4./1. * ( + src[sIdx + 1 * dataW + 5] - src[sIdx + 3 * dataW + 1] + t[2]) + 20./1. * ( - src[sIdx + 1 * dataW + 3] + src[sIdx + 2 * dataW + 3]) + 5./1. * ( + src[sIdx + 3 * dataW + 3] - src[sIdx + 4 * dataW + 3]) + 1./1. * ( - src[sIdx + 3 * dataW + 5] + src[sIdx + 4 * dataW + 5]);
		t[10] = - src[sIdx + 2 * dataW + 5] + src[sIdx + 4 * dataW + 5];
		t[2] = + src[sIdx + 2 * dataW + 3] - src[sIdx + 4 * dataW + 3];
		t[6] = - src[sIdx + 2 * dataW + 1] + src[sIdx + 4 * dataW + 1];
		t[25] = - src[sIdx + 1 * dataW + 1] + src[sIdx + 3 * dataW + 1];
		t[27] = + src[sIdx + 1 * dataW + 3] - src[sIdx + 3 * dataW + 3];
		dst[bIdx + 23 * gap] = + 1./1. * ( + t[10]) + 2./1. * ( - src[sIdx + 1 * dataW + 5] + src[sIdx + 3 * dataW + 5]) + 4./1. * ( + t[6]) + 5./1. * ( + t[2]) + 8./1. * ( + t[25]) + 10./1. * ( + t[27]);
		t[26] = - src[sIdx + 1 * dataW + 3] + src[sIdx + 3 * dataW + 3];
		t[29] = + src[sIdx + 1 * dataW + 1] - src[sIdx + 3 * dataW + 1];
		dst[bIdx + 29 * gap] = + 1./1. * ( + t[10]) + 2./1. * ( + src[sIdx + 1 * dataW + 5] - src[sIdx + 3 * dataW + 5]) + 4./1. * ( + t[6]) + 5./1. * ( + t[2]) + 8./1. * ( + t[29]) + 10./1. * ( + t[26]);
		t[16] = - src[sIdx + 1 * dataW + 3] - src[sIdx + 3 * dataW + 1];
		t[17] = + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4];
		t[19] = - src[sIdx + 2 * dataW + 3] - src[sIdx + 4 * dataW + 1];
		t[10] = - src[sIdx + 1 * dataW + 4] - src[sIdx + 3 * dataW + 2];
		dst[bIdx + 7 * gap] = + 16./1. * ( + src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] + t[5]) + 1./1. * ( + src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 3] + t[17]) + 4./1. * ( + t[3] + t[16] + t[19] + t[10]);
		t[24] = + src[sIdx + 2 * dataW + 3] + src[sIdx + 4 * dataW + 1];
		dst[bIdx + 8 * gap] = + 16./1. * ( - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 2] + t[4]) + 1./1. * ( + t[17] + t[9]) + 4./1. * ( + src[sIdx + 1 * dataW + 3] + src[sIdx + 3 * dataW + 1] + t[3] + t[10] + t[24]);
		t[18] = - src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 4];
		t[9] = + src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 4];
		t[10] = - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 3];
		t[15] = + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 3];
		dst[bIdx + 9 * gap] = + 8./1. * ( + t[13] + t[15]) + 1./1. * ( + t[1] + t[18]) + 2./1. * ( + t[10] + t[12]) + 4./1. * ( + t[0] + t[9]);
		t[11] = - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 3];
		t[14] = + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 3];
		dst[bIdx + 10 * gap] = + 8./1. * ( + t[11] + t[7]) + 1./1. * ( + t[1] + t[18]) + 2./1. * ( + t[14] + t[8]) + 4./1. * ( + t[0] + t[9]);
		t[21] = + src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4];
		t[20] = + src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2];
		t[23] = - src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4];
		t[18] = - src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2];
		dst[bIdx + 13 * gap] = + 16./1. * ( + t[18] + t[5]) + 1./1. * ( + t[23] + t[22]) + 4./1. * ( + t[3] + t[20] + t[21] + t[19]);
		t[22] = + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 2];
		t[19] = + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4];
		dst[bIdx + 14 * gap] = + 16./1. * ( + t[4] + t[22]) + 1./1. * ( + t[19] + t[28]) + 4./1. * ( + src[sIdx + 1 * dataW + 4] + src[sIdx + 3 * dataW + 2] + t[3] + t[16] + t[24]);
		t[3] = + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 4];
		t[4] = - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 4];
		dst[bIdx + 15 * gap] = + 8./1. * ( + t[11] + t[13]) + 1./1. * ( + t[1] + t[3]) + 2./1. * ( + t[12] + t[14]) + 4./1. * ( + t[0] + t[4]);
		dst[bIdx + 16 * gap] = + 8./1. * ( + t[15] + t[7]) + 1./1. * ( + t[1] + t[3]) + 2./1. * ( + t[10] + t[8]) + 4./1. * ( + t[0] + t[4]);
		t[9] = + src[sIdx + 1 * dataW + 2] - src[sIdx + 3 * dataW + 2];
		t[5] = - src[sIdx + 2 * dataW + 4] + src[sIdx + 4 * dataW + 4];
		t[7] = + src[sIdx + 2 * dataW + 2] - src[sIdx + 4 * dataW + 2];
		t[8] = - src[sIdx + 1 * dataW + 4] + src[sIdx + 3 * dataW + 4];
		t[28] = + src[sIdx + 2 * dataW + 1] - src[sIdx + 4 * dataW + 1];
		t[24] = - src[sIdx + 2 * dataW + 3] + src[sIdx + 4 * dataW + 3];
		dst[bIdx + 19 * gap] = + 8./1. * ( + t[9] + t[29]) + 1./1. * ( + t[5] + t[24]) + 2./1. * ( + t[8] + t[26]) + 4./1. * ( + t[7] + t[28]);
		dst[bIdx + 20 * gap] = + 8./1. * ( + t[9] + t[25]) + 1./1. * ( + t[2] + t[5]) + 2./1. * ( + t[8] + t[27]) + 4./1. * ( + t[6] + t[7]);
		dst[bIdx + 21 * gap] = + 1./1. * ( + t[0] + t[1]) + 2./1. * ( + t[8] + t[9] + t[12] + t[13]) + 4./1. * ( + t[10] + t[15]);
		dst[bIdx + 22 * gap] = + 1./1. * ( + t[0] + t[1]) + 2./1. * ( + t[2] + t[6] + t[8] + t[9]) + 4./1. * ( + t[11] + t[14]);
		dst[bIdx + 25 * gap] = + 8./1. * ( + t[18] + t[20]) + 1./1. * ( + t[5] + t[24]) + 2./1. * ( + t[21] + t[23]) + 4./1. * ( + t[7] + t[28]);
		t[24] = - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 2];
		t[25] = - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4];
		dst[bIdx + 26 * gap] = + 8./1. * ( + t[22] + t[24]) + 1./1. * ( + t[2] + t[5]) + 2./1. * ( + t[19] + t[25]) + 4./1. * ( + t[6] + t[7]);
		dst[bIdx + 27 * gap] = + 1./1. * ( + t[0] + t[1]) + 2./1. * ( + t[3] + t[4] + t[12] + t[13]) + 4./1. * ( + t[11] + t[14]);
		dst[bIdx + 28 * gap] = + 1./1. * ( + t[0] + t[1]) + 2./1. * ( + t[2] + t[3] + t[4] + t[6]) + 4./1. * ( + t[10] + t[15]);
		t[0] = - src[sIdx + 2 * dataW + 4] + src[sIdx + 4 * dataW + 0];
		dst[bIdx + 6 * gap] = + 16./1. * ( - src[sIdx + 1 * dataW + 0] - src[sIdx + 2 * dataW + 0]) + 4./1. * ( - src[sIdx + 1 * dataW + 4] + src[sIdx + 3 * dataW + 0] + t[0]) + 20./1. * ( + src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 2]) + 5./1. * ( - src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 1./1. * ( + t[17]);
		dst[bIdx + 12 * gap] = + 16./1. * ( + src[sIdx + 1 * dataW + 0] - src[sIdx + 2 * dataW + 0]) + 4./1. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 3 * dataW + 0] + t[0]) + 20./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 2]) + 5./1. * ( + src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2]) + 1./1. * ( - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4]);
		t[0] = - src[sIdx + 2 * dataW + 0] + src[sIdx + 4 * dataW + 0];
		dst[bIdx + 18 * gap] = + 1./1. * ( + t[5]) + 2./1. * ( + t[8]) + 4./1. * ( + t[0]) + 5./1. * ( + t[7]) + 8./1. * ( - src[sIdx + 1 * dataW + 0] + src[sIdx + 3 * dataW + 0]) + 10./1. * ( + t[9]);
		dst[bIdx + 24 * gap] = + 1./1. * ( + t[5]) + 2./1. * ( + src[sIdx + 1 * dataW + 4] - src[sIdx + 3 * dataW + 4]) + 4./1. * ( + t[0]) + 5./1. * ( + t[7]) + 8./1. * ( + src[sIdx + 1 * dataW + 0] - src[sIdx + 3 * dataW + 0]) + 10./1. * ( - src[sIdx + 1 * dataW + 2] + src[sIdx + 3 * dataW + 2]);
		dst[bIdx + 30 * gap] = + 1./1. * ( + src[sIdx + 5 * dataW + 4]) + 4./1. * ( + src[sIdx + 1 * dataW + 4] + src[sIdx + 5 * dataW + 0]) + 5./1. * ( - src[sIdx + 3 * dataW + 4] - src[sIdx + 5 * dataW + 2]) + 16./1. * ( + src[sIdx + 1 * dataW + 0]) + 20./1. * ( - src[sIdx + 1 * dataW + 2] - src[sIdx + 3 * dataW + 0]) + 25./1. * ( + src[sIdx + 3 * dataW + 2]);
		dst[bIdx + 31 * gap] = + 16./1. * ( + t[18]) + 20./1. * ( + t[20]) + 4./1. * ( - src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 2] + t[21]) + 5./1. * ( + t[23]) + 1./1. * ( + src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4]);
		dst[bIdx + 32 * gap] = + 16./1. * ( + t[22]) + 20./1. * ( + t[24]) + 4./1. * ( + src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 2] + t[25]) + 5./1. * ( + t[19]) + 1./1. * ( - src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4]);
		t[0] = - src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 4];
		dst[bIdx + 33 * gap] = + 1./1. * ( + t[0]) + 2./1. * ( - src[sIdx + 5 * dataW + 1] + src[sIdx + 5 * dataW + 3]) + 4./1. * ( + t[4]) + 5./1. * ( + t[3]) + 8./1. * ( + t[11]) + 10./1. * ( + t[14]);
		dst[bIdx + 34 * gap] = + 1./1. * ( + t[0]) + 2./1. * ( + src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 3]) + 4./1. * ( + t[4]) + 5./1. * ( + t[3]) + 8./1. * ( + t[15]) + 10./1. * ( + t[10]);
		dst[bIdx + 35 * gap] = + 1./1. * ( + src[sIdx + 5 * dataW + 5]) + 4./1. * ( + src[sIdx + 1 * dataW + 5] + src[sIdx + 5 * dataW + 1]) + 5./1. * ( - src[sIdx + 3 * dataW + 5] - src[sIdx + 5 * dataW + 3]) + 16./1. * ( + src[sIdx + 1 * dataW + 1]) + 20./1. * ( + t[16]) + 25./1. * ( + src[sIdx + 3 * dataW + 3]);
	}
}

template <typename Dtype> 
__global__ void wino6x6SrcAddOpt_gpu_kernel(const Dtype *src, Dtype *dst, int dataH, int dataW, int tileH, int tileW, int inputs, int batchs, int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int gap = inputs * batchs * tileH * tileW;
		int highIdx = idx / (tileH * tileW);
		int yIdx = idx / tileW % tileH;
		int xIdx = idx % tileW;
		int bIdx = idx;
		int sIdx = highIdx * dataW * dataH + yIdx * dataW * 6 + xIdx * 6;

		Dtype t[106];
		t[2] = - src[sIdx + 4 * dataW + 6] - src[sIdx + 6 * dataW + 4];
		t[1] = + src[sIdx + 2 * dataW + 6] + src[sIdx + 6 * dataW + 2];
		t[0] = - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 2];
		dst[bIdx + 0 * gap] = + 21./4. * ( - src[sIdx + 0 * dataW + 2] + src[sIdx + 0 * dataW + 4] - src[sIdx + 2 * dataW + 0] + src[sIdx + 4 * dataW + 0] + t[1] + t[2]) + 1./1. * ( + src[sIdx + 0 * dataW + 0] - src[sIdx + 0 * dataW + 6] - src[sIdx + 6 * dataW + 0] + src[sIdx + 6 * dataW + 6]) + 441./16. * ( + src[sIdx + 2 * dataW + 2] + src[sIdx + 4 * dataW + 4] + t[0]);
		t[20] = + src[sIdx + 4 * dataW + 1] + src[sIdx + 4 * dataW + 5];
		t[12] = - src[sIdx + 0 * dataW + 3] + src[sIdx + 6 * dataW + 3];
		t[6] = - src[sIdx + 0 * dataW + 4] + src[sIdx + 6 * dataW + 4];
		t[3] = - src[sIdx + 2 * dataW + 6] + src[sIdx + 4 * dataW + 6];
		t[5] = - src[sIdx + 2 * dataW + 2] + src[sIdx + 4 * dataW + 2];
		t[37] = - src[sIdx + 2 * dataW + 1] - src[sIdx + 2 * dataW + 5];
		t[33] = - src[sIdx + 6 * dataW + 1] - src[sIdx + 6 * dataW + 5];
		t[8] = + src[sIdx + 0 * dataW + 6] - src[sIdx + 6 * dataW + 6];
		t[13] = + src[sIdx + 2 * dataW + 3] - src[sIdx + 4 * dataW + 3];
		t[4] = + src[sIdx + 0 * dataW + 2] - src[sIdx + 6 * dataW + 2];
		t[7] = + src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 4];
		dst[bIdx + 1 * gap] = + 21./4. * ( + t[20] + t[3] + t[5] + t[37]) + 1./1. * ( + src[sIdx + 0 * dataW + 1] + src[sIdx + 0 * dataW + 5] + t[4] + t[8] + t[33]) + 357./16. * ( + t[7] + t[13]) + 17./4. * ( + t[6] + t[12]);
		t[9] = + src[sIdx + 0 * dataW + 3] - src[sIdx + 6 * dataW + 3];
		t[15] = - src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 5];
		t[14] = + src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 5];
		t[10] = - src[sIdx + 2 * dataW + 3] + src[sIdx + 4 * dataW + 3];
		t[26] = + src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 5];
		dst[bIdx + 2 * gap] = + 21./4. * ( + t[15] + t[26] + t[3] + t[5]) + 1./1. * ( - src[sIdx + 0 * dataW + 1] - src[sIdx + 0 * dataW + 5] + t[14] + t[4] + t[8]) + 357./16. * ( + t[7] + t[10]) + 17./4. * ( + t[6] + t[9]);
		t[23] = - src[sIdx + 2 * dataW + 5] + src[sIdx + 4 * dataW + 5];
		t[22] = + src[sIdx + 0 * dataW + 1] - src[sIdx + 6 * dataW + 1];
		t[16] = - src[sIdx + 2 * dataW + 1] + src[sIdx + 4 * dataW + 1];
		t[21] = + src[sIdx + 0 * dataW + 5] - src[sIdx + 6 * dataW + 5];
		dst[bIdx + 3 * gap] = + 105./16. * ( + t[7]) + 1./2. * ( + t[22]) + 2./1. * ( + t[21]) + 1./4. * ( + t[4]) + 21./2. * ( + t[23]) + 1./1. * ( + t[8]) + 5./4. * ( + t[6]) + 105./8. * ( + t[13]) + 5./2. * ( + t[12]) + 21./4. * ( + t[3]) + 21./16. * ( + t[5]) + 21./8. * ( + t[16]);
		t[19] = + src[sIdx + 2 * dataW + 1] - src[sIdx + 4 * dataW + 1];
		t[11] = - src[sIdx + 0 * dataW + 1] + src[sIdx + 6 * dataW + 1];
		t[18] = - src[sIdx + 0 * dataW + 5] + src[sIdx + 6 * dataW + 5];
		t[17] = + src[sIdx + 2 * dataW + 5] - src[sIdx + 4 * dataW + 5];
		dst[bIdx + 4 * gap] = + 105./16. * ( + t[7]) + 1./2. * ( + t[11]) + 2./1. * ( + t[18]) + 1./4. * ( + t[4]) + 21./2. * ( + t[17]) + 1./1. * ( + t[8]) + 5./4. * ( + t[6]) + 105./8. * ( + t[10]) + 5./2. * ( + t[9]) + 21./4. * ( + t[3]) + 21./16. * ( + t[5]) + 21./8. * ( + t[19]);
		dst[bIdx + 5 * gap] = + 1./2. * ( + t[21]) + 2./1. * ( + t[22]) + 4./1. * ( + t[4]) + 5./1. * ( + t[6]) + 21./2. * ( + t[16]) + 1./1. * ( + t[8]) + 105./8. * ( + t[13]) + 105./4. * ( + t[7]) + 21./1. * ( + t[5]) + 5./2. * ( + t[12]) + 21./4. * ( + t[3]) + 21./8. * ( + t[23]);
		dst[bIdx + 6 * gap] = + 1./2. * ( + t[18]) + 2./1. * ( + t[11]) + 4./1. * ( + t[4]) + 5./1. * ( + t[6]) + 21./2. * ( + t[19]) + 1./1. * ( + t[8]) + 105./8. * ( + t[10]) + 105./4. * ( + t[7]) + 21./1. * ( + t[5]) + 5./2. * ( + t[9]) + 21./4. * ( + t[3]) + 21./8. * ( + t[17]);
		t[35] = - src[sIdx + 4 * dataW + 1] - src[sIdx + 6 * dataW + 3];
		t[36] = + src[sIdx + 2 * dataW + 1] + src[sIdx + 6 * dataW + 5];
		t[40] = - src[sIdx + 2 * dataW + 3] - src[sIdx + 4 * dataW + 5];
		dst[bIdx + 7 * gap] = + 21./4. * ( + src[sIdx + 0 * dataW + 3] - src[sIdx + 0 * dataW + 5] - src[sIdx + 2 * dataW + 7] + src[sIdx + 4 * dataW + 7] + t[35] + t[36]) + 1./1. * ( + src[sIdx + 0 * dataW + 7] - src[sIdx + 6 * dataW + 7] + t[11]) + 441./16. * ( + src[sIdx + 2 * dataW + 5] + src[sIdx + 4 * dataW + 3] + t[40]);
		t[3] = - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 5];
		t[16] = - src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 5];
		t[7] = - src[sIdx + 2 * dataW + 1] - src[sIdx + 6 * dataW + 1];
		t[11] = - src[sIdx + 2 * dataW + 5] - src[sIdx + 6 * dataW + 5];
		t[12] = + src[sIdx + 2 * dataW + 3] + src[sIdx + 6 * dataW + 3];
		t[4] = + src[sIdx + 4 * dataW + 1] - src[sIdx + 4 * dataW + 7];
		t[22] = + src[sIdx + 1 * dataW + 3] + src[sIdx + 5 * dataW + 3];
		t[17] = - src[sIdx + 1 * dataW + 1] - src[sIdx + 5 * dataW + 1];
		t[18] = + src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 7];
		t[27] = - src[sIdx + 1 * dataW + 5] - src[sIdx + 5 * dataW + 5];
		t[5] = + src[sIdx + 2 * dataW + 7] + src[sIdx + 6 * dataW + 7];
		dst[bIdx + 15 * gap] = + 21./4. * ( + t[11] + t[12] + t[22] + t[27]) + 1./1. * ( + src[sIdx + 1 * dataW + 7] + src[sIdx + 5 * dataW + 7] + t[7] + t[17] + t[5]) + 357./16. * ( + t[3] + t[16]) + 17./4. * ( + t[4] + t[18]);
		t[10] = + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 5];
		t[34] = + src[sIdx + 1 * dataW + 1] + src[sIdx + 5 * dataW + 1];
		t[13] = - src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 7];
		t[39] = - src[sIdx + 1 * dataW + 3] - src[sIdx + 5 * dataW + 3];
		t[38] = + src[sIdx + 1 * dataW + 5] + src[sIdx + 5 * dataW + 5];
		dst[bIdx + 23 * gap] = + 21./4. * ( + t[11] + t[12] + t[38] + t[39]) + 1./1. * ( - src[sIdx + 1 * dataW + 7] - src[sIdx + 5 * dataW + 7] + t[7] + t[34] + t[5]) + 357./16. * ( + t[3] + t[10]) + 17./4. * ( + t[4] + t[13]);
		t[25] = + src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 5];
		t[6] = + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 5];
		t[5] = + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 5];
		t[28] = - src[sIdx + 5 * dataW + 1] + src[sIdx + 5 * dataW + 7];
		t[8] = - src[sIdx + 6 * dataW + 1] + src[sIdx + 6 * dataW + 7];
		t[9] = - src[sIdx + 2 * dataW + 1] + src[sIdx + 2 * dataW + 7];
		t[29] = + src[sIdx + 5 * dataW + 3] - src[sIdx + 5 * dataW + 5];
		t[19] = - src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 7];
		dst[bIdx + 31 * gap] = + 105./16. * ( + t[3]) + 1./2. * ( + t[19]) + 2./1. * ( + t[28]) + 1./4. * ( + t[9]) + 21./2. * ( + t[29]) + 1./1. * ( + t[8]) + 105./8. * ( + t[16]) + 5./4. * ( + t[4]) + 5./2. * ( + t[18]) + 21./4. * ( + t[5]) + 21./16. * ( + t[6]) + 21./8. * ( + t[25]);
		t[21] = - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 5];
		t[24] = - src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 5];
		t[84] = + src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 7];
		t[23] = + src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 7];
		dst[bIdx + 39 * gap] = + 105./16. * ( + t[3]) + 1./2. * ( + t[84]) + 2./1. * ( + t[23]) + 1./4. * ( + t[9]) + 21./2. * ( + t[24]) + 1./1. * ( + t[8]) + 105./8. * ( + t[10]) + 5./4. * ( + t[4]) + 5./2. * ( + t[13]) + 21./4. * ( + t[5]) + 21./16. * ( + t[6]) + 21./8. * ( + t[21]);
		dst[bIdx + 47 * gap] = + 1./2. * ( + t[28]) + 2./1. * ( + t[19]) + 4./1. * ( + t[9]) + 5./1. * ( + t[4]) + 21./2. * ( + t[25]) + 1./1. * ( + t[8]) + 105./8. * ( + t[16]) + 105./4. * ( + t[3]) + 21./1. * ( + t[6]) + 5./2. * ( + t[18]) + 21./4. * ( + t[5]) + 21./8. * ( + t[29]);
		dst[bIdx + 55 * gap] = + 1./2. * ( + t[23]) + 2./1. * ( + t[84]) + 4./1. * ( + t[9]) + 5./1. * ( + t[4]) + 21./2. * ( + t[21]) + 1./1. * ( + t[8]) + 105./8. * ( + t[10]) + 105./4. * ( + t[3]) + 21./1. * ( + t[6]) + 5./2. * ( + t[13]) + 21./4. * ( + t[5]) + 21./8. * ( + t[24]);
		t[13] = + src[sIdx + 1 * dataW + 5] + src[sIdx + 5 * dataW + 1];
		t[32] = - src[sIdx + 2 * dataW + 3] - src[sIdx + 6 * dataW + 3];
		t[85] = + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 3];
		t[30] = + src[sIdx + 1 * dataW + 1] + src[sIdx + 5 * dataW + 5];
		t[21] = + src[sIdx + 1 * dataW + 6] + src[sIdx + 5 * dataW + 6];
		t[31] = - src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 6];
		t[16] = - src[sIdx + 1 * dataW + 4] - src[sIdx + 5 * dataW + 4];
		t[19] = - src[sIdx + 3 * dataW + 5] - src[sIdx + 5 * dataW + 3];
		t[24] = - src[sIdx + 1 * dataW + 3] - src[sIdx + 3 * dataW + 1];
		t[23] = + src[sIdx + 1 * dataW + 2] + src[sIdx + 5 * dataW + 2];
		t[29] = + src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 4];
		t[3] = + src[sIdx + 2 * dataW + 2] + src[sIdx + 6 * dataW + 6];
		dst[bIdx + 9 * gap] = + 289./16. * ( + t[29] + t[85]) + 1./1. * ( + t[1] + t[3] + t[13] + t[14] + t[21] + t[23] + t[26] + t[30]) + 17./4. * ( + t[0] + t[2] + t[15] + t[16] + t[19] + t[24] + t[31] + t[32]);
		t[82] = + src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 3];
		t[28] = - src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 4];
		t[25] = + src[sIdx + 3 * dataW + 1] + src[sIdx + 3 * dataW + 5];
		dst[bIdx + 10 * gap] = + 289./16. * ( + t[28] + t[82]) + 1./1. * ( + t[1] + t[3] + t[7] + t[11] + t[17] + t[21] + t[23] + t[27]) + 17./4. * ( + t[0] + t[2] + t[12] + t[16] + t[20] + t[22] + t[25] + t[31]);
		t[4] = + src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4];
		t[47] = - src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 1];
		t[9] = - src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 6];
		t[42] = - src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 5];
		t[8] = + src[sIdx + 2 * dataW + 2] + src[sIdx + 6 * dataW + 2];
		t[45] = + src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 3];
		t[43] = + src[sIdx + 2 * dataW + 1] + src[sIdx + 6 * dataW + 1];
		t[46] = + src[sIdx + 2 * dataW + 5] + src[sIdx + 6 * dataW + 5];
		t[10] = - src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2];
		t[5] = + src[sIdx + 2 * dataW + 6] + src[sIdx + 6 * dataW + 6];
		t[6] = - src[sIdx + 2 * dataW + 4] - src[sIdx + 6 * dataW + 4];
		dst[bIdx + 11 * gap] = + 17./8. * ( + t[47]) + 1./2. * ( + t[34] + t[43]) + 2./1. * ( + t[38] + t[46]) + 1./4. * ( + t[8] + t[23]) + 1./1. * ( + t[5] + t[21]) + 5./4. * ( + t[6] + t[16]) + 17./2. * ( + t[42]) + 17./16. * ( + t[10]) + 17./4. * ( + t[9]) + 5./2. * ( + t[32] + t[39]) + 85./16. * ( + t[4]) + 85./8. * ( + t[45]);
		t[41] = - src[sIdx + 3 * dataW + 3] - src[sIdx + 4 * dataW + 3];
		t[18] = + src[sIdx + 3 * dataW + 1] + src[sIdx + 4 * dataW + 1];
		t[44] = + src[sIdx + 3 * dataW + 5] + src[sIdx + 4 * dataW + 5];
		dst[bIdx + 12 * gap] = + 17./8. * ( + t[18]) + 1./2. * ( + t[7] + t[17]) + 2./1. * ( + t[11] + t[27]) + 1./4. * ( + t[8] + t[23]) + 1./1. * ( + t[5] + t[21]) + 5./4. * ( + t[6] + t[16]) + 17./2. * ( + t[44]) + 17./16. * ( + t[10]) + 17./4. * ( + t[9]) + 5./2. * ( + t[12] + t[22]) + 85./16. * ( + t[4]) + 85./8. * ( + t[41]);
		dst[bIdx + 13 * gap] = + 17./8. * ( + t[42]) + 1./2. * ( + t[38] + t[46]) + 2./1. * ( + t[34] + t[43]) + 4./1. * ( + t[8] + t[23]) + 5./1. * ( + t[6] + t[16]) + 1./1. * ( + t[5] + t[21]) + 17./2. * ( + t[47]) + 17./1. * ( + t[10]) + 17./4. * ( + t[9]) + 5./2. * ( + t[32] + t[39]) + 85./4. * ( + t[4]) + 85./8. * ( + t[45]);
		dst[bIdx + 14 * gap] = + 17./8. * ( + t[44]) + 1./2. * ( + t[11] + t[27]) + 2./1. * ( + t[7] + t[17]) + 4./1. * ( + t[8] + t[23]) + 5./1. * ( + t[6] + t[16]) + 1./1. * ( + t[5] + t[21]) + 17./2. * ( + t[18]) + 17./1. * ( + t[10]) + 17./4. * ( + t[9]) + 5./2. * ( + t[12] + t[22]) + 85./4. * ( + t[4]) + 85./8. * ( + t[41]);
		t[4] = - src[sIdx + 5 * dataW + 2] - src[sIdx + 5 * dataW + 6];
		t[18] = + src[sIdx + 1 * dataW + 4] + src[sIdx + 5 * dataW + 4];
		t[9] = + src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 6];
		t[83] = - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 3];
		t[10] = - src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 6];
		dst[bIdx + 17 * gap] = + 289./16. * ( + t[28] + t[83]) + 1./1. * ( + t[1] + t[3] + t[4] + t[10] + t[14] + t[17] + t[26] + t[27]) + 17./4. * ( + t[0] + t[2] + t[9] + t[15] + t[18] + t[22] + t[25] + t[32]);
		t[81] = - src[sIdx + 3 * dataW + 4] - src[sIdx + 4 * dataW + 3];
		dst[bIdx + 18 * gap] = + 289./16. * ( + t[29] + t[81]) + 1./1. * ( + t[1] + t[3] + t[4] + t[7] + t[10] + t[11] + t[13] + t[30]) + 17./4. * ( + t[0] + t[2] + t[9] + t[12] + t[18] + t[19] + t[20] + t[24]);
		t[42] = - src[sIdx + 1 * dataW + 6] - src[sIdx + 5 * dataW + 6];
		t[53] = + src[sIdx + 3 * dataW + 1] - src[sIdx + 4 * dataW + 1];
		t[45] = + src[sIdx + 3 * dataW + 2] - src[sIdx + 4 * dataW + 2];
		t[44] = - src[sIdx + 3 * dataW + 4] + src[sIdx + 4 * dataW + 4];
		t[47] = + src[sIdx + 3 * dataW + 6] - src[sIdx + 4 * dataW + 6];
		t[41] = - src[sIdx + 1 * dataW + 2] - src[sIdx + 5 * dataW + 2];
		t[51] = - src[sIdx + 3 * dataW + 3] + src[sIdx + 4 * dataW + 3];
		t[48] = + src[sIdx + 3 * dataW + 5] - src[sIdx + 4 * dataW + 5];
		dst[bIdx + 19 * gap] = + 17./8. * ( + t[53]) + 1./2. * ( + t[17] + t[43]) + 2./1. * ( + t[27] + t[46]) + 1./4. * ( + t[8] + t[41]) + 1./1. * ( + t[5] + t[42]) + 5./4. * ( + t[6] + t[18]) + 17./2. * ( + t[48]) + 17./16. * ( + t[45]) + 17./4. * ( + t[47]) + 5./2. * ( + t[22] + t[32]) + 85./16. * ( + t[44]) + 85./8. * ( + t[51]);
		t[52] = - src[sIdx + 3 * dataW + 1] + src[sIdx + 4 * dataW + 1];
		t[49] = - src[sIdx + 3 * dataW + 5] + src[sIdx + 4 * dataW + 5];
		t[50] = + src[sIdx + 3 * dataW + 3] - src[sIdx + 4 * dataW + 3];
		dst[bIdx + 20 * gap] = + 17./8. * ( + t[52]) + 1./2. * ( + t[7] + t[34]) + 2./1. * ( + t[11] + t[38]) + 1./4. * ( + t[8] + t[41]) + 1./1. * ( + t[5] + t[42]) + 5./4. * ( + t[6] + t[18]) + 17./2. * ( + t[49]) + 17./16. * ( + t[45]) + 17./4. * ( + t[47]) + 5./2. * ( + t[12] + t[39]) + 85./16. * ( + t[44]) + 85./8. * ( + t[50]);
		dst[bIdx + 21 * gap] = + 17./8. * ( + t[48]) + 1./2. * ( + t[27] + t[46]) + 2./1. * ( + t[17] + t[43]) + 4./1. * ( + t[8] + t[41]) + 5./1. * ( + t[6] + t[18]) + 1./1. * ( + t[5] + t[42]) + 17./2. * ( + t[53]) + 17./1. * ( + t[45]) + 17./4. * ( + t[47]) + 5./2. * ( + t[22] + t[32]) + 85./4. * ( + t[44]) + 85./8. * ( + t[51]);
		dst[bIdx + 22 * gap] = + 17./8. * ( + t[49]) + 1./2. * ( + t[11] + t[38]) + 2./1. * ( + t[7] + t[34]) + 4./1. * ( + t[8] + t[41]) + 5./1. * ( + t[6] + t[18]) + 1./1. * ( + t[5] + t[42]) + 17./2. * ( + t[52]) + 17./1. * ( + t[45]) + 17./4. * ( + t[47]) + 5./2. * ( + t[12] + t[39]) + 85./4. * ( + t[44]) + 85./8. * ( + t[50]);
		t[43] = - src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4];
		t[22] = + src[sIdx + 5 * dataW + 1] + src[sIdx + 5 * dataW + 5];
		t[27] = - src[sIdx + 3 * dataW + 1] - src[sIdx + 3 * dataW + 5];
		t[50] = + src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 6];
		t[7] = - src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 6];
		t[12] = + src[sIdx + 1 * dataW + 1] + src[sIdx + 1 * dataW + 5];
		t[71] = - src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4];
		t[69] = + src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 6];
		t[101] = - src[sIdx + 5 * dataW + 3] - src[sIdx + 5 * dataW + 4];
		t[93] = + src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 4];
		t[99] = - src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 4];
		t[63] = + src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4];
		t[6] = + src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 6];
		t[8] = + src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 6];
		dst[bIdx + 25 * gap] = + 17./8. * ( + t[99]) + 1./2. * ( + t[12] + t[50]) + 2./1. * ( + t[22] + t[69]) + 1./4. * ( + t[8] + t[26]) + 1./1. * ( + t[6] + t[14]) + 17./16. * ( + t[71]) + 17./2. * ( + t[101]) + 5./4. * ( + t[7] + t[15]) + 17./4. * ( + t[43]) + 5./2. * ( + t[31] + t[27]) + 85./16. * ( + t[63]) + 85./8. * ( + t[93]);
		t[98] = + src[sIdx + 5 * dataW + 3] - src[sIdx + 5 * dataW + 4];
		t[102] = - src[sIdx + 3 * dataW + 3] + src[sIdx + 3 * dataW + 4];
		t[34] = - src[sIdx + 1 * dataW + 1] - src[sIdx + 1 * dataW + 5];
		t[91] = + src[sIdx + 1 * dataW + 3] - src[sIdx + 1 * dataW + 4];
		t[75] = - src[sIdx + 4 * dataW + 3] + src[sIdx + 4 * dataW + 4];
		t[66] = + src[sIdx + 6 * dataW + 3] - src[sIdx + 6 * dataW + 4];
		t[32] = - src[sIdx + 5 * dataW + 1] - src[sIdx + 5 * dataW + 5];
		t[56] = + src[sIdx + 2 * dataW + 3] - src[sIdx + 2 * dataW + 4];
		dst[bIdx + 26 * gap] = + 17./8. * ( + t[91]) + 1./2. * ( + t[34] + t[50]) + 2./1. * ( + t[32] + t[69]) + 1./4. * ( + t[8] + t[37]) + 1./1. * ( + t[6] + t[33]) + 17./16. * ( + t[56]) + 17./2. * ( + t[98]) + 5./4. * ( + t[7] + t[20]) + 17./4. * ( + t[66]) + 5./2. * ( + t[25] + t[31]) + 85./16. * ( + t[75]) + 85./8. * ( + t[102]);
		t[54] = - src[sIdx + 1 * dataW + 4] - src[sIdx + 3 * dataW + 2];
		t[55] = - src[sIdx + 3 * dataW + 6] - src[sIdx + 5 * dataW + 4];
		t[58] = + src[sIdx + 2 * dataW + 5] + src[sIdx + 6 * dataW + 1];
		t[90] = + src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 1];
		t[92] = + src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 5];
		t[78] = + src[sIdx + 1 * dataW + 6] + src[sIdx + 5 * dataW + 2];
		t[70] = - src[sIdx + 2 * dataW + 3] - src[sIdx + 4 * dataW + 1];
		t[76] = - src[sIdx + 4 * dataW + 5] - src[sIdx + 6 * dataW + 3];
		dst[bIdx + 27 * gap] = + 1./2. * ( + t[58] + t[78]) + 2./1. * ( + t[92]) + 1./4. * ( + src[sIdx + 1 * dataW + 1] + t[1]) + 5./1. * ( + t[19]) + 1./1. * ( + src[sIdx + 6 * dataW + 6] + t[13]) + 5./4. * ( + t[2] + t[24]) + 5./8. * ( + t[54] + t[70]) + 25./16. * ( + src[sIdx + 4 * dataW + 4]) + 1./8. * ( + t[90]) + 25./8. * ( + t[85]) + 5./16. * ( + t[0]) + 5./2. * ( + t[55] + t[76]) + 1./16. * ( + src[sIdx + 2 * dataW + 2]) + 4./1. * ( + src[sIdx + 5 * dataW + 5]) + 25./4. * ( + src[sIdx + 3 * dataW + 3]);
		t[47] = + src[sIdx + 4 * dataW + 5] + src[sIdx + 6 * dataW + 3];
		t[52] = - src[sIdx + 1 * dataW + 5] - src[sIdx + 5 * dataW + 1];
		t[64] = + src[sIdx + 3 * dataW + 5] + src[sIdx + 5 * dataW + 3];
		t[95] = + src[sIdx + 5 * dataW + 6] - src[sIdx + 6 * dataW + 5];
		t[44] = + src[sIdx + 1 * dataW + 3] + src[sIdx + 3 * dataW + 1];
		t[46] = - src[sIdx + 2 * dataW + 5] - src[sIdx + 6 * dataW + 1];
		t[94] = + src[sIdx + 1 * dataW + 2] - src[sIdx + 2 * dataW + 1];
		t[68] = + src[sIdx + 2 * dataW + 3] + src[sIdx + 4 * dataW + 1];
		dst[bIdx + 28 * gap] = + 1./2. * ( + t[46] + t[78]) + 2./1. * ( + t[95]) + 1./4. * ( - src[sIdx + 1 * dataW + 1] + t[1]) + 5./1. * ( + t[64]) + 1./1. * ( + src[sIdx + 6 * dataW + 6] + t[52]) + 5./4. * ( + t[2] + t[44]) + 5./8. * ( + t[54] + t[68]) + 25./16. * ( + src[sIdx + 4 * dataW + 4]) + 1./8. * ( + t[94]) + 25./8. * ( + t[82]) + 5./16. * ( + t[0]) + 5./2. * ( + t[47] + t[55]) + 1./16. * ( + src[sIdx + 2 * dataW + 2]) + 4./1. * ( - src[sIdx + 5 * dataW + 5]) + 25./4. * ( - src[sIdx + 3 * dataW + 3]);
		t[17] = - src[sIdx + 1 * dataW + 4] - src[sIdx + 3 * dataW + 6];
		t[65] = + src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 6];
		t[39] = - src[sIdx + 3 * dataW + 2] - src[sIdx + 5 * dataW + 4];
		t[11] = - src[sIdx + 4 * dataW + 2] - src[sIdx + 6 * dataW + 4];
		t[45] = - src[sIdx + 3 * dataW + 1] - src[sIdx + 5 * dataW + 3];
		t[59] = + src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 2];
		t[5] = - src[sIdx + 2 * dataW + 4] - src[sIdx + 4 * dataW + 6];
		t[74] = - src[sIdx + 1 * dataW + 3] - src[sIdx + 3 * dataW + 5];
		t[38] = + src[sIdx + 1 * dataW + 2] + src[sIdx + 5 * dataW + 6];
		dst[bIdx + 29 * gap] = + 1./1. * ( + t[3] + t[30]) + 2./1. * ( + src[sIdx + 6 * dataW + 1] + t[38]) + 1./4. * ( + t[65]) + 5./1. * ( + t[11] + t[45]) + 1./2. * ( + src[sIdx + 1 * dataW + 6] + t[36]) + 5./4. * ( + t[5] + t[74]) + 10./1. * ( + t[39]) + 5./8. * ( + t[40]) + 1./8. * ( + src[sIdx + 2 * dataW + 5]) + 8./1. * ( + src[sIdx + 5 * dataW + 2]) + 25./2. * ( + src[sIdx + 3 * dataW + 4]) + 5./2. * ( + t[35] + t[17]) + 25./8. * ( + src[sIdx + 4 * dataW + 3]) + 4./1. * ( + t[59]) + 25./4. * ( + t[29]);
		t[80] = + src[sIdx + 4 * dataW + 1] + src[sIdx + 6 * dataW + 3];
		t[49] = - src[sIdx + 2 * dataW + 1] - src[sIdx + 6 * dataW + 5];
		t[72] = + src[sIdx + 2 * dataW + 3] + src[sIdx + 4 * dataW + 5];
		t[57] = - src[sIdx + 1 * dataW + 5] + src[sIdx + 2 * dataW + 6];
		t[60] = + src[sIdx + 3 * dataW + 1] + src[sIdx + 5 * dataW + 3];
		t[79] = - src[sIdx + 1 * dataW + 1] - src[sIdx + 5 * dataW + 5];
		t[73] = + src[sIdx + 1 * dataW + 3] + src[sIdx + 3 * dataW + 5];
		t[77] = - src[sIdx + 5 * dataW + 1] + src[sIdx + 6 * dataW + 2];
		dst[bIdx + 30 * gap] = + 1./1. * ( + t[3] + t[79]) + 2./1. * ( - src[sIdx + 6 * dataW + 1] + t[38]) + 1./4. * ( + t[57]) + 5./1. * ( + t[11] + t[60]) + 1./2. * ( + src[sIdx + 1 * dataW + 6] + t[49]) + 5./4. * ( + t[5] + t[73]) + 10./1. * ( + t[39]) + 5./8. * ( + t[72]) + 1./8. * ( - src[sIdx + 2 * dataW + 5]) + 8./1. * ( + src[sIdx + 5 * dataW + 2]) + 25./2. * ( + src[sIdx + 3 * dataW + 4]) + 5./2. * ( + t[17] + t[80]) + 25./8. * ( - src[sIdx + 4 * dataW + 3]) + 4./1. * ( + t[77]) + 25./4. * ( + t[28]);
		t[103] = + src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4];
		t[88] = + src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4];
		t[89] = - src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4];
		dst[bIdx + 33 * gap] = + 17./8. * ( + t[103]) + 1./2. * ( + t[10] + t[34]) + 2./1. * ( + t[4] + t[32]) + 1./4. * ( + t[8] + t[26]) + 1./1. * ( + t[6] + t[14]) + 17./16. * ( + t[71]) + 17./2. * ( + t[88]) + 5./4. * ( + t[7] + t[15]) + 17./4. * ( + t[43]) + 5./2. * ( + t[9] + t[25]) + 85./16. * ( + t[63]) + 85./8. * ( + t[89]);
		t[96] = - src[sIdx + 5 * dataW + 3] + src[sIdx + 5 * dataW + 4];
		t[97] = - src[sIdx + 1 * dataW + 3] + src[sIdx + 1 * dataW + 4];
		t[104] = + src[sIdx + 3 * dataW + 3] - src[sIdx + 3 * dataW + 4];
		dst[bIdx + 34 * gap] = + 17./8. * ( + t[97]) + 1./2. * ( + t[10] + t[12]) + 2./1. * ( + t[4] + t[22]) + 1./4. * ( + t[8] + t[37]) + 1./1. * ( + t[6] + t[33]) + 17./16. * ( + t[56]) + 17./2. * ( + t[96]) + 5./4. * ( + t[7] + t[20]) + 17./4. * ( + t[66]) + 5./2. * ( + t[9] + t[27]) + 85./16. * ( + t[75]) + 85./8. * ( + t[104]);
		t[105] = - src[sIdx + 1 * dataW + 2] + src[sIdx + 2 * dataW + 1];
		t[87] = - src[sIdx + 5 * dataW + 6] + src[sIdx + 6 * dataW + 5];
		t[62] = - src[sIdx + 1 * dataW + 6] - src[sIdx + 5 * dataW + 2];
		t[51] = + src[sIdx + 1 * dataW + 4] + src[sIdx + 3 * dataW + 2];
		t[48] = + src[sIdx + 3 * dataW + 6] + src[sIdx + 5 * dataW + 4];
		dst[bIdx + 35 * gap] = + 1./2. * ( + t[58] + t[62]) + 2./1. * ( + t[87]) + 1./4. * ( - src[sIdx + 1 * dataW + 1] + t[1]) + 5./1. * ( + t[64]) + 1./1. * ( + src[sIdx + 6 * dataW + 6] + t[52]) + 5./4. * ( + t[2] + t[44]) + 5./8. * ( + t[51] + t[70]) + 25./16. * ( + src[sIdx + 4 * dataW + 4]) + 1./8. * ( + t[105]) + 25./8. * ( + t[83]) + 5./16. * ( + t[0]) + 5./2. * ( + t[48] + t[76]) + 1./16. * ( + src[sIdx + 2 * dataW + 2]) + 4./1. * ( - src[sIdx + 5 * dataW + 5]) + 25./4. * ( - src[sIdx + 3 * dataW + 3]);
		t[86] = - src[sIdx + 1 * dataW + 2] - src[sIdx + 2 * dataW + 1];
		t[100] = - src[sIdx + 5 * dataW + 6] - src[sIdx + 6 * dataW + 5];
		dst[bIdx + 36 * gap] = + 1./2. * ( + t[46] + t[62]) + 2./1. * ( + t[100]) + 1./4. * ( + src[sIdx + 1 * dataW + 1] + t[1]) + 5./1. * ( + t[19]) + 1./1. * ( + src[sIdx + 6 * dataW + 6] + t[13]) + 5./4. * ( + t[2] + t[24]) + 5./8. * ( + t[51] + t[68]) + 25./16. * ( + src[sIdx + 4 * dataW + 4]) + 1./8. * ( + t[86]) + 25./8. * ( + t[81]) + 5./16. * ( + t[0]) + 5./2. * ( + t[47] + t[48]) + 1./16. * ( + src[sIdx + 2 * dataW + 2]) + 4./1. * ( + src[sIdx + 5 * dataW + 5]) + 25./4. * ( + src[sIdx + 3 * dataW + 3]);
		t[67] = + src[sIdx + 1 * dataW + 4] + src[sIdx + 3 * dataW + 6];
		t[53] = + src[sIdx + 3 * dataW + 2] + src[sIdx + 5 * dataW + 4];
		t[61] = - src[sIdx + 1 * dataW + 2] - src[sIdx + 5 * dataW + 6];
		dst[bIdx + 37 * gap] = + 1./1. * ( + t[3] + t[79]) + 2./1. * ( + src[sIdx + 6 * dataW + 1] + t[61]) + 1./4. * ( + t[57]) + 5./1. * ( + t[11] + t[60]) + 1./2. * ( - src[sIdx + 1 * dataW + 6] + t[36]) + 5./4. * ( + t[5] + t[73]) + 10./1. * ( + t[53]) + 5./8. * ( + t[40]) + 1./8. * ( + src[sIdx + 2 * dataW + 5]) + 8./1. * ( - src[sIdx + 5 * dataW + 2]) + 25./2. * ( - src[sIdx + 3 * dataW + 4]) + 5./2. * ( + t[35] + t[67]) + 25./8. * ( + src[sIdx + 4 * dataW + 3]) + 4./1. * ( + t[77]) + 25./4. * ( + t[28]);
		dst[bIdx + 38 * gap] = + 1./1. * ( + t[3] + t[30]) + 2./1. * ( - src[sIdx + 6 * dataW + 1] + t[61]) + 1./4. * ( + t[65]) + 5./1. * ( + t[11] + t[45]) + 1./2. * ( - src[sIdx + 1 * dataW + 6] + t[49]) + 5./4. * ( + t[5] + t[74]) + 10./1. * ( + t[53]) + 5./8. * ( + t[72]) + 1./8. * ( - src[sIdx + 2 * dataW + 5]) + 8./1. * ( - src[sIdx + 5 * dataW + 2]) + 25./2. * ( - src[sIdx + 3 * dataW + 4]) + 5./2. * ( + t[67] + t[80]) + 25./8. * ( - src[sIdx + 4 * dataW + 3]) + 4./1. * ( + t[59]) + 25./4. * ( + t[29]);
		dst[bIdx + 41 * gap] = + 17./8. * ( + t[101]) + 1./2. * ( + t[22] + t[69]) + 2./1. * ( + t[12] + t[50]) + 4./1. * ( + t[8] + t[26]) + 5./1. * ( + t[7] + t[15]) + 1./1. * ( + t[6] + t[14]) + 17./2. * ( + t[99]) + 17./1. * ( + t[71]) + 17./4. * ( + t[43]) + 5./2. * ( + t[31] + t[27]) + 85./4. * ( + t[63]) + 85./8. * ( + t[93]);
		dst[bIdx + 42 * gap] = + 17./8. * ( + t[98]) + 1./2. * ( + t[32] + t[69]) + 2./1. * ( + t[34] + t[50]) + 4./1. * ( + t[8] + t[37]) + 5./1. * ( + t[7] + t[20]) + 1./1. * ( + t[6] + t[33]) + 17./2. * ( + t[91]) + 17./1. * ( + t[56]) + 17./4. * ( + t[66]) + 5./2. * ( + t[25] + t[31]) + 85./4. * ( + t[75]) + 85./8. * ( + t[102]);
		dst[bIdx + 43 * gap] = + 1./1. * ( + t[3] + t[30]) + 2./1. * ( + src[sIdx + 1 * dataW + 6] + t[36]) + 4./1. * ( + t[65]) + 5./1. * ( + t[5] + t[74]) + 1./2. * ( + src[sIdx + 6 * dataW + 1] + t[38]) + 8./1. * ( + src[sIdx + 2 * dataW + 5]) + 10./1. * ( + t[40]) + 5./8. * ( + t[39]) + 1./8. * ( + src[sIdx + 5 * dataW + 2]) + 5./4. * ( + t[11] + t[45]) + 25./2. * ( + src[sIdx + 4 * dataW + 3]) + 5./2. * ( + t[35] + t[17]) + 25./8. * ( + src[sIdx + 3 * dataW + 4]) + 1./4. * ( + t[59]) + 25./4. * ( + t[29]);
		dst[bIdx + 44 * gap] = + 1./1. * ( + t[3] + t[79]) + 2./1. * ( + src[sIdx + 1 * dataW + 6] + t[49]) + 4./1. * ( + t[57]) + 5./1. * ( + t[5] + t[73]) + 1./2. * ( - src[sIdx + 6 * dataW + 1] + t[38]) + 8./1. * ( - src[sIdx + 2 * dataW + 5]) + 10./1. * ( + t[72]) + 5./8. * ( + t[39]) + 1./8. * ( + src[sIdx + 5 * dataW + 2]) + 5./4. * ( + t[11] + t[60]) + 25./2. * ( - src[sIdx + 4 * dataW + 3]) + 5./2. * ( + t[17] + t[80]) + 25./8. * ( + src[sIdx + 3 * dataW + 4]) + 1./4. * ( + t[77]) + 25./4. * ( + t[28]);
		dst[bIdx + 45 * gap] = + 1./1. * ( + src[sIdx + 6 * dataW + 6] + t[13]) + 2./1. * ( + t[58] + t[78]) + 4./1. * ( + src[sIdx + 1 * dataW + 1] + t[1]) + 5./1. * ( + t[2] + t[24]) + 1./2. * ( + t[92]) + 8./1. * ( + t[90]) + 10./1. * ( + t[54] + t[70]) + 16./1. * ( + src[sIdx + 2 * dataW + 2]) + 5./4. * ( + t[19]) + 25./2. * ( + t[85]) + 20./1. * ( + t[0]) + 5./2. * ( + t[55] + t[76]) + 25./1. * ( + src[sIdx + 4 * dataW + 4]) + 1./4. * ( + src[sIdx + 5 * dataW + 5]) + 25./4. * ( + src[sIdx + 3 * dataW + 3]);
		dst[bIdx + 46 * gap] = + 1./1. * ( + src[sIdx + 6 * dataW + 6] + t[52]) + 2./1. * ( + t[46] + t[78]) + 4./1. * ( - src[sIdx + 1 * dataW + 1] + t[1]) + 5./1. * ( + t[2] + t[44]) + 1./2. * ( + t[95]) + 8./1. * ( + t[94]) + 10./1. * ( + t[54] + t[68]) + 16./1. * ( + src[sIdx + 2 * dataW + 2]) + 5./4. * ( + t[64]) + 25./2. * ( + t[82]) + 20./1. * ( + t[0]) + 5./2. * ( + t[47] + t[55]) + 25./1. * ( + src[sIdx + 4 * dataW + 4]) + 1./4. * ( - src[sIdx + 5 * dataW + 5]) + 25./4. * ( - src[sIdx + 3 * dataW + 3]);
		dst[bIdx + 49 * gap] = + 17./8. * ( + t[88]) + 1./2. * ( + t[4] + t[32]) + 2./1. * ( + t[10] + t[34]) + 4./1. * ( + t[8] + t[26]) + 5./1. * ( + t[7] + t[15]) + 1./1. * ( + t[6] + t[14]) + 17./2. * ( + t[103]) + 17./1. * ( + t[71]) + 17./4. * ( + t[43]) + 5./2. * ( + t[9] + t[25]) + 85./4. * ( + t[63]) + 85./8. * ( + t[89]);
		dst[bIdx + 50 * gap] = + 17./8. * ( + t[96]) + 1./2. * ( + t[4] + t[22]) + 2./1. * ( + t[10] + t[12]) + 4./1. * ( + t[8] + t[37]) + 5./1. * ( + t[7] + t[20]) + 1./1. * ( + t[6] + t[33]) + 17./2. * ( + t[97]) + 17./1. * ( + t[56]) + 17./4. * ( + t[66]) + 5./2. * ( + t[9] + t[27]) + 85./4. * ( + t[75]) + 85./8. * ( + t[104]);
		dst[bIdx + 51 * gap] = + 1./1. * ( + t[3] + t[79]) + 2./1. * ( - src[sIdx + 1 * dataW + 6] + t[36]) + 4./1. * ( + t[57]) + 5./1. * ( + t[5] + t[73]) + 1./2. * ( + src[sIdx + 6 * dataW + 1] + t[61]) + 8./1. * ( + src[sIdx + 2 * dataW + 5]) + 10./1. * ( + t[40]) + 5./8. * ( + t[53]) + 1./8. * ( - src[sIdx + 5 * dataW + 2]) + 5./4. * ( + t[11] + t[60]) + 25./2. * ( + src[sIdx + 4 * dataW + 3]) + 5./2. * ( + t[35] + t[67]) + 25./8. * ( - src[sIdx + 3 * dataW + 4]) + 1./4. * ( + t[77]) + 25./4. * ( + t[28]);
		dst[bIdx + 52 * gap] = + 1./1. * ( + t[3] + t[30]) + 2./1. * ( - src[sIdx + 1 * dataW + 6] + t[49]) + 4./1. * ( + t[65]) + 5./1. * ( + t[5] + t[74]) + 1./2. * ( - src[sIdx + 6 * dataW + 1] + t[61]) + 8./1. * ( - src[sIdx + 2 * dataW + 5]) + 10./1. * ( + t[72]) + 5./8. * ( + t[53]) + 1./8. * ( - src[sIdx + 5 * dataW + 2]) + 5./4. * ( + t[11] + t[45]) + 25./2. * ( - src[sIdx + 4 * dataW + 3]) + 5./2. * ( + t[67] + t[80]) + 25./8. * ( - src[sIdx + 3 * dataW + 4]) + 1./4. * ( + t[59]) + 25./4. * ( + t[29]);
		dst[bIdx + 53 * gap] = + 1./1. * ( + src[sIdx + 6 * dataW + 6] + t[52]) + 2./1. * ( + t[58] + t[62]) + 4./1. * ( - src[sIdx + 1 * dataW + 1] + t[1]) + 5./1. * ( + t[2] + t[44]) + 1./2. * ( + t[87]) + 8./1. * ( + t[105]) + 10./1. * ( + t[51] + t[70]) + 16./1. * ( + src[sIdx + 2 * dataW + 2]) + 5./4. * ( + t[64]) + 25./2. * ( + t[83]) + 20./1. * ( + t[0]) + 5./2. * ( + t[48] + t[76]) + 25./1. * ( + src[sIdx + 4 * dataW + 4]) + 1./4. * ( - src[sIdx + 5 * dataW + 5]) + 25./4. * ( - src[sIdx + 3 * dataW + 3]);
		dst[bIdx + 54 * gap] = + 1./1. * ( + src[sIdx + 6 * dataW + 6] + t[13]) + 2./1. * ( + t[46] + t[62]) + 4./1. * ( + src[sIdx + 1 * dataW + 1] + t[1]) + 5./1. * ( + t[2] + t[24]) + 1./2. * ( + t[100]) + 8./1. * ( + t[86]) + 10./1. * ( + t[51] + t[68]) + 16./1. * ( + src[sIdx + 2 * dataW + 2]) + 5./4. * ( + t[19]) + 25./2. * ( + t[81]) + 20./1. * ( + t[0]) + 5./2. * ( + t[47] + t[48]) + 25./1. * ( + src[sIdx + 4 * dataW + 4]) + 1./4. * ( + src[sIdx + 5 * dataW + 5]) + 25./4. * ( + src[sIdx + 3 * dataW + 3]);
		t[1] = + src[sIdx + 6 * dataW + 0] - src[sIdx + 6 * dataW + 6];
		t[0] = + src[sIdx + 2 * dataW + 0] - src[sIdx + 2 * dataW + 6];
		t[2] = - src[sIdx + 6 * dataW + 2] + src[sIdx + 6 * dataW + 4];
		t[5] = + src[sIdx + 4 * dataW + 2] - src[sIdx + 4 * dataW + 4];
		t[15] = + src[sIdx + 3 * dataW + 2] - src[sIdx + 3 * dataW + 4];
		t[3] = - src[sIdx + 4 * dataW + 0] + src[sIdx + 4 * dataW + 6];
		t[6] = - src[sIdx + 2 * dataW + 2] + src[sIdx + 2 * dataW + 4];
		t[11] = - src[sIdx + 3 * dataW + 0] + src[sIdx + 3 * dataW + 6];
		dst[bIdx + 8 * gap] = + 21./4. * ( + t[18] + t[2] + t[6] + t[41]) + 1./1. * ( + src[sIdx + 1 * dataW + 0] + src[sIdx + 5 * dataW + 0] + t[0] + t[1] + t[42]) + 357./16. * ( + t[5] + t[15]) + 17./4. * ( + t[3] + t[11]);
		t[8] = + src[sIdx + 3 * dataW + 0] - src[sIdx + 3 * dataW + 6];
		t[7] = - src[sIdx + 3 * dataW + 2] + src[sIdx + 3 * dataW + 4];
		dst[bIdx + 16 * gap] = + 21./4. * ( + t[16] + t[23] + t[2] + t[6]) + 1./1. * ( - src[sIdx + 1 * dataW + 0] - src[sIdx + 5 * dataW + 0] + t[21] + t[0] + t[1]) + 357./16. * ( + t[5] + t[7]) + 17./4. * ( + t[3] + t[8]);
		t[26] = - src[sIdx + 5 * dataW + 2] + src[sIdx + 5 * dataW + 4];
		t[23] = + src[sIdx + 5 * dataW + 0] - src[sIdx + 5 * dataW + 6];
		t[28] = + src[sIdx + 1 * dataW + 0] - src[sIdx + 1 * dataW + 6];
		t[21] = - src[sIdx + 1 * dataW + 2] + src[sIdx + 1 * dataW + 4];
		dst[bIdx + 24 * gap] = + 105./16. * ( + t[5]) + 1./2. * ( + t[28]) + 2./1. * ( + t[23]) + 1./4. * ( + t[0]) + 21./2. * ( + t[26]) + 1./1. * ( + t[1]) + 105./8. * ( + t[15]) + 5./4. * ( + t[3]) + 5./2. * ( + t[11]) + 21./4. * ( + t[2]) + 21./16. * ( + t[6]) + 21./8. * ( + t[21]);
		t[14] = - src[sIdx + 1 * dataW + 0] + src[sIdx + 1 * dataW + 6];
		t[20] = + src[sIdx + 1 * dataW + 2] - src[sIdx + 1 * dataW + 4];
		t[16] = - src[sIdx + 5 * dataW + 0] + src[sIdx + 5 * dataW + 6];
		t[18] = + src[sIdx + 5 * dataW + 2] - src[sIdx + 5 * dataW + 4];
		dst[bIdx + 32 * gap] = + 105./16. * ( + t[5]) + 1./2. * ( + t[14]) + 2./1. * ( + t[16]) + 1./4. * ( + t[0]) + 21./2. * ( + t[18]) + 1./1. * ( + t[1]) + 105./8. * ( + t[7]) + 5./4. * ( + t[3]) + 5./2. * ( + t[8]) + 21./4. * ( + t[2]) + 21./16. * ( + t[6]) + 21./8. * ( + t[20]);
		dst[bIdx + 40 * gap] = + 1./2. * ( + t[23]) + 2./1. * ( + t[28]) + 4./1. * ( + t[0]) + 5./1. * ( + t[3]) + 21./2. * ( + t[21]) + 1./1. * ( + t[1]) + 105./8. * ( + t[15]) + 105./4. * ( + t[5]) + 21./1. * ( + t[6]) + 5./2. * ( + t[11]) + 21./4. * ( + t[2]) + 21./8. * ( + t[26]);
		dst[bIdx + 48 * gap] = + 1./2. * ( + t[16]) + 2./1. * ( + t[14]) + 4./1. * ( + t[0]) + 5./1. * ( + t[3]) + 21./2. * ( + t[20]) + 1./1. * ( + t[1]) + 105./8. * ( + t[7]) + 105./4. * ( + t[5]) + 21./1. * ( + t[6]) + 5./2. * ( + t[8]) + 21./4. * ( + t[2]) + 21./8. * ( + t[18]);
		dst[bIdx + 56 * gap] = + 21./4. * ( + src[sIdx + 3 * dataW + 0] - src[sIdx + 5 * dataW + 0] - src[sIdx + 7 * dataW + 2] + src[sIdx + 7 * dataW + 4] + t[17] + t[38]) + 1./1. * ( + src[sIdx + 7 * dataW + 0] - src[sIdx + 7 * dataW + 6] + t[14]) + 441./16. * ( + src[sIdx + 3 * dataW + 4] + src[sIdx + 5 * dataW + 2] + t[39]);
		t[8] = + src[sIdx + 1 * dataW + 3] - src[sIdx + 7 * dataW + 3];
		t[2] = + src[sIdx + 7 * dataW + 2] + src[sIdx + 7 * dataW + 6];
		t[0] = - src[sIdx + 3 * dataW + 4] + src[sIdx + 5 * dataW + 4];
		t[1] = + src[sIdx + 1 * dataW + 4] - src[sIdx + 7 * dataW + 4];
		t[11] = - src[sIdx + 3 * dataW + 3] + src[sIdx + 5 * dataW + 3];
		dst[bIdx + 57 * gap] = + 21./4. * ( + t[4] + t[9] + t[25] + t[32]) + 1./1. * ( + src[sIdx + 7 * dataW + 1] + src[sIdx + 7 * dataW + 5] + t[10] + t[34] + t[2]) + 357./16. * ( + t[0] + t[11]) + 17./4. * ( + t[1] + t[8]);
		t[6] = + src[sIdx + 3 * dataW + 3] - src[sIdx + 5 * dataW + 3];
		t[7] = - src[sIdx + 1 * dataW + 3] + src[sIdx + 7 * dataW + 3];
		dst[bIdx + 58 * gap] = + 21./4. * ( + t[4] + t[9] + t[22] + t[27]) + 1./1. * ( - src[sIdx + 7 * dataW + 1] - src[sIdx + 7 * dataW + 5] + t[10] + t[12] + t[2]) + 357./16. * ( + t[0] + t[6]) + 17./4. * ( + t[1] + t[7]);
		t[17] = + src[sIdx + 3 * dataW + 1] - src[sIdx + 5 * dataW + 1];
		t[4] = - src[sIdx + 1 * dataW + 6] + src[sIdx + 7 * dataW + 6];
		t[2] = - src[sIdx + 1 * dataW + 2] + src[sIdx + 7 * dataW + 2];
		t[3] = + src[sIdx + 3 * dataW + 2] - src[sIdx + 5 * dataW + 2];
		t[16] = + src[sIdx + 3 * dataW + 5] - src[sIdx + 5 * dataW + 5];
		t[9] = - src[sIdx + 1 * dataW + 5] + src[sIdx + 7 * dataW + 5];
		t[18] = - src[sIdx + 1 * dataW + 1] + src[sIdx + 7 * dataW + 1];
		t[5] = + src[sIdx + 3 * dataW + 6] - src[sIdx + 5 * dataW + 6];
		dst[bIdx + 59 * gap] = + 105./16. * ( + t[0]) + 1./2. * ( + t[18]) + 2./1. * ( + t[9]) + 1./4. * ( + t[2]) + 21./2. * ( + t[16]) + 1./1. * ( + t[4]) + 5./4. * ( + t[1]) + 105./8. * ( + t[11]) + 5./2. * ( + t[8]) + 21./4. * ( + t[5]) + 21./16. * ( + t[3]) + 21./8. * ( + t[17]);
		t[12] = + src[sIdx + 1 * dataW + 1] - src[sIdx + 7 * dataW + 1];
		t[10] = - src[sIdx + 3 * dataW + 1] + src[sIdx + 5 * dataW + 1];
		t[14] = + src[sIdx + 1 * dataW + 5] - src[sIdx + 7 * dataW + 5];
		t[15] = - src[sIdx + 3 * dataW + 5] + src[sIdx + 5 * dataW + 5];
		dst[bIdx + 60 * gap] = + 105./16. * ( + t[0]) + 1./2. * ( + t[12]) + 2./1. * ( + t[14]) + 1./4. * ( + t[2]) + 21./2. * ( + t[15]) + 1./1. * ( + t[4]) + 5./4. * ( + t[1]) + 105./8. * ( + t[6]) + 5./2. * ( + t[7]) + 21./4. * ( + t[5]) + 21./16. * ( + t[3]) + 21./8. * ( + t[10]);
		dst[bIdx + 61 * gap] = + 1./2. * ( + t[9]) + 2./1. * ( + t[18]) + 4./1. * ( + t[2]) + 5./1. * ( + t[1]) + 21./2. * ( + t[17]) + 1./1. * ( + t[4]) + 105./8. * ( + t[11]) + 105./4. * ( + t[0]) + 21./1. * ( + t[3]) + 5./2. * ( + t[8]) + 21./4. * ( + t[5]) + 21./8. * ( + t[16]);
		dst[bIdx + 62 * gap] = + 1./2. * ( + t[14]) + 2./1. * ( + t[12]) + 4./1. * ( + t[2]) + 5./1. * ( + t[1]) + 21./2. * ( + t[10]) + 1./1. * ( + t[4]) + 105./8. * ( + t[6]) + 105./4. * ( + t[0]) + 21./1. * ( + t[3]) + 5./2. * ( + t[7]) + 21./4. * ( + t[5]) + 21./8. * ( + t[15]);
		dst[bIdx + 63 * gap] = + 21./4. * ( + src[sIdx + 3 * dataW + 7] - src[sIdx + 5 * dataW + 7] + src[sIdx + 7 * dataW + 3] - src[sIdx + 7 * dataW + 5] + t[13] + t[24]) + 1./1. * ( - src[sIdx + 7 * dataW + 1] + src[sIdx + 7 * dataW + 7] + t[84]) + 441./16. * ( + src[sIdx + 3 * dataW + 3] + src[sIdx + 5 * dataW + 5] + t[19]);
	}
}



template <typename Dtype> 
__global__ void winoMulti_gpu_kernel(const Dtype *u_matrix, const Dtype *v_matrix, Dtype *m_matrix, const int M, const int N, const int K)
{
	const Dtype *A = u_matrix + blockIdx.z * M * K;
	const Dtype *B = v_matrix + blockIdx.z * K * N;
	Dtype *C = m_matrix + blockIdx.z * M * N;

	int br = blockIdx.y, bc = blockIdx.x;
	int tr = threadIdx.y, tc = threadIdx.x;
	int Cr = br * BLOCK_SIZE + tr;
	int Cc = bc * BLOCK_SIZE + tc;
	Dtype s = 0;
	int BN = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
	for (int i = 0; i < BN; ++i) {
		__shared__ float a[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float b[BLOCK_SIZE][BLOCK_SIZE];
		int Ar = Cr, Ac = i * BLOCK_SIZE + tc;
		if (Ar < M && Ac < K)
			a[tr][tc] = A[Ar * K + Ac];
		else
			a[tr][tc] = 0;
		int Br = i * BLOCK_SIZE + tr, Bc = Cc;
		if (Br < K && Bc < N)
			b[tr][tc] = B[Br * N + Bc];
		else 
			b[tr][tc] = 0;
		__syncthreads();
		for (int j = 0; j < BLOCK_SIZE; ++j)
			s += a[tr][j] * b[j][tc];
		__syncthreads();
	}
	if (Cr < M && Cc < N)
		C[Cr * N + Cc] = s;
}


template <typename Dtype> 
__global__ void winoDst_gpu_kernel(const Dtype *src, Dtype * dst, const int tileH, const int tileW, const int outH, const int outW, const int outputs, const int batchs, const int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int highIdx = idx / (tileW * tileH);
		int yIdx = (idx % (tileW * tileH)) / tileW;
		int xIdx = idx % tileW;
		int rIdx = highIdx * outW * outH + yIdx * outW * 2 + xIdx * 2;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;

        dst[rIdx + 0 * outW + 0] = + 1./1. * ( + src[mIdx + 0 * gap] + src[mIdx + 1 * gap] + src[mIdx + 2 * gap] + src[mIdx + 4 * gap] + src[mIdx + 5 * gap] + src[mIdx + 6 * gap] + src[mIdx + 8 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap]);
        dst[rIdx + 0 * outW + 1] = + 1./1. * ( + src[mIdx + 1 * gap] - src[mIdx + 2 * gap] + src[mIdx + 3 * gap] + src[mIdx + 5 * gap] - src[mIdx + 6 * gap] + src[mIdx + 7 * gap] + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 11 * gap]);
        dst[rIdx + 1 * outW + 0] = + 1./1. * ( + src[mIdx + 4 * gap] + src[mIdx + 5 * gap] + src[mIdx + 6 * gap] - src[mIdx + 8 * gap] - src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 12 * gap] + src[mIdx + 13 * gap] + src[mIdx + 14 * gap]);
        dst[rIdx + 1 * outW + 1] = + 1./1. * ( + src[mIdx + 5 * gap] - src[mIdx + 6 * gap] + src[mIdx + 7 * gap] - src[mIdx + 9 * gap] + src[mIdx + 10 * gap] - src[mIdx + 11 * gap] + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 15 * gap]);
	}
}

template <typename Dtype> 
__global__ void wino4x4Dst_gpu_kernel(const Dtype *src, Dtype * dst, const int tileH, const int tileW, const int outH, const int outW, const int outputs, const int batchs, const int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int highIdx = idx / (tileW * tileH);
		int yIdx = (idx % (tileW * tileH)) / tileW;
		int xIdx = idx % tileW;
		int rIdx = highIdx * outW * outH + yIdx * outW * 4 + xIdx * 4;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;

        dst[rIdx + 0 * outW + 0] = + 1./1. * ( + src[mIdx + 0 * gap] + src[mIdx + 1 * gap] + src[mIdx + 2 * gap] + src[mIdx + 3 * gap] + src[mIdx + 4 * gap] + src[mIdx + 6 * gap] + src[mIdx + 7 * gap] + src[mIdx + 8 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 12 * gap] + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] + src[mIdx + 15 * gap] + src[mIdx + 16 * gap] + src[mIdx + 18 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 24 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 27 * gap] + src[mIdx + 28 * gap]);
        dst[rIdx + 0 * outW + 1] = + 1./1. * ( + src[mIdx + 1 * gap] - src[mIdx + 2 * gap] + src[mIdx + 7 * gap] - src[mIdx + 8 * gap] + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 19 * gap] - src[mIdx + 20 * gap] + src[mIdx + 25 * gap] - src[mIdx + 26 * gap]) + 2./1. * ( + src[mIdx + 3 * gap] - src[mIdx + 4 * gap] + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 15 * gap] - src[mIdx + 16 * gap] + src[mIdx + 21 * gap] - src[mIdx + 22 * gap] + src[mIdx + 27 * gap] - src[mIdx + 28 * gap]);
        dst[rIdx + 0 * outW + 2] = + 1./1. * ( + src[mIdx + 1 * gap] + src[mIdx + 2 * gap] + src[mIdx + 7 * gap] + src[mIdx + 8 * gap] + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap]) + 4./1. * ( + src[mIdx + 3 * gap] + src[mIdx + 4 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 15 * gap] + src[mIdx + 16 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 27 * gap] + src[mIdx + 28 * gap]);
        dst[rIdx + 0 * outW + 3] = + 8./1. * ( + src[mIdx + 3 * gap] - src[mIdx + 4 * gap] + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 15 * gap] - src[mIdx + 16 * gap] + src[mIdx + 21 * gap] - src[mIdx + 22 * gap] + src[mIdx + 27 * gap] - src[mIdx + 28 * gap]) + 1./1. * ( + src[mIdx + 1 * gap] - src[mIdx + 2 * gap] + src[mIdx + 5 * gap] + src[mIdx + 7 * gap] - src[mIdx + 8 * gap] + src[mIdx + 11 * gap] + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 17 * gap] + src[mIdx + 19 * gap] - src[mIdx + 20 * gap] + src[mIdx + 23 * gap] + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 29 * gap]);
        dst[rIdx + 1 * outW + 0] = + 1./1. * ( + src[mIdx + 6 * gap] + src[mIdx + 7 * gap] + src[mIdx + 8 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] - src[mIdx + 12 * gap] - src[mIdx + 13 * gap] - src[mIdx + 14 * gap] - src[mIdx + 15 * gap] - src[mIdx + 16 * gap]) + 2./1. * ( + src[mIdx + 18 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] - src[mIdx + 24 * gap] - src[mIdx + 25 * gap] - src[mIdx + 26 * gap] - src[mIdx + 27 * gap] - src[mIdx + 28 * gap]);
        dst[rIdx + 1 * outW + 1] = + 1./1. * ( + src[mIdx + 7 * gap] - src[mIdx + 8 * gap] - src[mIdx + 13 * gap] + src[mIdx + 14 * gap]) + 2./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] - src[mIdx + 15 * gap] + src[mIdx + 16 * gap] + src[mIdx + 19 * gap] - src[mIdx + 20 * gap] - src[mIdx + 25 * gap] + src[mIdx + 26 * gap]) + 4./1. * ( + src[mIdx + 21 * gap] - src[mIdx + 22 * gap] - src[mIdx + 27 * gap] + src[mIdx + 28 * gap]);
        dst[rIdx + 1 * outW + 2] = + 8./1. * ( + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] - src[mIdx + 27 * gap] - src[mIdx + 28 * gap]) + 1./1. * ( + src[mIdx + 7 * gap] + src[mIdx + 8 * gap] - src[mIdx + 13 * gap] - src[mIdx + 14 * gap]) + 2./1. * ( + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] - src[mIdx + 25 * gap] - src[mIdx + 26 * gap]) + 4./1. * ( + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] - src[mIdx + 15 * gap] - src[mIdx + 16 * gap]);
        dst[rIdx + 1 * outW + 3] = + 8./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] - src[mIdx + 15 * gap] + src[mIdx + 16 * gap]) + 1./1. * ( + src[mIdx + 7 * gap] - src[mIdx + 8 * gap] + src[mIdx + 11 * gap] - src[mIdx + 13 * gap] + src[mIdx + 14 * gap] - src[mIdx + 17 * gap]) + 2./1. * ( + src[mIdx + 19 * gap] - src[mIdx + 20 * gap] + src[mIdx + 23 * gap] - src[mIdx + 25 * gap] + src[mIdx + 26 * gap] - src[mIdx + 29 * gap]) + 16./1. * ( + src[mIdx + 21 * gap] - src[mIdx + 22 * gap] - src[mIdx + 27 * gap] + src[mIdx + 28 * gap]);
        dst[rIdx + 2 * outW + 0] = + 1./1. * ( + src[mIdx + 6 * gap] + src[mIdx + 7 * gap] + src[mIdx + 8 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 12 * gap] + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] + src[mIdx + 15 * gap] + src[mIdx + 16 * gap]) + 4./1. * ( + src[mIdx + 18 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 24 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 27 * gap] + src[mIdx + 28 * gap]);
        dst[rIdx + 2 * outW + 1] = + 8./1. * ( + src[mIdx + 21 * gap] - src[mIdx + 22 * gap] + src[mIdx + 27 * gap] - src[mIdx + 28 * gap]) + 1./1. * ( + src[mIdx + 7 * gap] - src[mIdx + 8 * gap] + src[mIdx + 13 * gap] - src[mIdx + 14 * gap]) + 2./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 15 * gap] - src[mIdx + 16 * gap]) + 4./1. * ( + src[mIdx + 19 * gap] - src[mIdx + 20 * gap] + src[mIdx + 25 * gap] - src[mIdx + 26 * gap]);
        dst[rIdx + 2 * outW + 2] = + 16./1. * ( + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 27 * gap] + src[mIdx + 28 * gap]) + 1./1. * ( + src[mIdx + 7 * gap] + src[mIdx + 8 * gap] + src[mIdx + 13 * gap] + src[mIdx + 14 * gap]) + 4./1. * ( + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 15 * gap] + src[mIdx + 16 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap]);
        dst[rIdx + 2 * outW + 3] = + 8./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 15 * gap] - src[mIdx + 16 * gap]) + 1./1. * ( + src[mIdx + 7 * gap] - src[mIdx + 8 * gap] + src[mIdx + 11 * gap] + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 17 * gap]) + 4./1. * ( + src[mIdx + 19 * gap] - src[mIdx + 20 * gap] + src[mIdx + 23 * gap] + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 29 * gap]) + 32./1. * ( + src[mIdx + 21 * gap] - src[mIdx + 22 * gap] + src[mIdx + 27 * gap] - src[mIdx + 28 * gap]);
        dst[rIdx + 3 * outW + 0] = + 8./1. * ( + src[mIdx + 18 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] - src[mIdx + 24 * gap] - src[mIdx + 25 * gap] - src[mIdx + 26 * gap] - src[mIdx + 27 * gap] - src[mIdx + 28 * gap]) + 1./1. * ( + src[mIdx + 6 * gap] + src[mIdx + 7 * gap] + src[mIdx + 8 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] - src[mIdx + 12 * gap] - src[mIdx + 13 * gap] - src[mIdx + 14 * gap] - src[mIdx + 15 * gap] - src[mIdx + 16 * gap] + src[mIdx + 30 * gap] + src[mIdx + 31 * gap] + src[mIdx + 32 * gap] + src[mIdx + 33 * gap] + src[mIdx + 34 * gap]);
        dst[rIdx + 3 * outW + 1] = + 8./1. * ( + src[mIdx + 19 * gap] - src[mIdx + 20 * gap] - src[mIdx + 25 * gap] + src[mIdx + 26 * gap]) + 1./1. * ( + src[mIdx + 7 * gap] - src[mIdx + 8 * gap] - src[mIdx + 13 * gap] + src[mIdx + 14 * gap] + src[mIdx + 31 * gap] - src[mIdx + 32 * gap]) + 2./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] - src[mIdx + 15 * gap] + src[mIdx + 16 * gap] + src[mIdx + 33 * gap] - src[mIdx + 34 * gap]) + 16./1. * ( + src[mIdx + 21 * gap] - src[mIdx + 22 * gap] - src[mIdx + 27 * gap] + src[mIdx + 28 * gap]);
        dst[rIdx + 3 * outW + 2] = + 8./1. * ( + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] - src[mIdx + 25 * gap] - src[mIdx + 26 * gap]) + 1./1. * ( + src[mIdx + 7 * gap] + src[mIdx + 8 * gap] - src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 31 * gap] + src[mIdx + 32 * gap]) + 4./1. * ( + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] - src[mIdx + 15 * gap] - src[mIdx + 16 * gap] + src[mIdx + 33 * gap] + src[mIdx + 34 * gap]) + 32./1. * ( + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] - src[mIdx + 27 * gap] - src[mIdx + 28 * gap]);
        dst[rIdx + 3 * outW + 3] = + 8./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] - src[mIdx + 15 * gap] + src[mIdx + 16 * gap] + src[mIdx + 19 * gap] - src[mIdx + 20 * gap] + src[mIdx + 23 * gap] - src[mIdx + 25 * gap] + src[mIdx + 26 * gap] - src[mIdx + 29 * gap] + src[mIdx + 33 * gap] - src[mIdx + 34 * gap]) + 1./1. * ( + src[mIdx + 7 * gap] - src[mIdx + 8 * gap] + src[mIdx + 11 * gap] - src[mIdx + 13 * gap] + src[mIdx + 14 * gap] - src[mIdx + 17 * gap] + src[mIdx + 31 * gap] - src[mIdx + 32 * gap] + src[mIdx + 35 * gap]) + 64./1. * ( + src[mIdx + 21 * gap] - src[mIdx + 22 * gap] - src[mIdx + 27 * gap] + src[mIdx + 28 * gap]);
	}
}

template <typename Dtype> 
__global__ void wino6x6Dst_gpu_kernel(const Dtype *src, Dtype * dst, const int tileH, const int tileW, const int outH, const int outW, const int outputs, const int batchs, const int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int highIdx = idx / (tileW * tileH);
		int yIdx = (idx % (tileW * tileH)) / tileW;
		int xIdx = idx % tileW;
		int rIdx = highIdx * outW * outH + yIdx * outW * 6 + xIdx * 6;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;

        dst[rIdx + 0 * outW + 0] = + 1./1. * ( + src[mIdx + 0 * gap] + src[mIdx + 1 * gap] + src[mIdx + 2 * gap] + src[mIdx + 3 * gap] + src[mIdx + 4 * gap] + src[mIdx + 5 * gap] + src[mIdx + 6 * gap] + src[mIdx + 8 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] + src[mIdx + 16 * gap] + src[mIdx + 17 * gap] + src[mIdx + 18 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 24 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] + src[mIdx + 32 * gap] + src[mIdx + 33 * gap] + src[mIdx + 34 * gap] + src[mIdx + 35 * gap] + src[mIdx + 36 * gap] + src[mIdx + 37 * gap] + src[mIdx + 38 * gap] + src[mIdx + 40 * gap] + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] + src[mIdx + 48 * gap] + src[mIdx + 49 * gap] + src[mIdx + 50 * gap] + src[mIdx + 51 * gap] + src[mIdx + 52 * gap] + src[mIdx + 53 * gap] + src[mIdx + 54 * gap]);
        dst[rIdx + 0 * outW + 1] = + 1./1. * ( + src[mIdx + 1 * gap] - src[mIdx + 2 * gap] + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 17 * gap] - src[mIdx + 18 * gap] + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 33 * gap] - src[mIdx + 34 * gap] + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] + src[mIdx + 49 * gap] - src[mIdx + 50 * gap]) + 2./1. * ( + src[mIdx + 3 * gap] - src[mIdx + 4 * gap] + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] + src[mIdx + 19 * gap] - src[mIdx + 20 * gap] + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] + src[mIdx + 35 * gap] - src[mIdx + 36 * gap] + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] + src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 1./2. * ( + src[mIdx + 5 * gap] - src[mIdx + 6 * gap] + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 21 * gap] - src[mIdx + 22 * gap] + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] + src[mIdx + 37 * gap] - src[mIdx + 38 * gap] + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] + src[mIdx + 53 * gap] - src[mIdx + 54 * gap]);
        dst[rIdx + 0 * outW + 2] = + 1./1. * ( + src[mIdx + 1 * gap] + src[mIdx + 2 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 17 * gap] + src[mIdx + 18 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 33 * gap] + src[mIdx + 34 * gap] + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] + src[mIdx + 49 * gap] + src[mIdx + 50 * gap]) + 1./4. * ( + src[mIdx + 5 * gap] + src[mIdx + 6 * gap] + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] + src[mIdx + 37 * gap] + src[mIdx + 38 * gap] + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] + src[mIdx + 53 * gap] + src[mIdx + 54 * gap]) + 4./1. * ( + src[mIdx + 3 * gap] + src[mIdx + 4 * gap] + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] + src[mIdx + 35 * gap] + src[mIdx + 36 * gap] + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] + src[mIdx + 51 * gap] + src[mIdx + 52 * gap]);
        dst[rIdx + 0 * outW + 3] = + 8./1. * ( + src[mIdx + 3 * gap] - src[mIdx + 4 * gap] + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] + src[mIdx + 19 * gap] - src[mIdx + 20 * gap] + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] + src[mIdx + 35 * gap] - src[mIdx + 36 * gap] + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] + src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 1./1. * ( + src[mIdx + 1 * gap] - src[mIdx + 2 * gap] + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 17 * gap] - src[mIdx + 18 * gap] + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 33 * gap] - src[mIdx + 34 * gap] + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] + src[mIdx + 49 * gap] - src[mIdx + 50 * gap]) + 1./8. * ( + src[mIdx + 5 * gap] - src[mIdx + 6 * gap] + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 21 * gap] - src[mIdx + 22 * gap] + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] + src[mIdx + 37 * gap] - src[mIdx + 38 * gap] + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] + src[mIdx + 53 * gap] - src[mIdx + 54 * gap]);
        dst[rIdx + 0 * outW + 4] = + 16./1. * ( + src[mIdx + 3 * gap] + src[mIdx + 4 * gap] + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] + src[mIdx + 35 * gap] + src[mIdx + 36 * gap] + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] + src[mIdx + 51 * gap] + src[mIdx + 52 * gap]) + 1./1. * ( + src[mIdx + 1 * gap] + src[mIdx + 2 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 17 * gap] + src[mIdx + 18 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 33 * gap] + src[mIdx + 34 * gap] + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] + src[mIdx + 49 * gap] + src[mIdx + 50 * gap]) + 1./16. * ( + src[mIdx + 5 * gap] + src[mIdx + 6 * gap] + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] + src[mIdx + 37 * gap] + src[mIdx + 38 * gap] + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] + src[mIdx + 53 * gap] + src[mIdx + 54 * gap]);
        dst[rIdx + 0 * outW + 5] = + 32./1. * ( + src[mIdx + 3 * gap] - src[mIdx + 4 * gap] + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] + src[mIdx + 19 * gap] - src[mIdx + 20 * gap] + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] + src[mIdx + 35 * gap] - src[mIdx + 36 * gap] + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] + src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 1./1. * ( + src[mIdx + 1 * gap] - src[mIdx + 2 * gap] + src[mIdx + 7 * gap] + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 15 * gap] + src[mIdx + 17 * gap] - src[mIdx + 18 * gap] + src[mIdx + 23 * gap] + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 31 * gap] + src[mIdx + 33 * gap] - src[mIdx + 34 * gap] + src[mIdx + 39 * gap] + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] + src[mIdx + 47 * gap] + src[mIdx + 49 * gap] - src[mIdx + 50 * gap] + src[mIdx + 55 * gap]) + 1./32. * ( + src[mIdx + 5 * gap] - src[mIdx + 6 * gap] + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 21 * gap] - src[mIdx + 22 * gap] + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] + src[mIdx + 37 * gap] - src[mIdx + 38 * gap] + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] + src[mIdx + 53 * gap] - src[mIdx + 54 * gap]);
        dst[rIdx + 1 * outW + 0] = + 1./1. * ( + src[mIdx + 8 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] - src[mIdx + 16 * gap] - src[mIdx + 17 * gap] - src[mIdx + 18 * gap] - src[mIdx + 19 * gap] - src[mIdx + 20 * gap] - src[mIdx + 21 * gap] - src[mIdx + 22 * gap]) + 2./1. * ( + src[mIdx + 24 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] - src[mIdx + 32 * gap] - src[mIdx + 33 * gap] - src[mIdx + 34 * gap] - src[mIdx + 35 * gap] - src[mIdx + 36 * gap] - src[mIdx + 37 * gap] - src[mIdx + 38 * gap]) + 1./2. * ( + src[mIdx + 40 * gap] + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] - src[mIdx + 48 * gap] - src[mIdx + 49 * gap] - src[mIdx + 50 * gap] - src[mIdx + 51 * gap] - src[mIdx + 52 * gap] - src[mIdx + 53 * gap] - src[mIdx + 54 * gap]);
        dst[rIdx + 1 * outW + 1] = + 1./4. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] - src[mIdx + 53 * gap] + src[mIdx + 54 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] - src[mIdx + 17 * gap] + src[mIdx + 18 * gap] + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] - src[mIdx + 37 * gap] + src[mIdx + 38 * gap] + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] - src[mIdx + 51 * gap] + src[mIdx + 52 * gap]) + 2./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] - src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] - src[mIdx + 33 * gap] + src[mIdx + 34 * gap]) + 4./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] - src[mIdx + 35 * gap] + src[mIdx + 36 * gap]) + 1./2. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] - src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] - src[mIdx + 49 * gap] + src[mIdx + 50 * gap]);
        dst[rIdx + 1 * outW + 2] = + 1./1. * ( + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] - src[mIdx + 17 * gap] - src[mIdx + 18 * gap]) + 2./1. * ( + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] - src[mIdx + 33 * gap] - src[mIdx + 34 * gap] + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] - src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 1./4. * ( + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] - src[mIdx + 21 * gap] - src[mIdx + 22 * gap]) + 1./2. * ( + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] - src[mIdx + 37 * gap] - src[mIdx + 38 * gap] + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] - src[mIdx + 49 * gap] - src[mIdx + 50 * gap]) + 8./1. * ( + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] - src[mIdx + 35 * gap] - src[mIdx + 36 * gap]) + 1./8. * ( + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] - src[mIdx + 53 * gap] - src[mIdx + 54 * gap]) + 4./1. * ( + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] - src[mIdx + 19 * gap] - src[mIdx + 20 * gap]);
        dst[rIdx + 1 * outW + 3] = + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] - src[mIdx + 17 * gap] + src[mIdx + 18 * gap]) + 2./1. * ( + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] - src[mIdx + 33 * gap] + src[mIdx + 34 * gap]) + 1./4. * ( + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] - src[mIdx + 37 * gap] + src[mIdx + 38 * gap]) + 1./2. * ( + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] - src[mIdx + 49 * gap] + src[mIdx + 50 * gap]) + 8./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] - src[mIdx + 19 * gap] + src[mIdx + 20 * gap]) + 16./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] - src[mIdx + 35 * gap] + src[mIdx + 36 * gap]) + 1./16. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] - src[mIdx + 53 * gap] + src[mIdx + 54 * gap]) + 4./1. * ( + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] - src[mIdx + 51 * gap] + src[mIdx + 52 * gap]) + 1./8. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] - src[mIdx + 21 * gap] + src[mIdx + 22 * gap]);
        dst[rIdx + 1 * outW + 4] = + 32./1. * ( + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] - src[mIdx + 35 * gap] - src[mIdx + 36 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] - src[mIdx + 17 * gap] - src[mIdx + 18 * gap]) + 2./1. * ( + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] - src[mIdx + 33 * gap] - src[mIdx + 34 * gap]) + 1./2. * ( + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] - src[mIdx + 49 * gap] - src[mIdx + 50 * gap]) + 8./1. * ( + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] - src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 16./1. * ( + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] - src[mIdx + 19 * gap] - src[mIdx + 20 * gap]) + 1./32. * ( + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] - src[mIdx + 53 * gap] - src[mIdx + 54 * gap]) + 1./16. * ( + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] - src[mIdx + 21 * gap] - src[mIdx + 22 * gap]) + 1./8. * ( + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] - src[mIdx + 37 * gap] - src[mIdx + 38 * gap]);
        dst[rIdx + 1 * outW + 5] = + 32./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] - src[mIdx + 19 * gap] + src[mIdx + 20 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 15 * gap] - src[mIdx + 17 * gap] + src[mIdx + 18 * gap] - src[mIdx + 23 * gap]) + 2./1. * ( + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 31 * gap] - src[mIdx + 33 * gap] + src[mIdx + 34 * gap] - src[mIdx + 39 * gap]) + 1./2. * ( + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] + src[mIdx + 47 * gap] - src[mIdx + 49 * gap] + src[mIdx + 50 * gap] - src[mIdx + 55 * gap]) + 1./32. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] - src[mIdx + 21 * gap] + src[mIdx + 22 * gap]) + 64./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] - src[mIdx + 35 * gap] + src[mIdx + 36 * gap]) + 16./1. * ( + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] - src[mIdx + 51 * gap] + src[mIdx + 52 * gap]) + 1./64. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] - src[mIdx + 53 * gap] + src[mIdx + 54 * gap]) + 1./16. * ( + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] - src[mIdx + 37 * gap] + src[mIdx + 38 * gap]);
        dst[rIdx + 2 * outW + 0] = + 1./1. * ( + src[mIdx + 8 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] + src[mIdx + 16 * gap] + src[mIdx + 17 * gap] + src[mIdx + 18 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap]) + 1./4. * ( + src[mIdx + 40 * gap] + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] + src[mIdx + 48 * gap] + src[mIdx + 49 * gap] + src[mIdx + 50 * gap] + src[mIdx + 51 * gap] + src[mIdx + 52 * gap] + src[mIdx + 53 * gap] + src[mIdx + 54 * gap]) + 4./1. * ( + src[mIdx + 24 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] + src[mIdx + 32 * gap] + src[mIdx + 33 * gap] + src[mIdx + 34 * gap] + src[mIdx + 35 * gap] + src[mIdx + 36 * gap] + src[mIdx + 37 * gap] + src[mIdx + 38 * gap]);
        dst[rIdx + 2 * outW + 1] = + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 17 * gap] - src[mIdx + 18 * gap]) + 2./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] + src[mIdx + 19 * gap] - src[mIdx + 20 * gap] + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] + src[mIdx + 37 * gap] - src[mIdx + 38 * gap]) + 4./1. * ( + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 33 * gap] - src[mIdx + 34 * gap]) + 1./2. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 21 * gap] - src[mIdx + 22 * gap] + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] + src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 8./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] + src[mIdx + 35 * gap] - src[mIdx + 36 * gap]) + 1./8. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] + src[mIdx + 53 * gap] - src[mIdx + 54 * gap]) + 1./4. * ( + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] + src[mIdx + 49 * gap] - src[mIdx + 50 * gap]);
        dst[rIdx + 2 * outW + 2] = + 16./1. * ( + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] + src[mIdx + 35 * gap] + src[mIdx + 36 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 17 * gap] + src[mIdx + 18 * gap] + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] + src[mIdx + 37 * gap] + src[mIdx + 38 * gap] + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] + src[mIdx + 51 * gap] + src[mIdx + 52 * gap]) + 1./4. * ( + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] + src[mIdx + 49 * gap] + src[mIdx + 50 * gap]) + 4./1. * ( + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 33 * gap] + src[mIdx + 34 * gap]) + 1./16. * ( + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] + src[mIdx + 53 * gap] + src[mIdx + 54 * gap]);
        dst[rIdx + 2 * outW + 3] = + 32./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] + src[mIdx + 35 * gap] - src[mIdx + 36 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 17 * gap] - src[mIdx + 18 * gap]) + 2./1. * ( + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] + src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 4./1. * ( + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 33 * gap] - src[mIdx + 34 * gap]) + 1./2. * ( + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] + src[mIdx + 37 * gap] - src[mIdx + 38 * gap]) + 8./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] + src[mIdx + 19 * gap] - src[mIdx + 20 * gap]) + 1./8. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 21 * gap] - src[mIdx + 22 * gap]) + 1./32. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] + src[mIdx + 53 * gap] - src[mIdx + 54 * gap]) + 1./4. * ( + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] + src[mIdx + 49 * gap] - src[mIdx + 50 * gap]);
        dst[rIdx + 2 * outW + 4] = + 64./1. * ( + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] + src[mIdx + 35 * gap] + src[mIdx + 36 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 17 * gap] + src[mIdx + 18 * gap]) + 1./4. * ( + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] + src[mIdx + 37 * gap] + src[mIdx + 38 * gap] + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] + src[mIdx + 49 * gap] + src[mIdx + 50 * gap]) + 1./64. * ( + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] + src[mIdx + 53 * gap] + src[mIdx + 54 * gap]) + 16./1. * ( + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap]) + 1./16. * ( + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap]) + 4./1. * ( + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 33 * gap] + src[mIdx + 34 * gap] + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] + src[mIdx + 51 * gap] + src[mIdx + 52 * gap]);
        dst[rIdx + 2 * outW + 5] = + 32./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] + src[mIdx + 19 * gap] - src[mIdx + 20 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 15 * gap] + src[mIdx + 17 * gap] - src[mIdx + 18 * gap] + src[mIdx + 23 * gap]) + 4./1. * ( + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 31 * gap] + src[mIdx + 33 * gap] - src[mIdx + 34 * gap] + src[mIdx + 39 * gap]) + 1./32. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 21 * gap] - src[mIdx + 22 * gap]) + 128./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] + src[mIdx + 35 * gap] - src[mIdx + 36 * gap]) + 1./8. * ( + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] + src[mIdx + 37 * gap] - src[mIdx + 38 * gap]) + 8./1. * ( + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] + src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 1./128. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] + src[mIdx + 53 * gap] - src[mIdx + 54 * gap]) + 1./4. * ( + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] + src[mIdx + 47 * gap] + src[mIdx + 49 * gap] - src[mIdx + 50 * gap] + src[mIdx + 55 * gap]);
        dst[rIdx + 3 * outW + 0] = + 8./1. * ( + src[mIdx + 24 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] - src[mIdx + 32 * gap] - src[mIdx + 33 * gap] - src[mIdx + 34 * gap] - src[mIdx + 35 * gap] - src[mIdx + 36 * gap] - src[mIdx + 37 * gap] - src[mIdx + 38 * gap]) + 1./1. * ( + src[mIdx + 8 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] - src[mIdx + 16 * gap] - src[mIdx + 17 * gap] - src[mIdx + 18 * gap] - src[mIdx + 19 * gap] - src[mIdx + 20 * gap] - src[mIdx + 21 * gap] - src[mIdx + 22 * gap]) + 1./8. * ( + src[mIdx + 40 * gap] + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] - src[mIdx + 48 * gap] - src[mIdx + 49 * gap] - src[mIdx + 50 * gap] - src[mIdx + 51 * gap] - src[mIdx + 52 * gap] - src[mIdx + 53 * gap] - src[mIdx + 54 * gap]);
        dst[rIdx + 3 * outW + 1] = + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] - src[mIdx + 17 * gap] + src[mIdx + 18 * gap]) + 2./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] - src[mIdx + 19 * gap] + src[mIdx + 20 * gap]) + 4./1. * ( + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] - src[mIdx + 37 * gap] + src[mIdx + 38 * gap]) + 1./2. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] - src[mIdx + 21 * gap] + src[mIdx + 22 * gap]) + 8./1. * ( + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] - src[mIdx + 33 * gap] + src[mIdx + 34 * gap]) + 16./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] - src[mIdx + 35 * gap] + src[mIdx + 36 * gap]) + 1./16. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] - src[mIdx + 53 * gap] + src[mIdx + 54 * gap]) + 1./4. * ( + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] - src[mIdx + 51 * gap] + src[mIdx + 52 * gap]) + 1./8. * ( + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] - src[mIdx + 49 * gap] + src[mIdx + 50 * gap]);
        dst[rIdx + 3 * outW + 2] = + 32./1. * ( + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] - src[mIdx + 35 * gap] - src[mIdx + 36 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] - src[mIdx + 17 * gap] - src[mIdx + 18 * gap]) + 2./1. * ( + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] - src[mIdx + 37 * gap] - src[mIdx + 38 * gap]) + 1./4. * ( + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] - src[mIdx + 21 * gap] - src[mIdx + 22 * gap]) + 1./2. * ( + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] - src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 8./1. * ( + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] - src[mIdx + 33 * gap] - src[mIdx + 34 * gap]) + 1./8. * ( + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] - src[mIdx + 49 * gap] - src[mIdx + 50 * gap]) + 1./32. * ( + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] - src[mIdx + 53 * gap] - src[mIdx + 54 * gap]) + 4./1. * ( + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] - src[mIdx + 19 * gap] - src[mIdx + 20 * gap]);
        dst[rIdx + 3 * outW + 3] = + 8./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] - src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] - src[mIdx + 33 * gap] + src[mIdx + 34 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] - src[mIdx + 17 * gap] + src[mIdx + 18 * gap] + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] - src[mIdx + 37 * gap] + src[mIdx + 38 * gap] + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] - src[mIdx + 51 * gap] + src[mIdx + 52 * gap]) + 1./64. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] - src[mIdx + 53 * gap] + src[mIdx + 54 * gap]) + 64./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] - src[mIdx + 35 * gap] + src[mIdx + 36 * gap]) + 1./8. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] - src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] - src[mIdx + 49 * gap] + src[mIdx + 50 * gap]);
        dst[rIdx + 3 * outW + 4] = + 128./1. * ( + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] - src[mIdx + 35 * gap] - src[mIdx + 36 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] - src[mIdx + 17 * gap] - src[mIdx + 18 * gap]) + 2./1. * ( + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] - src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 1./2. * ( + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] - src[mIdx + 37 * gap] - src[mIdx + 38 * gap]) + 8./1. * ( + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] - src[mIdx + 33 * gap] - src[mIdx + 34 * gap]) + 16./1. * ( + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] - src[mIdx + 19 * gap] - src[mIdx + 20 * gap]) + 1./128. * ( + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] - src[mIdx + 53 * gap] - src[mIdx + 54 * gap]) + 1./16. * ( + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] - src[mIdx + 21 * gap] - src[mIdx + 22 * gap]) + 1./8. * ( + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] - src[mIdx + 49 * gap] - src[mIdx + 50 * gap]);
        dst[rIdx + 3 * outW + 5] = + 32./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] - src[mIdx + 19 * gap] + src[mIdx + 20 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 15 * gap] - src[mIdx + 17 * gap] + src[mIdx + 18 * gap] - src[mIdx + 23 * gap]) + 1./4. * ( + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] - src[mIdx + 37 * gap] + src[mIdx + 38 * gap]) + 1./32. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] - src[mIdx + 21 * gap] + src[mIdx + 22 * gap]) + 256./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] - src[mIdx + 35 * gap] + src[mIdx + 36 * gap]) + 1./8. * ( + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] + src[mIdx + 47 * gap] - src[mIdx + 49 * gap] + src[mIdx + 50 * gap] - src[mIdx + 55 * gap]) + 8./1. * ( + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 31 * gap] - src[mIdx + 33 * gap] + src[mIdx + 34 * gap] - src[mIdx + 39 * gap]) + 4./1. * ( + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] - src[mIdx + 51 * gap] + src[mIdx + 52 * gap]) + 1./256. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] - src[mIdx + 53 * gap] + src[mIdx + 54 * gap]);
        dst[rIdx + 4 * outW + 0] = + 16./1. * ( + src[mIdx + 24 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] + src[mIdx + 32 * gap] + src[mIdx + 33 * gap] + src[mIdx + 34 * gap] + src[mIdx + 35 * gap] + src[mIdx + 36 * gap] + src[mIdx + 37 * gap] + src[mIdx + 38 * gap]) + 1./1. * ( + src[mIdx + 8 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] + src[mIdx + 16 * gap] + src[mIdx + 17 * gap] + src[mIdx + 18 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap]) + 1./16. * ( + src[mIdx + 40 * gap] + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] + src[mIdx + 48 * gap] + src[mIdx + 49 * gap] + src[mIdx + 50 * gap] + src[mIdx + 51 * gap] + src[mIdx + 52 * gap] + src[mIdx + 53 * gap] + src[mIdx + 54 * gap]);
        dst[rIdx + 4 * outW + 1] = + 32./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] + src[mIdx + 35 * gap] - src[mIdx + 36 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 17 * gap] - src[mIdx + 18 * gap]) + 2./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] + src[mIdx + 19 * gap] - src[mIdx + 20 * gap]) + 1./2. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 21 * gap] - src[mIdx + 22 * gap]) + 8./1. * ( + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] + src[mIdx + 37 * gap] - src[mIdx + 38 * gap]) + 16./1. * ( + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 33 * gap] - src[mIdx + 34 * gap]) + 1./32. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] + src[mIdx + 53 * gap] - src[mIdx + 54 * gap]) + 1./16. * ( + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] + src[mIdx + 49 * gap] - src[mIdx + 50 * gap]) + 1./8. * ( + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] + src[mIdx + 51 * gap] - src[mIdx + 52 * gap]);
        dst[rIdx + 4 * outW + 2] = + 64./1. * ( + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] + src[mIdx + 35 * gap] + src[mIdx + 36 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 17 * gap] + src[mIdx + 18 * gap]) + 1./4. * ( + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] + src[mIdx + 51 * gap] + src[mIdx + 52 * gap]) + 1./64. * ( + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] + src[mIdx + 53 * gap] + src[mIdx + 54 * gap]) + 16./1. * ( + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 33 * gap] + src[mIdx + 34 * gap]) + 1./16. * ( + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] + src[mIdx + 49 * gap] + src[mIdx + 50 * gap]) + 4./1. * ( + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] + src[mIdx + 37 * gap] + src[mIdx + 38 * gap]);
        dst[rIdx + 4 * outW + 3] = + 128./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] + src[mIdx + 35 * gap] - src[mIdx + 36 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 17 * gap] - src[mIdx + 18 * gap]) + 2./1. * ( + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] + src[mIdx + 37 * gap] - src[mIdx + 38 * gap]) + 1./2. * ( + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] + src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 8./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] + src[mIdx + 19 * gap] - src[mIdx + 20 * gap]) + 16./1. * ( + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 33 * gap] - src[mIdx + 34 * gap]) + 1./128. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] + src[mIdx + 53 * gap] - src[mIdx + 54 * gap]) + 1./16. * ( + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] + src[mIdx + 49 * gap] - src[mIdx + 50 * gap]) + 1./8. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 21 * gap] - src[mIdx + 22 * gap]);
        dst[rIdx + 4 * outW + 4] = + 16./1. * ( + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] + src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 33 * gap] + src[mIdx + 34 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 17 * gap] + src[mIdx + 18 * gap] + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] + src[mIdx + 37 * gap] + src[mIdx + 38 * gap] + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] + src[mIdx + 51 * gap] + src[mIdx + 52 * gap]) + 1./256. * ( + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] + src[mIdx + 53 * gap] + src[mIdx + 54 * gap]) + 1./16. * ( + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] + src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] + src[mIdx + 49 * gap] + src[mIdx + 50 * gap]) + 256./1. * ( + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] + src[mIdx + 35 * gap] + src[mIdx + 36 * gap]);
        dst[rIdx + 4 * outW + 5] = + 32./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] + src[mIdx + 19 * gap] - src[mIdx + 20 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 15 * gap] + src[mIdx + 17 * gap] - src[mIdx + 18 * gap] + src[mIdx + 23 * gap]) + 2./1. * ( + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] + src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 1./2. * ( + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] + src[mIdx + 37 * gap] - src[mIdx + 38 * gap]) + 1./32. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + src[mIdx + 21 * gap] - src[mIdx + 22 * gap]) + 16./1. * ( + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 31 * gap] + src[mIdx + 33 * gap] - src[mIdx + 34 * gap] + src[mIdx + 39 * gap]) + 1./512. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] + src[mIdx + 53 * gap] - src[mIdx + 54 * gap]) + 512./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] + src[mIdx + 35 * gap] - src[mIdx + 36 * gap]) + 1./16. * ( + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] + src[mIdx + 47 * gap] + src[mIdx + 49 * gap] - src[mIdx + 50 * gap] + src[mIdx + 55 * gap]);
        dst[rIdx + 5 * outW + 0] = + 32./1. * ( + src[mIdx + 24 * gap] + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] - src[mIdx + 32 * gap] - src[mIdx + 33 * gap] - src[mIdx + 34 * gap] - src[mIdx + 35 * gap] - src[mIdx + 36 * gap] - src[mIdx + 37 * gap] - src[mIdx + 38 * gap]) + 1./1. * ( + src[mIdx + 8 * gap] + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] - src[mIdx + 16 * gap] - src[mIdx + 17 * gap] - src[mIdx + 18 * gap] - src[mIdx + 19 * gap] - src[mIdx + 20 * gap] - src[mIdx + 21 * gap] - src[mIdx + 22 * gap] + src[mIdx + 56 * gap] + src[mIdx + 57 * gap] + src[mIdx + 58 * gap] + src[mIdx + 59 * gap] + src[mIdx + 60 * gap] + src[mIdx + 61 * gap] + src[mIdx + 62 * gap]) + 1./32. * ( + src[mIdx + 40 * gap] + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] - src[mIdx + 48 * gap] - src[mIdx + 49 * gap] - src[mIdx + 50 * gap] - src[mIdx + 51 * gap] - src[mIdx + 52 * gap] - src[mIdx + 53 * gap] - src[mIdx + 54 * gap]);
        dst[rIdx + 5 * outW + 1] = + 32./1. * ( + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] - src[mIdx + 33 * gap] + src[mIdx + 34 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] - src[mIdx + 17 * gap] + src[mIdx + 18 * gap] + src[mIdx + 57 * gap] - src[mIdx + 58 * gap]) + 2./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] - src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 59 * gap] - src[mIdx + 60 * gap]) + 1./64. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] - src[mIdx + 53 * gap] + src[mIdx + 54 * gap]) + 1./2. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] - src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 61 * gap] - src[mIdx + 62 * gap]) + 64./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] - src[mIdx + 35 * gap] + src[mIdx + 36 * gap]) + 16./1. * ( + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] - src[mIdx + 37 * gap] + src[mIdx + 38 * gap]) + 1./32. * ( + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] - src[mIdx + 49 * gap] + src[mIdx + 50 * gap]) + 1./16. * ( + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] - src[mIdx + 51 * gap] + src[mIdx + 52 * gap]);
        dst[rIdx + 5 * outW + 2] = + 32./1. * ( + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] - src[mIdx + 33 * gap] - src[mIdx + 34 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] - src[mIdx + 17 * gap] - src[mIdx + 18 * gap] + src[mIdx + 57 * gap] + src[mIdx + 58 * gap]) + 1./4. * ( + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] - src[mIdx + 21 * gap] - src[mIdx + 22 * gap] + src[mIdx + 61 * gap] + src[mIdx + 62 * gap]) + 8./1. * ( + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] - src[mIdx + 37 * gap] - src[mIdx + 38 * gap]) + 128./1. * ( + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] - src[mIdx + 35 * gap] - src[mIdx + 36 * gap]) + 1./8. * ( + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] - src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 1./32. * ( + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] - src[mIdx + 49 * gap] - src[mIdx + 50 * gap]) + 1./128. * ( + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] - src[mIdx + 53 * gap] - src[mIdx + 54 * gap]) + 4./1. * ( + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] - src[mIdx + 19 * gap] - src[mIdx + 20 * gap] + src[mIdx + 59 * gap] + src[mIdx + 60 * gap]);
        dst[rIdx + 5 * outW + 3] = + 32./1. * ( + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] - src[mIdx + 33 * gap] + src[mIdx + 34 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] - src[mIdx + 17 * gap] + src[mIdx + 18 * gap] + src[mIdx + 57 * gap] - src[mIdx + 58 * gap]) + 4./1. * ( + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] - src[mIdx + 37 * gap] + src[mIdx + 38 * gap]) + 8./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] - src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 59 * gap] - src[mIdx + 60 * gap]) + 256./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] - src[mIdx + 35 * gap] + src[mIdx + 36 * gap]) + 1./8. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] - src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 61 * gap] - src[mIdx + 62 * gap]) + 1./32. * ( + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] - src[mIdx + 49 * gap] + src[mIdx + 50 * gap]) + 1./4. * ( + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] - src[mIdx + 51 * gap] + src[mIdx + 52 * gap]) + 1./256. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] - src[mIdx + 53 * gap] + src[mIdx + 54 * gap]);
        dst[rIdx + 5 * outW + 4] = + 512./1. * ( + src[mIdx + 27 * gap] + src[mIdx + 28 * gap] - src[mIdx + 35 * gap] - src[mIdx + 36 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] + src[mIdx + 10 * gap] - src[mIdx + 17 * gap] - src[mIdx + 18 * gap] + src[mIdx + 57 * gap] + src[mIdx + 58 * gap]) + 2./1. * ( + src[mIdx + 29 * gap] + src[mIdx + 30 * gap] - src[mIdx + 37 * gap] - src[mIdx + 38 * gap]) + 32./1. * ( + src[mIdx + 25 * gap] + src[mIdx + 26 * gap] - src[mIdx + 33 * gap] - src[mIdx + 34 * gap]) + 1./32. * ( + src[mIdx + 41 * gap] + src[mIdx + 42 * gap] - src[mIdx + 49 * gap] - src[mIdx + 50 * gap]) + 1./2. * ( + src[mIdx + 43 * gap] + src[mIdx + 44 * gap] - src[mIdx + 51 * gap] - src[mIdx + 52 * gap]) + 16./1. * ( + src[mIdx + 11 * gap] + src[mIdx + 12 * gap] - src[mIdx + 19 * gap] - src[mIdx + 20 * gap] + src[mIdx + 59 * gap] + src[mIdx + 60 * gap]) + 1./512. * ( + src[mIdx + 45 * gap] + src[mIdx + 46 * gap] - src[mIdx + 53 * gap] - src[mIdx + 54 * gap]) + 1./16. * ( + src[mIdx + 13 * gap] + src[mIdx + 14 * gap] - src[mIdx + 21 * gap] - src[mIdx + 22 * gap] + src[mIdx + 61 * gap] + src[mIdx + 62 * gap]);
        dst[rIdx + 5 * outW + 5] = + 32./1. * ( + src[mIdx + 11 * gap] - src[mIdx + 12 * gap] - src[mIdx + 19 * gap] + src[mIdx + 20 * gap] + src[mIdx + 25 * gap] - src[mIdx + 26 * gap] + src[mIdx + 31 * gap] - src[mIdx + 33 * gap] + src[mIdx + 34 * gap] - src[mIdx + 39 * gap] + src[mIdx + 59 * gap] - src[mIdx + 60 * gap]) + 1./1. * ( + src[mIdx + 9 * gap] - src[mIdx + 10 * gap] + src[mIdx + 15 * gap] - src[mIdx + 17 * gap] + src[mIdx + 18 * gap] - src[mIdx + 23 * gap] + src[mIdx + 29 * gap] - src[mIdx + 30 * gap] - src[mIdx + 37 * gap] + src[mIdx + 38 * gap] + src[mIdx + 43 * gap] - src[mIdx + 44 * gap] - src[mIdx + 51 * gap] + src[mIdx + 52 * gap] + src[mIdx + 57 * gap] - src[mIdx + 58 * gap] + src[mIdx + 63 * gap]) + 1./32. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] - src[mIdx + 21 * gap] + src[mIdx + 22 * gap] + src[mIdx + 41 * gap] - src[mIdx + 42 * gap] + src[mIdx + 47 * gap] - src[mIdx + 49 * gap] + src[mIdx + 50 * gap] - src[mIdx + 55 * gap] + src[mIdx + 61 * gap] - src[mIdx + 62 * gap]) + 1./1024. * ( + src[mIdx + 45 * gap] - src[mIdx + 46 * gap] - src[mIdx + 53 * gap] + src[mIdx + 54 * gap]) + 1024./1. * ( + src[mIdx + 27 * gap] - src[mIdx + 28 * gap] - src[mIdx + 35 * gap] + src[mIdx + 36 * gap]);
	}
}



template <typename Dtype> 
__global__ void winoDstAddOpt_gpu_kernel(const Dtype *src, Dtype * dst, const int tileH, const int tileW, const int outH, const int outW, const int outputs, const int batchs, const int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int highIdx = idx / (tileW * tileH);
		int yIdx = (idx % (tileW * tileH)) / tileW;
		int xIdx = idx % tileW;
		int rIdx = highIdx * outW * outH + yIdx * outW * 2 + xIdx * 2;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;

		Dtype t[3];
		t[2] = + src[mIdx + 1 * gap] + src[mIdx + 9 * gap];
		t[0] = + src[mIdx + 4 * gap] + src[mIdx + 5 * gap];
		dst[rIdx + 0 * outW + 0] = + 1./1. * ( + src[mIdx + 0 * gap] + src[mIdx + 2 * gap] + src[mIdx + 6 * gap] + src[mIdx + 8 * gap] + src[mIdx + 10 * gap] + t[0] + t[2]);
		t[1] = + src[mIdx + 5 * gap] - src[mIdx + 6 * gap];
		dst[rIdx + 0 * outW + 1] = + 1./1. * ( - src[mIdx + 2 * gap] + src[mIdx + 3 * gap] + src[mIdx + 7 * gap] - src[mIdx + 10 * gap] + src[mIdx + 11 * gap] + t[1] + t[2]);
		t[2] = - src[mIdx + 9 * gap] + src[mIdx + 13 * gap];
		dst[rIdx + 1 * outW + 0] = + 1./1. * ( + src[mIdx + 6 * gap] - src[mIdx + 8 * gap] - src[mIdx + 10 * gap] + src[mIdx + 12 * gap] + src[mIdx + 14 * gap] + t[0] + t[2]);
		dst[rIdx + 1 * outW + 1] = + 1./1. * ( + src[mIdx + 7 * gap] + src[mIdx + 10 * gap] - src[mIdx + 11 * gap] - src[mIdx + 14 * gap] + src[mIdx + 15 * gap] + t[1] + t[2]);
	}
}


template <typename Dtype> 
__global__ void wino4x4DstAddOpt_gpu_kernel(const Dtype *src, Dtype * dst, const int tileH, const int tileW, const int outH, const int outW, const int outputs, const int batchs, const int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int highIdx = idx / (tileW * tileH);
		int yIdx = (idx % (tileW * tileH)) / tileW;
		int xIdx = idx % tileW;
		int rIdx = highIdx * outW * outH + yIdx * outW * 4 + xIdx * 4;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;

		Dtype t[26];
		t[20] = + src[mIdx + 24 * gap] + src[mIdx + 25 * gap];
		t[3] = + src[mIdx + 7 * gap] + src[mIdx + 8 * gap];
		t[8] = + src[mIdx + 1 * gap] + src[mIdx + 13 * gap];
		t[14] = + src[mIdx + 4 * gap] + src[mIdx + 22 * gap];
		t[5] = + src[mIdx + 19 * gap] + src[mIdx + 26 * gap];
		t[12] = + src[mIdx + 2 * gap] + src[mIdx + 14 * gap];
		t[23] = + src[mIdx + 6 * gap] + src[mIdx + 12 * gap];
		t[6] = + src[mIdx + 21 * gap] + src[mIdx + 28 * gap];
		t[9] = + src[mIdx + 15 * gap] + src[mIdx + 16 * gap];
		t[10] = + src[mIdx + 3 * gap] + src[mIdx + 27 * gap];
		t[11] = + src[mIdx + 18 * gap] + src[mIdx + 20 * gap];
		t[1] = + src[mIdx + 9 * gap] + src[mIdx + 10 * gap];
		dst[rIdx + 0 * outW + 0] = + 1./1. * ( + src[mIdx + 0 * gap] + t[1] + t[3] + t[5] + t[6] + t[9] + t[8] + t[11] + t[10] + t[20] + t[12] + t[23] + t[14]);
		t[4] = + src[mIdx + 21 * gap] - src[mIdx + 28 * gap];
		t[18] = - src[mIdx + 20 * gap] + src[mIdx + 25 * gap];
		t[16] = - src[mIdx + 4 * gap] - src[mIdx + 22 * gap];
		t[7] = + src[mIdx + 9 * gap] - src[mIdx + 10 * gap];
		t[0] = + src[mIdx + 19 * gap] - src[mIdx + 26 * gap];
		t[2] = + src[mIdx + 7 * gap] - src[mIdx + 8 * gap];
		t[13] = + src[mIdx + 15 * gap] - src[mIdx + 16 * gap];
		dst[rIdx + 0 * outW + 1] = + 1./1. * ( - src[mIdx + 2 * gap] - src[mIdx + 14 * gap] + t[0] + t[2] + t[8] + t[18]) + 2./1. * ( + t[4] + t[7] + t[10] + t[13] + t[16]);
		t[21] = + src[mIdx + 20 * gap] + src[mIdx + 25 * gap];
		dst[rIdx + 0 * outW + 2] = + 1./1. * ( + t[3] + t[5] + t[8] + t[12] + t[21]) + 4./1. * ( + t[1] + t[6] + t[9] + t[10] + t[14]);
		t[14] = - src[mIdx + 14 * gap] + src[mIdx + 17 * gap];
		t[15] = + src[mIdx + 23 * gap] + src[mIdx + 29 * gap];
		dst[rIdx + 0 * outW + 3] = + 8./1. * ( + t[4] + t[7] + t[10] + t[13] + t[16]) + 1./1. * ( - src[mIdx + 2 * gap] + src[mIdx + 5 * gap] + src[mIdx + 11 * gap] + t[0] + t[2] + t[8] + t[18] + t[14] + t[15]);
		t[17] = - src[mIdx + 22 * gap] - src[mIdx + 27 * gap];
		t[16] = - src[mIdx + 20 * gap] - src[mIdx + 25 * gap];
		t[8] = - src[mIdx + 13 * gap] + src[mIdx + 14 * gap];
		t[12] = - src[mIdx + 15 * gap] + src[mIdx + 16 * gap];
		t[19] = + src[mIdx + 11 * gap] - src[mIdx + 17 * gap];
		t[24] = + src[mIdx + 23 * gap] - src[mIdx + 29 * gap];
		dst[rIdx + 1 * outW + 3] = + 8./1. * ( + t[7] + t[12]) + 1./1. * ( + t[2] + t[8] + t[19]) + 2./1. * ( + t[5] + t[16] + t[24]) + 16./1. * ( + t[6] + t[17]);
		t[25] = - src[mIdx + 22 * gap] + src[mIdx + 27 * gap];
		dst[rIdx + 2 * outW + 3] = + 8./1. * ( + t[7] + t[13]) + 1./1. * ( + src[mIdx + 11 * gap] + src[mIdx + 13 * gap] + t[2] + t[14]) + 4./1. * ( + t[0] + t[18] + t[15]) + 32./1. * ( + t[4] + t[25]);
		dst[rIdx + 1 * outW + 1] = + 1./1. * ( + t[2] + t[8]) + 2./1. * ( + t[5] + t[7] + t[12] + t[16]) + 4./1. * ( + t[6] + t[17]);
		t[22] = + src[mIdx + 20 * gap] - src[mIdx + 25 * gap];
		t[15] = + src[mIdx + 22 * gap] - src[mIdx + 27 * gap];
		t[14] = - src[mIdx + 13 * gap] - src[mIdx + 14 * gap];
		t[10] = - src[mIdx + 15 * gap] - src[mIdx + 16 * gap];
		dst[rIdx + 1 * outW + 2] = + 8./1. * ( + t[4] + t[15]) + 1./1. * ( + t[3] + t[14]) + 2./1. * ( + t[0] + t[22]) + 4./1. * ( + t[1] + t[10]);
		dst[rIdx + 2 * outW + 1] = + 8./1. * ( + t[4] + t[25]) + 1./1. * ( + src[mIdx + 13 * gap] - src[mIdx + 14 * gap] + t[2]) + 2./1. * ( + t[7] + t[13]) + 4./1. * ( + t[0] + t[18]);
		t[13] = + src[mIdx + 22 * gap] + src[mIdx + 27 * gap];
		t[25] = + src[mIdx + 13 * gap] + src[mIdx + 14 * gap];
		dst[rIdx + 2 * outW + 2] = + 16./1. * ( + t[6] + t[13]) + 1./1. * ( + t[3] + t[25]) + 4./1. * ( + t[1] + t[5] + t[9] + t[21]);
		t[18] = - src[mIdx + 24 * gap] - src[mIdx + 25 * gap];
		t[21] = + src[mIdx + 6 * gap] - src[mIdx + 12 * gap];
		dst[rIdx + 1 * outW + 0] = + 1./1. * ( + t[1] + t[3] + t[10] + t[14] + t[21]) + 2./1. * ( + t[0] + t[4] + t[11] + t[15] + t[18]);
		dst[rIdx + 2 * outW + 0] = + 1./1. * ( + t[1] + t[3] + t[9] + t[23] + t[25]) + 4./1. * ( + t[5] + t[6] + t[11] + t[20] + t[13]);
		t[13] = + src[mIdx + 33 * gap] + src[mIdx + 34 * gap];
		t[20] = + src[mIdx + 31 * gap] + src[mIdx + 32 * gap];
		dst[rIdx + 3 * outW + 0] = + 8./1. * ( + t[0] + t[4] + t[11] + t[15] + t[18]) + 1./1. * ( + src[mIdx + 30 * gap] + t[1] + t[3] + t[10] + t[14] + t[13] + t[21] + t[20]);
		t[11] = + src[mIdx + 33 * gap] - src[mIdx + 34 * gap];
		t[9] = + src[mIdx + 31 * gap] - src[mIdx + 32 * gap];
		dst[rIdx + 3 * outW + 1] = + 8./1. * ( + t[5] + t[16]) + 1./1. * ( + t[2] + t[8] + t[9]) + 2./1. * ( + t[7] + t[12] + t[11]) + 16./1. * ( + t[6] + t[17]);
		dst[rIdx + 3 * outW + 2] = + 8./1. * ( + t[0] + t[22]) + 1./1. * ( + t[3] + t[14] + t[20]) + 4./1. * ( + t[1] + t[10] + t[13]) + 32./1. * ( + t[4] + t[15]);
		dst[rIdx + 3 * outW + 3] = + 8./1. * ( + t[5] + t[7] + t[12] + t[16] + t[24] + t[11]) + 1./1. * ( + src[mIdx + 35 * gap] + t[2] + t[8] + t[19] + t[9]) + 64./1. * ( + t[6] + t[17]);
	}
}

template <typename Dtype> 
__global__ void wino6x6DstAddOpt_gpu_kernel(const Dtype *src, Dtype * dst, const int tileH, const int tileW, const int outH, const int outW, const int outputs, const int batchs, const int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int highIdx = idx / (tileW * tileH);
		int yIdx = (idx % (tileW * tileH)) / tileW;
		int xIdx = idx % tileW;
		int rIdx = highIdx * outW * outH + yIdx * outW * 6 + xIdx * 6;
		int mIdx = (idx % tNums); 
		int gap = batchs * outputs * tileH * tileW;

		Dtype t[60];
		t[24] = + src[mIdx + 1 * gap] + src[mIdx + 2 * gap];
		t[41] = + src[mIdx + 51 * gap] + src[mIdx + 52 * gap];
		t[8] = + src[mIdx + 41 * gap] + src[mIdx + 50 * gap];
		t[58] = + src[mIdx + 24 * gap] + src[mIdx + 32 * gap];
		t[50] = + src[mIdx + 46 * gap] + src[mIdx + 53 * gap];
		t[11] = + src[mIdx + 29 * gap] + src[mIdx + 30 * gap];
		t[57] = + src[mIdx + 40 * gap] + src[mIdx + 48 * gap];
		t[30] = + src[mIdx + 12 * gap] + src[mIdx + 20 * gap];
		t[13] = + src[mIdx + 27 * gap] + src[mIdx + 35 * gap];
		t[53] = + src[mIdx + 21 * gap] + src[mIdx + 22 * gap];
		t[5] = + src[mIdx + 5 * gap] + src[mIdx + 6 * gap];
		t[1] = + src[mIdx + 43 * gap] + src[mIdx + 44 * gap];
		t[21] = + src[mIdx + 10 * gap] + src[mIdx + 17 * gap];
		t[46] = + src[mIdx + 28 * gap] + src[mIdx + 36 * gap];
		t[55] = + src[mIdx + 8 * gap] + src[mIdx + 16 * gap];
		t[10] = + src[mIdx + 45 * gap] + src[mIdx + 54 * gap];
		t[27] = + src[mIdx + 3 * gap] + src[mIdx + 4 * gap];
		t[17] = + src[mIdx + 11 * gap] + src[mIdx + 19 * gap];
		t[47] = + src[mIdx + 42 * gap] + src[mIdx + 49 * gap];
		t[15] = + src[mIdx + 9 * gap] + src[mIdx + 18 * gap];
		t[16] = + src[mIdx + 13 * gap] + src[mIdx + 14 * gap];
		t[36] = + src[mIdx + 37 * gap] + src[mIdx + 38 * gap];
		t[26] = + src[mIdx + 33 * gap] + src[mIdx + 34 * gap];
		t[0] = + src[mIdx + 25 * gap] + src[mIdx + 26 * gap];
		dst[rIdx + 0 * outW + 0] = + 1./1. * ( + src[mIdx + 0 * gap] + t[0] + t[1] + t[8] + t[10] + t[11] + t[13] + t[15] + t[16] + t[17] + t[21] + t[26] + t[30] + t[36] + t[41] + t[46] + t[47] + t[50] + t[53] + t[55] + t[5] + t[57] + t[58] + t[24] + t[27]);
		t[20] = + src[mIdx + 5 * gap] - src[mIdx + 6 * gap];
		t[23] = + src[mIdx + 1 * gap] - src[mIdx + 2 * gap];
		t[7] = + src[mIdx + 9 * gap] - src[mIdx + 18 * gap];
		t[4] = + src[mIdx + 45 * gap] - src[mIdx + 54 * gap];
		t[3] = + src[mIdx + 41 * gap] - src[mIdx + 50 * gap];
		t[6] = + src[mIdx + 25 * gap] - src[mIdx + 26 * gap];
		t[52] = + src[mIdx + 37 * gap] - src[mIdx + 38 * gap];
		t[25] = - src[mIdx + 10 * gap] + src[mIdx + 17 * gap];
		t[40] = - src[mIdx + 28 * gap] - src[mIdx + 36 * gap];
		t[12] = + src[mIdx + 43 * gap] - src[mIdx + 44 * gap];
		t[9] = + src[mIdx + 29 * gap] - src[mIdx + 30 * gap];
		t[2] = + src[mIdx + 3 * gap] - src[mIdx + 4 * gap];
		t[22] = + src[mIdx + 51 * gap] - src[mIdx + 52 * gap];
		t[33] = - src[mIdx + 42 * gap] + src[mIdx + 49 * gap];
		t[18] = + src[mIdx + 21 * gap] - src[mIdx + 22 * gap];
		t[43] = + src[mIdx + 33 * gap] - src[mIdx + 34 * gap];
		t[42] = - src[mIdx + 12 * gap] - src[mIdx + 20 * gap];
		t[14] = + src[mIdx + 13 * gap] - src[mIdx + 14 * gap];
		t[19] = - src[mIdx + 46 * gap] + src[mIdx + 53 * gap];
		dst[rIdx + 0 * outW + 1] = + 1./1. * ( + t[3] + t[6] + t[7] + t[25] + t[33] + t[43] + t[23]) + 2./1. * ( + t[12] + t[13] + t[17] + t[22] + t[40] + t[42] + t[2]) + 1./2. * ( + t[4] + t[9] + t[14] + t[18] + t[19] + t[52] + t[20]);
		dst[rIdx + 0 * outW + 2] = + 1./1. * ( + t[0] + t[8] + t[15] + t[21] + t[26] + t[47] + t[24]) + 1./4. * ( + t[10] + t[11] + t[16] + t[36] + t[50] + t[53] + t[5]) + 4./1. * ( + t[1] + t[13] + t[17] + t[30] + t[41] + t[46] + t[27]);
		dst[rIdx + 0 * outW + 3] = + 8./1. * ( + t[12] + t[13] + t[17] + t[22] + t[40] + t[42] + t[2]) + 1./1. * ( + t[3] + t[6] + t[7] + t[25] + t[33] + t[43] + t[23]) + 1./8. * ( + t[4] + t[9] + t[14] + t[18] + t[19] + t[52] + t[20]);
		dst[rIdx + 0 * outW + 4] = + 16./1. * ( + t[1] + t[13] + t[17] + t[30] + t[41] + t[46] + t[27]) + 1./1. * ( + t[0] + t[8] + t[15] + t[21] + t[26] + t[47] + t[24]) + 1./16. * ( + t[10] + t[11] + t[16] + t[36] + t[50] + t[53] + t[5]);
		t[27] = + src[mIdx + 31 * gap] + src[mIdx + 39 * gap];
		t[24] = + src[mIdx + 15 * gap] + src[mIdx + 23 * gap];
		t[29] = + src[mIdx + 47 * gap] + src[mIdx + 55 * gap];
		dst[rIdx + 0 * outW + 5] = + 32./1. * ( + t[12] + t[13] + t[17] + t[22] + t[40] + t[42] + t[2]) + 1./1. * ( + src[mIdx + 7 * gap] + t[3] + t[6] + t[7] + t[25] + t[33] + t[43] + t[24] + t[27] + t[23] + t[29]) + 1./32. * ( + t[4] + t[9] + t[14] + t[18] + t[19] + t[52] + t[20]);
		t[28] = - src[mIdx + 28 * gap] + src[mIdx + 36 * gap];
		t[59] = + src[mIdx + 31 * gap] - src[mIdx + 39 * gap];
		t[23] = - src[mIdx + 21 * gap] + src[mIdx + 22 * gap];
		t[20] = - src[mIdx + 33 * gap] + src[mIdx + 34 * gap];
		t[5] = + src[mIdx + 27 * gap] - src[mIdx + 35 * gap];
		t[48] = - src[mIdx + 42 * gap] - src[mIdx + 49 * gap];
		t[35] = - src[mIdx + 10 * gap] - src[mIdx + 17 * gap];
		t[49] = - src[mIdx + 46 * gap] - src[mIdx + 53 * gap];
		t[56] = + src[mIdx + 15 * gap] - src[mIdx + 23 * gap];
		t[2] = + src[mIdx + 11 * gap] - src[mIdx + 19 * gap];
		t[34] = - src[mIdx + 37 * gap] + src[mIdx + 38 * gap];
		t[32] = - src[mIdx + 51 * gap] + src[mIdx + 52 * gap];
		t[54] = + src[mIdx + 47 * gap] - src[mIdx + 55 * gap];
		t[38] = - src[mIdx + 12 * gap] + src[mIdx + 20 * gap];
		dst[rIdx + 1 * outW + 5] = + 32./1. * ( + t[2] + t[38]) + 1./1. * ( + t[15] + t[35] + t[56]) + 2./1. * ( + t[6] + t[20] + t[59]) + 1./2. * ( + t[8] + t[48] + t[54]) + 1./32. * ( + t[14] + t[23]) + 64./1. * ( + t[5] + t[28]) + 16./1. * ( + t[12] + t[32]) + 1./64. * ( + t[10] + t[49]) + 1./16. * ( + t[9] + t[34]);
		dst[rIdx + 2 * outW + 5] = + 32./1. * ( + t[17] + t[42]) + 1./1. * ( + t[7] + t[25] + t[24]) + 4./1. * ( + t[6] + t[43] + t[27]) + 1./32. * ( + t[14] + t[18]) + 128./1. * ( + t[13] + t[40]) + 1./8. * ( + t[9] + t[52]) + 8./1. * ( + t[12] + t[22]) + 1./128. * ( + t[4] + t[19]) + 1./4. * ( + t[3] + t[33] + t[29]);
		dst[rIdx + 3 * outW + 5] = + 32./1. * ( + t[2] + t[38]) + 1./1. * ( + t[15] + t[35] + t[56]) + 1./4. * ( + t[9] + t[34]) + 1./32. * ( + t[14] + t[23]) + 256./1. * ( + t[5] + t[28]) + 1./8. * ( + t[8] + t[48] + t[54]) + 8./1. * ( + t[6] + t[20] + t[59]) + 4./1. * ( + t[12] + t[32]) + 1./256. * ( + t[10] + t[49]);
		dst[rIdx + 4 * outW + 5] = + 32./1. * ( + t[17] + t[42]) + 1./1. * ( + t[7] + t[25] + t[24]) + 2./1. * ( + t[12] + t[22]) + 1./2. * ( + t[9] + t[52]) + 1./32. * ( + t[14] + t[18]) + 16./1. * ( + t[6] + t[43] + t[27]) + 1./512. * ( + t[4] + t[19]) + 512./1. * ( + t[13] + t[40]) + 1./16. * ( + t[3] + t[33] + t[29]);
		dst[rIdx + 1 * outW + 1] = + 1./4. * ( + t[10] + t[49]) + 1./1. * ( + t[9] + t[12] + t[15] + t[32] + t[34] + t[35]) + 2./1. * ( + t[2] + t[6] + t[20] + t[38]) + 4./1. * ( + t[5] + t[28]) + 1./2. * ( + t[8] + t[14] + t[23] + t[48]);
		t[45] = - src[mIdx + 37 * gap] - src[mIdx + 38 * gap];
		t[39] = + src[mIdx + 12 * gap] - src[mIdx + 20 * gap];
		t[44] = - src[mIdx + 51 * gap] - src[mIdx + 52 * gap];
		t[27] = - src[mIdx + 21 * gap] - src[mIdx + 22 * gap];
		t[51] = + src[mIdx + 46 * gap] - src[mIdx + 53 * gap];
		t[31] = + src[mIdx + 42 * gap] - src[mIdx + 49 * gap];
		t[24] = + src[mIdx + 28 * gap] - src[mIdx + 36 * gap];
		t[37] = - src[mIdx + 33 * gap] - src[mIdx + 34 * gap];
		t[29] = + src[mIdx + 10 * gap] - src[mIdx + 17 * gap];
		dst[rIdx + 1 * outW + 2] = + 1./1. * ( + t[7] + t[29]) + 2./1. * ( + t[0] + t[1] + t[37] + t[44]) + 1./4. * ( + t[16] + t[27]) + 1./2. * ( + t[3] + t[11] + t[31] + t[45]) + 8./1. * ( + t[5] + t[24]) + 1./8. * ( + t[4] + t[51]) + 4./1. * ( + t[2] + t[39]);
		dst[rIdx + 1 * outW + 3] = + 1./1. * ( + t[15] + t[35]) + 2./1. * ( + t[6] + t[20]) + 1./4. * ( + t[9] + t[34]) + 1./2. * ( + t[8] + t[48]) + 8./1. * ( + t[2] + t[38]) + 16./1. * ( + t[5] + t[28]) + 1./16. * ( + t[10] + t[49]) + 4./1. * ( + t[12] + t[32]) + 1./8. * ( + t[14] + t[23]);
		dst[rIdx + 1 * outW + 4] = + 32./1. * ( + t[5] + t[24]) + 1./1. * ( + t[7] + t[29]) + 2./1. * ( + t[0] + t[37]) + 1./2. * ( + t[3] + t[31]) + 8./1. * ( + t[1] + t[44]) + 16./1. * ( + t[2] + t[39]) + 1./32. * ( + t[4] + t[51]) + 1./16. * ( + t[16] + t[27]) + 1./8. * ( + t[11] + t[45]);
		dst[rIdx + 2 * outW + 1] = + 1./1. * ( + t[7] + t[25]) + 2./1. * ( + t[9] + t[17] + t[42] + t[52]) + 4./1. * ( + t[6] + t[43]) + 1./2. * ( + t[12] + t[14] + t[18] + t[22]) + 8./1. * ( + t[13] + t[40]) + 1./8. * ( + t[4] + t[19]) + 1./4. * ( + t[3] + t[33]);
		dst[rIdx + 2 * outW + 2] = + 16./1. * ( + t[13] + t[46]) + 1./1. * ( + t[1] + t[11] + t[15] + t[21] + t[36] + t[41]) + 1./4. * ( + t[8] + t[16] + t[47] + t[53]) + 4./1. * ( + t[0] + t[17] + t[26] + t[30]) + 1./16. * ( + t[10] + t[50]);
		dst[rIdx + 2 * outW + 3] = + 32./1. * ( + t[13] + t[40]) + 1./1. * ( + t[7] + t[25]) + 2./1. * ( + t[12] + t[22]) + 4./1. * ( + t[6] + t[43]) + 1./2. * ( + t[9] + t[52]) + 8./1. * ( + t[17] + t[42]) + 1./8. * ( + t[14] + t[18]) + 1./32. * ( + t[4] + t[19]) + 1./4. * ( + t[3] + t[33]);
		dst[rIdx + 2 * outW + 4] = + 64./1. * ( + t[13] + t[46]) + 1./1. * ( + t[15] + t[21]) + 1./4. * ( + t[8] + t[11] + t[36] + t[47]) + 1./64. * ( + t[10] + t[50]) + 16./1. * ( + t[17] + t[30]) + 1./16. * ( + t[16] + t[53]) + 4./1. * ( + t[0] + t[1] + t[26] + t[41]);
		dst[rIdx + 3 * outW + 1] = + 1./1. * ( + t[15] + t[35]) + 2./1. * ( + t[2] + t[38]) + 4./1. * ( + t[9] + t[34]) + 1./2. * ( + t[14] + t[23]) + 8./1. * ( + t[6] + t[20]) + 16./1. * ( + t[5] + t[28]) + 1./16. * ( + t[10] + t[49]) + 1./4. * ( + t[12] + t[32]) + 1./8. * ( + t[8] + t[48]);
		dst[rIdx + 3 * outW + 2] = + 32./1. * ( + t[5] + t[24]) + 1./1. * ( + t[7] + t[29]) + 2./1. * ( + t[11] + t[45]) + 1./4. * ( + t[16] + t[27]) + 1./2. * ( + t[1] + t[44]) + 8./1. * ( + t[0] + t[37]) + 1./8. * ( + t[3] + t[31]) + 1./32. * ( + t[4] + t[51]) + 4./1. * ( + t[2] + t[39]);
		dst[rIdx + 3 * outW + 3] = + 8./1. * ( + t[2] + t[6] + t[20] + t[38]) + 1./1. * ( + t[9] + t[12] + t[15] + t[32] + t[34] + t[35]) + 1./64. * ( + t[10] + t[49]) + 64./1. * ( + t[5] + t[28]) + 1./8. * ( + t[8] + t[14] + t[23] + t[48]);
		dst[rIdx + 3 * outW + 4] = + 128./1. * ( + t[5] + t[24]) + 1./1. * ( + t[7] + t[29]) + 2./1. * ( + t[1] + t[44]) + 1./2. * ( + t[11] + t[45]) + 8./1. * ( + t[0] + t[37]) + 16./1. * ( + t[2] + t[39]) + 1./128. * ( + t[4] + t[51]) + 1./16. * ( + t[16] + t[27]) + 1./8. * ( + t[3] + t[31]);
		dst[rIdx + 4 * outW + 1] = + 32./1. * ( + t[13] + t[40]) + 1./1. * ( + t[7] + t[25]) + 2./1. * ( + t[17] + t[42]) + 1./2. * ( + t[14] + t[18]) + 8./1. * ( + t[9] + t[52]) + 16./1. * ( + t[6] + t[43]) + 1./32. * ( + t[4] + t[19]) + 1./16. * ( + t[3] + t[33]) + 1./8. * ( + t[12] + t[22]);
		dst[rIdx + 4 * outW + 2] = + 64./1. * ( + t[13] + t[46]) + 1./1. * ( + t[15] + t[21]) + 1./4. * ( + t[1] + t[16] + t[41] + t[53]) + 1./64. * ( + t[10] + t[50]) + 16./1. * ( + t[0] + t[26]) + 1./16. * ( + t[8] + t[47]) + 4./1. * ( + t[11] + t[17] + t[30] + t[36]);
		dst[rIdx + 4 * outW + 3] = + 128./1. * ( + t[13] + t[40]) + 1./1. * ( + t[7] + t[25]) + 2./1. * ( + t[9] + t[52]) + 1./2. * ( + t[12] + t[22]) + 8./1. * ( + t[17] + t[42]) + 16./1. * ( + t[6] + t[43]) + 1./128. * ( + t[4] + t[19]) + 1./16. * ( + t[3] + t[33]) + 1./8. * ( + t[14] + t[18]);
		dst[rIdx + 4 * outW + 4] = + 16./1. * ( + t[0] + t[17] + t[26] + t[30]) + 1./1. * ( + t[1] + t[11] + t[15] + t[21] + t[36] + t[41]) + 1./256. * ( + t[10] + t[50]) + 1./16. * ( + t[8] + t[16] + t[47] + t[53]) + 256./1. * ( + t[13] + t[46]);
		t[22] = + src[mIdx + 8 * gap] - src[mIdx + 16 * gap];
		t[19] = + src[mIdx + 24 * gap] - src[mIdx + 32 * gap];
		t[18] = + src[mIdx + 40 * gap] - src[mIdx + 48 * gap];
		dst[rIdx + 1 * outW + 0] = + 1./1. * ( + t[2] + t[7] + t[16] + t[27] + t[29] + t[39] + t[22]) + 2./1. * ( + t[0] + t[5] + t[11] + t[24] + t[37] + t[45] + t[19]) + 1./2. * ( + t[1] + t[3] + t[4] + t[31] + t[44] + t[51] + t[18]);
		dst[rIdx + 2 * outW + 0] = + 1./1. * ( + t[15] + t[16] + t[17] + t[21] + t[30] + t[53] + t[55]) + 1./4. * ( + t[1] + t[8] + t[10] + t[41] + t[47] + t[50] + t[57]) + 4./1. * ( + t[0] + t[11] + t[13] + t[26] + t[36] + t[46] + t[58]);
		dst[rIdx + 3 * outW + 0] = + 8./1. * ( + t[0] + t[5] + t[11] + t[24] + t[37] + t[45] + t[19]) + 1./1. * ( + t[2] + t[7] + t[16] + t[27] + t[29] + t[39] + t[22]) + 1./8. * ( + t[1] + t[3] + t[4] + t[31] + t[44] + t[51] + t[18]);
		dst[rIdx + 4 * outW + 0] = + 16./1. * ( + t[0] + t[11] + t[13] + t[26] + t[36] + t[46] + t[58]) + 1./1. * ( + t[15] + t[16] + t[17] + t[21] + t[30] + t[53] + t[55]) + 1./16. * ( + t[1] + t[8] + t[10] + t[41] + t[47] + t[50] + t[57]);
		t[26] = + src[mIdx + 59 * gap] + src[mIdx + 60 * gap];
		t[21] = + src[mIdx + 57 * gap] + src[mIdx + 58 * gap];
		t[25] = + src[mIdx + 61 * gap] + src[mIdx + 62 * gap];
		dst[rIdx + 5 * outW + 0] = + 32./1. * ( + t[0] + t[5] + t[11] + t[24] + t[37] + t[45] + t[19]) + 1./1. * ( + src[mIdx + 56 * gap] + t[2] + t[7] + t[16] + t[27] + t[29] + t[39] + t[22] + t[21] + t[25] + t[26]) + 1./32. * ( + t[1] + t[3] + t[4] + t[31] + t[44] + t[51] + t[18]);
		t[13] = + src[mIdx + 61 * gap] - src[mIdx + 62 * gap];
		t[17] = + src[mIdx + 57 * gap] - src[mIdx + 58 * gap];
		t[18] = + src[mIdx + 59 * gap] - src[mIdx + 60 * gap];
		dst[rIdx + 5 * outW + 1] = + 32./1. * ( + t[6] + t[20]) + 1./1. * ( + t[15] + t[35] + t[17]) + 2./1. * ( + t[2] + t[38] + t[18]) + 1./64. * ( + t[10] + t[49]) + 1./2. * ( + t[14] + t[23] + t[13]) + 64./1. * ( + t[5] + t[28]) + 16./1. * ( + t[9] + t[34]) + 1./32. * ( + t[8] + t[48]) + 1./16. * ( + t[12] + t[32]);
		dst[rIdx + 5 * outW + 2] = + 32./1. * ( + t[0] + t[37]) + 1./1. * ( + t[7] + t[29] + t[21]) + 1./4. * ( + t[16] + t[27] + t[25]) + 8./1. * ( + t[11] + t[45]) + 128./1. * ( + t[5] + t[24]) + 1./8. * ( + t[1] + t[44]) + 1./32. * ( + t[3] + t[31]) + 1./128. * ( + t[4] + t[51]) + 4./1. * ( + t[2] + t[39] + t[26]);
		dst[rIdx + 5 * outW + 3] = + 32./1. * ( + t[6] + t[20]) + 1./1. * ( + t[15] + t[35] + t[17]) + 4./1. * ( + t[9] + t[34]) + 8./1. * ( + t[2] + t[38] + t[18]) + 256./1. * ( + t[5] + t[28]) + 1./8. * ( + t[14] + t[23] + t[13]) + 1./32. * ( + t[8] + t[48]) + 1./4. * ( + t[12] + t[32]) + 1./256. * ( + t[10] + t[49]);
		dst[rIdx + 5 * outW + 4] = + 512./1. * ( + t[5] + t[24]) + 1./1. * ( + t[7] + t[29] + t[21]) + 2./1. * ( + t[11] + t[45]) + 32./1. * ( + t[0] + t[37]) + 1./32. * ( + t[3] + t[31]) + 1./2. * ( + t[1] + t[44]) + 16./1. * ( + t[2] + t[39] + t[26]) + 1./512. * ( + t[4] + t[51]) + 1./16. * ( + t[16] + t[27] + t[25]);
		dst[rIdx + 5 * outW + 5] = + 32./1. * ( + t[2] + t[6] + t[20] + t[38] + t[59] + t[18]) + 1./1. * ( + src[mIdx + 63 * gap] + t[9] + t[12] + t[15] + t[32] + t[34] + t[35] + t[56] + t[17]) + 1./32. * ( + t[8] + t[14] + t[23] + t[48] + t[54] + t[13]) + 1./1024. * ( + t[10] + t[49]) + 1024./1. * ( + t[5] + t[28]);
	}
}

template <typename Dtype> 
__global__ void unpadDst_gpu_kernel(const Dtype *src, Dtype *dst,
		const int batchs, const int num_outputs,
		const int height_out_p, const int width_out_p,
		const int height_out, const int width_out, int tNums)
{
	CUDA_KERNEL_LOOP(idx, tNums) {
		int bIdx = idx / (num_outputs * height_out * width_out);
		int cIdx = idx / (height_out * width_out) % num_outputs;
		int yIdx = idx / width_out % height_out;
		int xIdx = idx % width_out;
		dst[idx] = src[((cIdx * batchs + bIdx) * height_out_p + yIdx) * width_out_p + xIdx]; 
	}
}

template <typename Dtype>
void winoWeight_gpu(const int num_inputs, const int num_outputs, 
					const Dtype *weight, Dtype *wino_weight, const int wino_tile_size )
{
	int num_kernels = num_inputs * num_outputs;

	if((wino_tile_size == 2) || (wino_tile_size == 12))
		winoWeight_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
			                         CAFFE_CUDA_NUM_THREADS>>>(weight, wino_weight, num_inputs, num_outputs, num_kernels); 
	else if((wino_tile_size == 4) || (wino_tile_size == 14))
		wino4x4Weight_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
			                         CAFFE_CUDA_NUM_THREADS>>>(weight, wino_weight, num_inputs, num_outputs, num_kernels); 
	else if((wino_tile_size == 6) || (wino_tile_size == 16))
		wino6x6Weight_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
			                         CAFFE_CUDA_NUM_THREADS>>>(weight, wino_weight, num_inputs, num_outputs, num_kernels); 

}

template void winoWeight_gpu<float>(const int num_inputs, const int num_outputs, 
									const float *weight, float *wino_weight, const int wino_tile_size); 
template void winoWeight_gpu<double>(const int num_inputs, const int num_outputs, 
									const double *weight, double *wino_weight, const int wino_tile_size); 




template <typename Dtype>
void padSrc_gpu(const int batchs, const int num_inputs, const int height, const int width, 
				const int height_pad, const int width_pad,
				int height_p, int width_p,
				const Dtype *input, Dtype *input_pad)
{
	int num_kernels = batchs * num_inputs * height_p * width_p;

	padSrc_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(input, input_pad, height, width, height_p, width_p, num_inputs, batchs, height_pad, 0, num_kernels); 
}

template void padSrc_gpu<float>(const int batchs, const int num_inputs, const int height, const int width, 
				const int height_pad, const int width_pad,
				int height_p, int width_p,
				const float *input, float *input_pad); 
template void padSrc_gpu<double>(const int batchs, const int num_inputs, const int height, const int width, 
				const int height_pad, const int width_pad,
				int height_p, int width_p,
				const double *input, double *input_pad); 


template <typename Dtype>
void winoSrc_gpu(const int batchs, const int num_inputs, const int tileH, const int tileW, 
				const int height, const int width, // include padding 
				const Dtype *m_matrix, Dtype *v_matrix, const int wino_tile_size)
{
	int num_kernels = batchs * num_inputs * tileH * tileW;

	if(wino_tile_size == 2)
	{
		winoSrc_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
				                         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, v_matrix, height, width,  tileH, tileW, num_inputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 12)
	{
		winoSrcAddOpt_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
			                         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, v_matrix, height, width,  tileH, tileW, num_inputs, batchs, num_kernels); 
	}
	else if (wino_tile_size == 4)
	{
		wino4x4Src_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
				                         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, v_matrix, height, width,  tileH, tileW, num_inputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 14)
	{
		wino4x4SrcAddOpt_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
			                         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, v_matrix, height, width,  tileH, tileW, num_inputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 6)
	{
		wino6x6Src_gpu_kernel<Dtype><<< CAFFE_GET_BLOCKS(num_kernels),
				                         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, v_matrix, height, width,  tileH, tileW, num_inputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 16)
	{
        int t = 256;
        int b = (num_kernels + t - 1) / t;
		wino6x6SrcAddOpt_gpu_kernel<Dtype><<<b, t>>>(m_matrix, v_matrix, height, width,  tileH, tileW, num_inputs, batchs, num_kernels); 
	}
}

template void winoSrc_gpu<float>(const int batchs, const int num_inputs, const int tileH, const int tileW, 
						const int height, const int width, // include padding 
						const float *m_matrix, float *v_matrix, const int wino_tile_size); 
template void winoSrc_gpu<double>(const int batchs, const int num_inputs, const int tileH, const int tileW, 
						const int height, const int width, // include padding 
						const double *m_matrix, double *v_matrix, const int wino_tile_size); 



template <typename Dtype>
void winoMulti_gpu(const int batchs, const int num_inputs, const int num_outputs, const int tileH, const int tileW, 
					const Dtype *u_matrix, Dtype *v_matrix, Dtype *m_matrix, const int wino_tile_size)
{
	int M = num_outputs, N = tileH * tileW * batchs, K = num_inputs;
	int MM = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int NN = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int batched = (wino_tile_size + 2) * (wino_tile_size + 2); 
	dim3 numBlocks(NN, MM, batched);
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	winoMulti_gpu_kernel<Dtype><<<numBlocks, threadsPerBlock>>>(u_matrix, v_matrix, m_matrix, M, N, K);
}

template void winoMulti_gpu<float>(const int batchs, const int num_inputs, const int num_outputs, const int tileH, const int tileW, 
									const float *u_matrix, float *v_matrix, float *m_matrix, const int wino_tile_size); 
template void winoMulti_gpu<double>(const int batchs, const int num_inputs, const int num_outputs, const int tileH, const int tileW, 
									const double *u_matrix, double *v_matrix, double *m_matrix, const int wino_tile_size); 




template <typename Dtype>
void winoDst_gpu(const int batchs, const int num_outputs, const int tileH, const int tileW, const int height, const int width,
				 Dtype *m_matrix, Dtype *output, const int wino_tile_size)
{
	
	int num_kernels = batchs * num_outputs * tileH * tileW;

	if(wino_tile_size == 2)
	{
		winoDst_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
					                 CAFFE_CUDA_NUM_THREADS>>>(m_matrix, output, tileH, tileW, height, width, num_outputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 12)
	{
		winoDstAddOpt_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
								         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, output, tileH, tileW, height, width, num_outputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 4)
	{
		wino4x4Dst_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
					                 CAFFE_CUDA_NUM_THREADS>>>(m_matrix, output, tileH, tileW, height, width, num_outputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 14)
	{
		wino4x4DstAddOpt_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
								         CAFFE_CUDA_NUM_THREADS>>>(m_matrix, output, tileH, tileW, height, width, num_outputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 6)
	{
        int t = 256;
        int b = (num_kernels + t - 1) / t;
		wino6x6Dst_gpu_kernel<Dtype><<<b, t>>>(m_matrix, output, tileH, tileW, height, width, num_outputs, batchs, num_kernels); 
	}
	else if(wino_tile_size == 16)
	{
        int t = 256;
        int b = (num_kernels + t - 1) / t;
		wino6x6DstAddOpt_gpu_kernel<Dtype><<<b, t>>>(m_matrix, output, tileH, tileW, height, width, num_outputs, batchs, num_kernels); 
	}
}

template void winoDst_gpu(const int batchs, const int num_outputs, const int tileH, const int tileW, const int height, const int width,
						 float *m_matrix, float *output, const int wino_tile_size); 

template void winoDst_gpu(const int batchs, const int num_outputs, const int tileH, const int tileW, const int height, const int width,
						 double *m_matrix, double *output, const int wino_tile_size); 

template <typename Dtype>
void unpadDst_gpu(const int batchs, const int num_outputs,
		const int height_out_p, const int width_out_p,
		const int height_out, const int width_out,
		const Dtype *o_matrix, Dtype *output)
{
	int num_kernels = batchs * num_outputs * height_out * width_out;
	unpadDst_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(o_matrix, output, batchs, num_outputs, height_out_p, width_out_p, height_out, width_out, num_kernels); 
}

template void unpadDst_gpu(const int batchs, const int num_outputs,
		const int height_out_p, const int width_out_p,
		const int height_out, const int width_out,
		const float *o_matrix, float *output);

template void unpadDst_gpu(const int batchs, const int num_outputs,
		const int height_out_p, const int width_out_p,
		const int height_out, const int width_out,
		const double *o_matrix, double *output);

} // namespaece caffe 
