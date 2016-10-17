#include "caffe/util/mm_func.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
	template <typename Dtype>
	void __global__ matrix_multiply_kernel(
		const int M, const int N, const int K,
		const Dtype* A, const Dtype* B, Dtype* C)
	{
		Dtype s = 0;
		int r = blockIdx.y * blockDim.y + threadIdx.y;
		int c = blockIdx.x * blockDim.x + threadIdx.x;
		if (r >= M || c >= N) return;
		for (int i = 0; i < K; ++i) s += A[r * K + i] * B[i * N + c];
		C[r * N + c] = s;
	}	
	
	template <typename Dtype>
	void matrix_multiply(const int M, const int N, const int K,
		const Dtype* A, const Dtype* B, Dtype* C)
	{
		dim3 dimBlock(32, 32);
		dim3 dimGrid(N / dimBlock.x, M / dimBlock.y);
		matrix_multiply_kernel<<<dimGrid, dimBlock>>>(M, N, K, A, B, C);
	}
	
	template
	void matrix_multiply<float>(
		const int M, const int N, const int K,
		const float* A, const float* B, float* C);
		
	template
	void matrix_multiply<double>(
		const int M, const int N, const int K,
		const double* A, const double* B, double* C);
		
}
