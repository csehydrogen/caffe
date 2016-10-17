#include "caffe/util/mm_func.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
	const int BLOCK_SIZE = 32;
	template <typename Dtype>
	void __global__ matrix_multiply_kernel(
		const int M, const int N, const int K,
		const Dtype* A, const Dtype* B, Dtype* C)
	{
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
	void matrix_multiply(const int M, const int N, const int K,
		const Dtype* A, const Dtype* B, Dtype* C)
	{
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid(N / BLOCK_SIZE, M / BLOCK_SIZE);
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
