#include "caffe/util/mm_func.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
	template <typename Dtype>
	void __global__ matrix_multiply_kernel(
		const int M, const int N, const int K,
		const Dtype* A, const Dtype* B, Dtype* C)
	{
		
	}	
	
	template <typename Dtype>
	void matrix_multiply(const int M, const int N, const int K,
		const Dtype* A, const Dtype* B, Dtype* C)
	{
		//matrix_multiply_kernel<Dtype><<<>>>(M,N,K,A,B,C);			
		
		caffe_gpu_gemm<Dtype>(
			CblasNoTrans, CblasNoTrans, M, N, K,
			(Dtype)1., A, B,(Dtype)0., C);
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