#ifndef MM_FUNC_HPP
#define MM_FUNC_HPP

namespace caffe
{
	template <typename Dtype>
	void matrix_multiply(const int M, const int N, const int K,
		const Dtype* A, const Dtype* B, Dtype* C);	
}


#endif