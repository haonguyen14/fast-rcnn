#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include "smooth_l1_loss_cuda.h"

struct smoothl1_functor {
	template<typename Tuple>
	__host__ __device__ float operator()(Tuple t) const {
		float z = abs(thrust::get<0>(t) - thrust::get<1>(t));
		z = z < 1.0f ? z*0.5f*0.5f : z-0.5f;
		return z * thrust::get<2>(t);
	}
};

float smoothl1lossForward_cuda(
		cudaStream_t stream,
		float *input,
		float *target,
		float *weights,
		ptrdiff_t size) {

	thrust::device_ptr<float> input_ptr = thrust::device_pointer_cast(input);
	thrust::device_ptr<float> target_ptr = thrust::device_pointer_cast(target);
	thrust::device_ptr<float> weight_ptr = thrust::device_pointer_cast(weights);

	return thrust::transform_reduce(
			thrust::make_zip_iterator(thrust::make_tuple(input_ptr, target_ptr, weight_ptr)),
			thrust::make_zip_iterator(thrust::make_tuple(input_ptr+size, target_ptr+size, weight_ptr+size)),
			smoothl1_functor(),
			0.0f,
			thrust::plus<float>());
}
