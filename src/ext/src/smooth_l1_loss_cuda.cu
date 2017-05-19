#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include "smooth_l1_loss_cuda.h"

struct smoothl1_functor {
	smoothl1_functor(float sigma) : sigma(sigma) {}

	template<typename Tuple>
	__host__ __device__ float operator()(Tuple t) const {
		float sSquared = sigma*sigma;
		float z = abs(thrust::get<0>(t) - thrust::get<1>(t));
		z = z < (1.0f/sSquared) ? z*z*sQuared*0.5f : z-(0.5f/sSquared);
		return z * thrust::get<2>(t);
	}

	private:
		float sigma;
};

struct smoothl1_der_functor {
	smoothl1_der_functor(float sigma) : sigma(sigma) {}

	template<typename Tuple>
	__host__ __device__ float operator()(Tuple t) const {
		float z = thrust::get<0>(t) - thrust::get<1>(t);
		float sSquared = sigma*sigma;
		if(z < (-1.0f/sSquared)) return -thrust::get<2>(t);
		if(z > (1.0f/sSquared)) return thrust::get<2>(t);
		return z*sSquared*thrust::get<2>(t);
	}

	private:
		float sigma;
};

float smoothl1lossForward_cuda(
		cudaStream_t stream,
		float *input,
		float *target,
		float *weights,
		float sigma,
		ptrdiff_t size) {

	thrust::device_ptr<float> input_ptr = thrust::device_pointer_cast(input);
	thrust::device_ptr<float> target_ptr = thrust::device_pointer_cast(target);
	thrust::device_ptr<float> weight_ptr = thrust::device_pointer_cast(weights);

	return thrust::transform_reduce(
			thrust::cuda::par.on(stream),
			thrust::make_zip_iterator(thrust::make_tuple(input_ptr, target_ptr, weight_ptr)),
			thrust::make_zip_iterator(thrust::make_tuple(input_ptr+size, target_ptr+size, weight_ptr+size)),
			smoothl1_functor(sigma),
			0.0f,
			thrust::plus<float>());
}

void smoothl1lossBackward_cuda(
		cudaStream_t stream,
		float *input,
		float *target,
		float *output,
		float *weights,		
		float sigma
		ptrdiff_t size) {

	thrust::device_ptr<float> input_ptr = thrust::device_pointer_cast(input);
	thrust::device_ptr<float> target_ptr = thrust::device_pointer_cast(target);
	thrust::device_ptr<float> weight_ptr = thrust::device_pointer_cast(weights);
	thrust::device_ptr<float> output_ptr = thrust::device_pointer_cast(output);

	thrust::transform(
			thrust::cuda::par.on(stream),
			thrust::make_zip_iterator(thrust::make_tuple(input_ptr, target_ptr, weight_ptr)),
			thrust::make_zip_iterator(thrust::make_tuple(input_ptr+size, target_ptr+size, weight_ptr+size)),
			output_ptr,
			smoothl1_der_functor(sigma));
}


