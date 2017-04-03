#include <THC/THC.h>
#include "smooth_l1_loss_cuda.h"

extern THCState* state;

void smoothl1lossForwardCuda(
		THCudaTensor* input,
		THCudaTensor* target,
		THCudaTensor* output,
		THCudaTensor* weights) {
	
	ptrdiff_t size = THCudaTensor_nElement(state, input);	

	input = THCudaTensor_newContiguous(state, input);
	target = THCudaTensor_newContiguous(state, target);
	weights = THCudaTensor_newContiguous(state, weights);

	cudaStream_t stream = THCState_getCurrentStream(state);
	float *input_ptr = THCudaTensor_data(state, input);
	float *target_ptr = THCudaTensor_data(state, target);
	float *weight_ptr = THCudaTensor_data(state, weights);
	
	float loss  = smoothl1lossForward_cuda(stream, input_ptr, target_ptr, weight_ptr, size);

	THCudaTensor_free(state, input);
	THCudaTensor_free(state, target);
	THCudaTensor_free(state, weights);

	THCudaTensor_set1d(state, output, 0, loss);
}

void smoothl1lossBackwardCuda(
		THCudaTensor* input,
		THCudaTensor* target,
		THCudaTensor* grad_input,
		THCudaTensor* weights) {

	THCudaTensor_resizeAs(state, grad_input, input);
	ptrdiff_t size = THCudaTensor_nElement(state, input);

	input = THCudaTensor_newContiguous(state, input);
	target = THCudaTensor_newContiguous(state, target);
	weights = THCudaTensor_newContiguous(state, weights);

	cudaStream_t stream = THCState_getCurrentStream(state);
	float *input_ptr = THCudaTensor_data(state, input);
	float *target_ptr = THCudaTensor_data(state, target);
	float *weight_ptr = THCudaTensor_data(state, weights);
	float *grad_input_ptr= THCudaTensor_data(state, grad_input);

	smoothl1lossBackward_cuda(stream, input_ptr, target_ptr, grad_input_ptr, weight_ptr, size);

	THCudaTensor_free(state, input);
	THCudaTensor_free(state, target);
	THCudaTensor_free(state, weights);
}
