void smoothl1lossForwardCuda(
		THCudaTensor* input,
		THCudaTensor* target,
		THCudaTensor* output,
		THCudaTensor* weights);

void smoothl1lossBackwardCuda(
		THCudaTensor* input,
		THCudaTensor* target,
		THCudaTensor* grad_input,
		THCudaTensor* weights);
