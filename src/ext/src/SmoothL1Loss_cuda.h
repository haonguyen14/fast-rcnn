void smoothl1lossForwardCuda(
		THCudaTensor* input,
		THCudaTensor* target,
		THCudaTensor* output,
		THCudaTensor* weights,
		float sigma);

void smoothl1lossBackwardCuda(
		THCudaTensor* input,
		THCudaTensor* target,
		THCudaTensor* grad_input,
		THCudaTensor* weights,
		float sigma);
