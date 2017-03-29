float smoothl1lossForward_cuda(
		cudaStream_t stream,
		float *input,
		float *target,
		float *weights,
		ptrdiff_t size);
