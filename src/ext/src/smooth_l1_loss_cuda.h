#ifdef __cplusplus
extern "C" {
#endif

float smoothl1lossForward_cuda(
		cudaStream_t stream,
		float *input,
		float *target,
		float *weights,
		float sigma,
		ptrdiff_t size);

void smoothl1lossBackward_cuda(
		cudaStream_t stream,
		float *input,
		float *target,
		float *output,
		float *weights,
		float sigma,
		ptrdiff_t size);

#ifdef __cplusplus
}
#endif
