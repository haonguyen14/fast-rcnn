#include <TH/TH.h>

void smoothl1lossForward(
          THFloatTensor* input,
          THFloatTensor* target,
          THFloatTensor* output,
          THFloatTensor* weights) {

    float sum = 0.0;
    TH_TENSOR_APPLY3(float, input, float, target, float, weights,
        float z = fabs(*input_data - *target_data);
        z = z < 1 ? 0.5*z*z : z-0.5;
        sum += (z * *weights_data);
    );

    THFloatTensor_set1d(output, 0, sum);
}


void smoothl1lossBackward(
          THFloatTensor* input,
          THFloatTensor* target,
          THFloatTensor* grad_input,
          THFloatTensor* weights) {

    THFloatTensor* diff = THFloatTensor_newWithSize(
            THFloatTensor_newSizeOf(input),
            THFloatTensor_newStrideOf(input));

    THFloatTensor_csub(diff, input, 1.0, target);

    TH_TENSOR_APPLY3(float, grad_input, float, diff, float, weights,
       if(*diff_data < -1.) {
            *grad_input_data = - *weights_data;
       } else if(*diff_data > 1) {
            *grad_input_data = *weights_data; 
       } else {
            *grad_input_data = *weights_data * *diff_data; 
       }
    );
}
