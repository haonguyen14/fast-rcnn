void smoothl1lossForward(
          THFloatTensor* input,
          THFloatTensor* target,
          THFloatTensor* output,
          THFloatTensor* weights);

void smoothl1lossBackward(
          THFloatTensor* input,
          THFloatTensor* target,
          THFloatTensor* grad_input,
          THFloatTensor* weights);
