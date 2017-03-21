import torch
import torch.autograd as autograd
import numpy as np


class SmoothL1LossFunc(autograd.Function):

    def _f(self, abs_x):
        return (0.5*abs_x*abs_x) if abs_x < 1 else (abs_x-0.5)

    def _f_derivative(self, x):
        if x < -1.:
            return -1.0
        if x > 1.:
            return 1.0
        return x 

    def forward(self, input, target, weights):
        self.save_for_backward(input, target, weights)

        # apply smooth l1 function
        diffs = input - target

        x = torch.abs(diffs)
        x.apply_(self._f)
        x.mul_(weights)

        # sum of all losses
        x = torch.sum(x)

        return torch.Tensor([x])

    def backward(self, grad_output):
        input, target, weights = self.saved_tensors
        
        x = input - target
        x.apply_(self._f_derivative)
        x.mul_(weights)

        return (x, None, None, None)
