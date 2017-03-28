import torch
from torch.nn import Module
from torch.autograd import Function, Variable, gradcheck
from _ext import smooth_l1_loss

import numpy as np

class SmoothL1LossFunc(Function):
    def forward(self, input, target, weights):
        self.save_for_backward(input, target, weights)

        output = torch.FloatTensor([1])
        smooth_l1_loss.smoothl1lossForward(input, target, output, weights)
        return output

    def backward(self, grad_output):
        input, target, weights = self.saved_tensors

        grad_input = torch.FloatTensor(input.size())
        smooth_l1_loss.smoothl1lossBackward(input, target, grad_input, weights)
        return grad_input, None, None

class SmoothL1Loss(Module):
    def forward(self, input, target, weights):
        return SmoothL1LossFunc()(input, target, weights)

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.loss = SmoothL1Loss()

    def forward(self, input, target, weights):
        return self.loss(input, target, weights)

if __name__ == "__main__":
    input = Variable(torch.zeros(5, 5, 5))

    net = Net()

    # |x| > 1
    weights1 = Variable(torch.ones(input.size()))
    target1 = Variable(torch.ones(input.size()) * 2.)
    result1 = net(input, target1, weights1)
    assert result1.data[0] == np.sum(target1.data.numpy() - 0.5)

    # |x| < 1
    weights2 = Variable(torch.ones(input.size()))
    target2 = Variable(torch.ones(input.size()) * 0.5)
    result2 = net(input, target2, weights2)
    assert result2.data[0] == np.sum(target2.data.numpy()*target2.data.numpy()*0.5)

    # zero weights 
    weights3 = Variable(torch.zeros(input.size()))
    target3 = Variable(torch.ones(input.size()) * 0.5)
    result3 = net(input, target3, weights3)
    assert result3.data[0] == 0.
