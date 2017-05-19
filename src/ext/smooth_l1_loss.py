import torch
from torch.nn import Module
from torch.autograd import Function, Variable, gradcheck
from _ext import smooth_l1_loss

import numpy as np

class SmoothL1LossFunc(Function):
    def forward(self, input, target, weights, sigma):
        self.save_for_backward(input, target, weights)
        output = torch.FloatTensor([1])

        if input.is_cuda:
            output = output.cuda()
            smooth_l1_loss.smoothl1lossForwardCuda(input, target, output, weights)
        else:
            smooth_l1_loss.smoothl1lossForward(input, target, output, weights)

        return output

    def backward(self, grad_output):
        input, target, weights = self.saved_tensors
        grad_input = input.new()

        if grad_output.is_cuda:
           smooth_l1_loss.smoothl1lossBackwardCuda(input, target, grad_input, weights)
        else:
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
    print("Pass |x| > 1")

    weights1 = weights1.cuda() 
    target1 = target1.cuda()
    input1 = Variable(torch.zeros(input.size())).cuda()
    result1 = net(input1, target1, weights1).cuda()
    assert result1.data[0] == torch.sum(target1 - 0.5)
    print("Pass |x| > 1 CUDA")

    # |x| < 1
    weights2 = Variable(torch.ones(input.size()))
    target2 = Variable(torch.ones(input.size()) * 0.5)
    result2 = net(input, target2, weights2)
    assert result2.data[0] == np.sum(target2.data.numpy()*target2.data.numpy()*0.5)
    print("Pass |x| < 1")

    weights2 = weights2.cuda()
    target2 = target2.cuda()
    input2 = Variable(torch.zeros(input.size())).cuda()
    result2 = net(input2, target2, weights2).cuda()
    assert result2.data[0] == torch.sum(target2*target2*0.5)
    print("Pass |x| < 1 CUDA")

    # zero weights 
    weights3 = Variable(torch.zeros(input.size()))
    target3 = Variable(torch.ones(input.size()) * 0.5)
    result3 = net(input, target3, weights3)
    assert result3.data[0] == 0.
    print("Pass zero weights")

    weights3 = weights3.cuda()
    target3 = target3.cuda()
    input3 = Variable(torch.zeros(input.size())).cuda()
    result3 = net(input3, target3, weights3).cuda()
    assert result3.data[0] == 0.
    print("Pass zero weights CUDA")

    #  autograd check
    input4 = Variable(torch.rand(30, 30, 30), requires_grad=True)
    target4 = Variable(torch.rand(30, 30, 30))
    weights4 = Variable(torch.ones(30, 30, 30))
    test = gradcheck(SmoothL1LossFunc(), (input4, target4, weights4))
    print("Grad check CPU: %s" % test)

    input4 = Variable(torch.rand(30, 30, 30).cuda(), requires_grad=True)
    target4 = target4.cuda()
    weights4 = weights4.cuda()
    test = gradcheck(SmoothL1Loss(), (input4, target4, weights4))
    print("Grad check GPU: %s" % test)
