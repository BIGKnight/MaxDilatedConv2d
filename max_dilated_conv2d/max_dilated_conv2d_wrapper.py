import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Module
import max_dilated_conv2d_gpu as max_dilated_conv2d


class MaxDilatedFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 8:
            print("wrong input parameters number, check the input")
            return
        input = args[0]
        weights = args[1]
        ctx.stride_h = args[2]
        ctx.stride_w = args[3]
        ctx.dilation_h = args[4]
        ctx.dilation_w = args[5]
        ctx.groups = args[6]
        ctx.im2col_step = args[7]
        output = max_dilated_conv2d.forward(input, weights, ctx.stride_h, ctx.stride_w, ctx.dilation_h, ctx.dilation_w, ctx.groups, ctx.im2col_step)
        ctx.save_for_backward(input, weights)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        if len(grad_outputs) != 1:
            print("Wrong output number, check your output")
            return
        input, weights = ctx.saved_tensors
        grad_input, grad_weight = max_dilated_conv2d.backward(input, weights, grad_outputs[0], ctx.stride_h, ctx.stride_w, ctx.dilation_h, ctx.dilation_w, ctx.groups, ctx.im2col_step)
        return grad_input, grad_weight, None, None, None, None, None, None


class MaxDilatedConv2dLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_h, stride_w, dilation_h, dilation_w, groups, im2col_step):
        super(MaxDilatedConv2dLayer, self).__init__()
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.im2col_step = im2col_step
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight, gain=1)


    def forward(self, inputs):
        return MaxDilatedFunction.apply(inputs, self.weight, self.stride_h, self.stride_w, self.dilation_h, self.dilation_w, self.groups, self.im2col_step)

class BasicMaxDilatedConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_max, groups=1, im2col_step=1):
        super(BasicMaxDilatedConv2D, self).__init__()
        self.pad = dilation_max * (kernel_size // 2)
        
        self.max_dilated_conv2d = MaxDilatedConv2dLayer(in_channels, out_channels, kernel_size, 1, 1, dilation_max, dilation_max, groups, im2col_step)
        
    def forward(self, x):
        x = nn.functional.pad(x, [self.pad, self.pad + 1, self.pad, self.pad + 1])
        return self.max_dilated_conv2d(x)

