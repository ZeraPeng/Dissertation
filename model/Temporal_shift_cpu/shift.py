import torch
from torch.autograd import Function
import torch.nn as nn
from torch.nn import Module, Parameter

class ShiftFunction(Function):
    @staticmethod
    def forward(ctx, input, xpos, ypos, stride):
        # Ensure input tensors are contiguous for efficient operations
        if not input.is_contiguous():
            input = input.contiguous()
        if not xpos.is_contiguous():
            xpos = xpos.contiguous()
        if not ypos.is_contiguous():
            ypos = ypos.contiguous()
            
        batch_size, channels, height, width = input.shape
        output = input.new_zeros(input.size())
        
        # Implement shift operation
        for b in range(batch_size):
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        # Calculate shifted positions
                        shifted_h = h + int(ypos[h].item() * stride)
                        shifted_w = w + int(xpos[w].item() * stride)
                        
                        # Ensure positions are within bounds
                        shifted_h = max(0, min(shifted_h, height - 1))
                        shifted_w = max(0, min(shifted_w, width - 1))
                        
                        # Perform the shift
                        output[b, c, h, w] = input[b, c, shifted_h, shifted_w]

        # Save context for backward pass
        ctx.save_for_backward(input, xpos, ypos)
        ctx.stride = stride
        ctx.spatial_size = (height, width)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, xpos, ypos = ctx.saved_tensors
        stride = ctx.stride
        height, width = ctx.spatial_size
        
        # Initialize gradients
        grad_input = torch.zeros_like(input)
        grad_xpos = torch.zeros_like(xpos)
        grad_ypos = torch.zeros_like(ypos)
        
        batch_size, channels = input.shape[:2]
        
        # Compute gradients
        for b in range(batch_size):
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        # Calculate shifted positions
                        shifted_h = h + int(ypos[h].item() * stride)
                        shifted_w = w + int(xpos[w].item() * stride)
                        
                        # Ensure positions are within bounds
                        shifted_h = max(0, min(shifted_h, height - 1))
                        shifted_w = max(0, min(shifted_w, width - 1))
                        
                        # Accumulate gradients
                        grad_input[b, c, shifted_h, shifted_w] += grad_output[b, c, h, w]
                        
                        # Compute position gradients if needed
                        if shifted_h > 0 and shifted_h < height - 1:
                            grad_ypos[h] += grad_output[b, c, h, w] * (
                                input[b, c, shifted_h + 1, shifted_w] - 
                                input[b, c, shifted_h - 1, shifted_w]
                            ) * stride / 2
                            
                        if shifted_w > 0 and shifted_w < width - 1:
                            grad_xpos[w] += grad_output[b, c, h, w] * (
                                input[b, c, shifted_h, shifted_w + 1] - 
                                input[b, c, shifted_h, shifted_w - 1]
                            ) * stride / 2
        
        return grad_input, grad_xpos, grad_ypos, None


class Shift(Module):

    def __init__(self, channel, stride, init_scale=3):
        super(Shift, self).__init__()

        self.stride = stride

        self.xpos = Parameter(torch.zeros(channel,requires_grad=True,device='cpu')*1.5)
        self.ypos = Parameter(torch.zeros(channel,requires_grad=True,device='cpu')*1.5)

        self.xpos.data.uniform_(-1e-8,1e-8)
        self.ypos.data.uniform_(-init_scale,init_scale)

    def forward(self, input):
        return ShiftFunction.apply(input,self.xpos,self.ypos,self.stride)