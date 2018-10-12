from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)
        self.reduction = reduction


class Bce_logit_weight(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean', pos_weight=None):
        super(Bce_logit_weight, self).__init__(size_average, weight, reduction)
        self.register_buffer('pos_weight', pos_weight)
        self.reduce = reduce

    def forward(self, input, target, weight):
        _assert_no_grad(target)
        return F.binary_cross_entropy_with_logits(input, target, weight, pos_weight=self.pos_weight, reduction=self.reduction)