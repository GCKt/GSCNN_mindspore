"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import mindspore
from mindspore import nn as nn
from mindspore.numpy import tile
import numpy as np


def calc_pad_same(in_siz, out_siz, stride, ksize):
    """Calculate same padding width.
    Args:
    ksize: kernel size [I, J].
    Returns:
    pad_: Actual padding width.
    """
    return (out_siz - 1) * stride + ksize - in_siz


def conv2d_same(input, kernel, groups,bias=None,stride=1,padding=0,dilation=1):
    n, c, h, w = input.shape
    kout, ki_c_g, kh, kw = kernel.shape
    pw = calc_pad_same(w, w, 1, kw)
    ph = calc_pad_same(h, h, 1, kh)
    pw_l = pw // 2
    pw_r = pw - pw_l
    ph_t = ph // 2
    ph_b = ph - ph_t

    pad_op = nn.Pad(paddings=((0, 0), (0, 0), (ph_t, ph_b), (pw_l, pw_r)))
    input_ = pad_op(input)
    kernel_shape = kernel.shape
    conv2d = mindspore.ops.Conv2D(out_channel=kernel_shape[0], kernel_size=(kernel_shape[2], kernel[3]), stride=stride, dilation=dilation)
    result = conv2d(input_, kernel, group=groups)
    assert result.shape == input.shape
    return result


def gradient_central_diff(input, cuda):
    return input, input
    # kernel = [[1, 0, -1]]
    # kernel_t = 0.5 * mindspore.Tensor(kernel) * -1.  # pytorch implements correlation instead of conv
    # if type(cuda) is int:
    #     if cuda != -1:
    #         kernel_t = kernel_t.cuda(device=cuda)
    # else:
    #     if cuda is True:
    #         kernel_t = kernel_t.cuda()
    # n, c, h, w = input.shape
    #
    # unsqueeze = mindspore.ops.ExpandDims()
    # kernel_t = unsqueeze(kernel_t, 0)
    # kernel_t = unsqueeze(kernel_t, 0)
    # kernel_t = tile(kernel_t, ([c, 1, 1, 1]))
    # kernel_ty = unsqueeze(kernel_t.t(), 0)
    # kernel_ty = unsqueeze(kernel_ty, 0)
    # kernel_ty = tile(kernel_ty, ([c, 1, 1, 1]))
    #
    # x = conv2d_same(input, kernel_t, c)
    # y = conv2d_same(input, kernel_ty, c)
    # return x, y





def numerical_gradients_2d(input, cuda=False):
    """
    numerical gradients implementation over batches using torch group conv operator.
    the single sided differences are re-computed later.
    it matches np.gradient(image) with the difference than here output=x,y for an image while there output=y,x
    :param input: N,C,H,W
    :param cuda: whether or not use cuda
    :return: X,Y
    """
    n, c, h, w = input.shape
    #assert h > 1 and w > 1
    x, y = gradient_central_diff(input, cuda)
    return x, y


def convTri(input, r, cuda=False):
    """
    Convolves an image by a 2D triangle filter (the 1D triangle filter f is
    [1:r r+1 r:-1:1]/(r+1)^2, the 2D version is simply conv2(f,f'))
    :param input:
    :param r: integer filter radius
    :param cuda: move the kernel to gpu
    :return:
    """
    # if (r <= 1):
    #     raise ValueError()
    n, c, h, w = input.shape
    return input
    # f = list(range(1, r + 1)) + [r + 1] + list(reversed(range(1, r + 1)))
    # kernel = mindspore.Tensor([f]) / (r + 1) ** 2
    # if type(cuda) is int:
    #     if cuda != -1:
    #         kernel = kernel.cuda(device=cuda)
    # else:
    #     if cuda is True:
    #         kernel = kernel.cuda()
    #
    # # padding w
    # pad_op = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (1, 1)), mode='SYMMETRIC')
    # input_ = pad_op(input)
    #
    # pad_opnew = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (r, r)), mode='REFLECT')
    # input_ = pad_opnew(input_)
    # input_ = [input_[:, :, :, :r], input, input_[:, :, :, -r:]]
    # concat_op = mindspore.ops.Concat(axis=3)
    #
    # out = concat_op((input_[0], input_[1]))
    # input_ = concat_op((out, input_[2]))
    #
    # t = input_
    #
    # # padding h
    # pad_newop = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (0, 0)), mode='SYMMETRIC')
    # input_ = pad_newop(input)
    #
    # pad_xinop = nn.Pad(paddings=((0, 0), (0, 0), (r, r), (0, 0)), mode='REFLECT')
    # input_ = pad_xinop(input_)
    #
    # input_ = [input_[:, :, :r, :], t, input_[:, :, -r:, :]]
    # concat_op = mindspore.ops.Concat(axis=2)
    #
    # out = concat_op((input_[0], input_[1]))
    # input_ = concat_op((out, input_[2]))
    #
    #
    # unsqueeze = mindspore.ops.ExpandDims()
    # kernel_t = unsqueeze(kernel_t, 0)
    # kernel_t = unsqueeze(kernel_t, 0)
    # kernel_t = tile(kernel_t, ([c, 1, 1, 1]))
    # kernel_ty = unsqueeze(kernel_t.t(), 0)
    # kernel_ty = unsqueeze(kernel_ty, 0)
    # kernel_ty = tile(kernel_ty, ([c, 1, 1, 1]))
    #
    # kernel_shape = kernel.shape
    # conv2d_one = mindspore.ops.Conv2D(out_channel=kernel_shape[0], kernel_size=(kernel_shape[2], kernel_shape[3]))
    #
    # output = conv2d_one(input_, kernel_t)
    # kernelt_shape = kernel_ty.shape
    # conv2d_two = mindspore.ops.Conv2D(out_channel=kernelt_shape[0], kernel_size=(kernelt_shape[2], kernelt_shape[3]))
    # output = conv2d_two(output, kernel_ty)
    # return output





def compute_grad_mag(E, cuda=False):
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    sqrt = mindspore.ops.Sqrt()
    mul = mindspore.ops.Mul()
    mag = sqrt(mul(Ox,Ox) + mul(Oy,Oy) + 1e-6)
    mag = mag / mag.max();

    return mag


# inputs = mindspore.Tensor(np.ones((2, 3, 24, 24)).astype("float32"))
# output = compute_grad_mag(inputs, cuda=False)
# #print(output)
# from numpy import random
#
# randArray = random.random(size=(2, 3, 24, 24))
# input = randArray
# input = mindspore.Tensor.from_numpy(input)
# output1 = compute_grad_mag(input, cuda=False)
# #print(output1)
