# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import mindspore
# import mindspore
# import numpy as np
# import mindspore
# import mindspore.ops as ops
# from mindspore import Tensor
# import numpy as np
# from mindspore import nn as nn
# from network.mynn import initialize_weights, Norm2d
# import cv2
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
# inputs = mindspore.Tensor(np.ones((2, 3, 24, 24)).astype("float32"))
# print(inputs.shape)
# class GlobalAvgPool2d(mindspore.nn.Cell):
#
#     def __init__(self):
#         """Global average pooling over the input's spatial dimensions"""
#         super(GlobalAvgPool2d, self).__init__()
#
#     def construct(self, inputs):
#         in_size = inputs.shape
#         return inputs.view((in_size[0], in_size[1], -1)).mean(axis=2)
#
# net = GlobalAvgPool2d()
# output = net(inputs)
# print(output)
#
# concat_op = ops.Concat(axis=1)
# cast_op = ops.Cast()
# a = Tensor(np.ones([2, 3]).astype(np.float32))
# b = Tensor(np.ones([2, 3]).astype(np.float32))
# alphas = concat_op((a, b))
# print(alphas.shape)
# inp = mindspore.Tensor(np.ones((2, 3, 24, 24)).astype("float32"))
# x_size = inp.shape
# canny = np.zeros([x_size[0], 1, x_size[2], x_size[3]])
# print(canny)
# im_arr = inp.asnumpy().transpose((0, 2, 3, 1)).astype(np.uint8)
# '''im_arr : torch.Size([2, 96, 96, 3])'''
# canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
# '''canny : torch.Size([2, 1, 96, 96])'''
# # canny获得边缘图像
# # cv2.Canny()方法可以获得图像的边缘图像
# for i in range(x_size[0]):
#     canny[i] = cv2.Canny(im_arr[i], 10, 100)
# canny = mindspore.Tensor.from_numpy(canny)
#
# # inputs = mindspore.Tensor(inputs, dtype=mindspore.double)
# # flow = np.load('flow_t2.npy', allow_pickle=True)
# # print(flow)
