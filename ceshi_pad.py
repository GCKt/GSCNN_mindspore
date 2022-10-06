# import math

# import mindspore
# from mindspore import nn as nn
# import numpy as np
# from mindspore.numpy import tile
# from mindspore import context
# context.set_context(device_id=0)
# context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
# # context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
# from loss import JointEdgeSegLoss
# from mindspore  import numpy as np_new
# # inputs = np.load("input.npy")
# # inputs = mindspore.Tensor.from_numpy(inputs)
# # pad_op = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (1, 1)), mode='SYMMETRIC')
# # input_ = pad_op(inputs)
# # pad_opnew = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (1, 1)), mode='REFLECT')
# # input_new = pad_opnew(inputs)
# # pad_opnewnew = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (1, 1)), mode='CONSTANT')
# # input_newnew = pad_opnewnew(inputs)
# # c = np.load("filename.npy")
# # c = mindspore.Tensor.from_numpy(c)
# # print(input_ == c)
# # print("-------------下一个------------")
# # print(input_new == c)
# # print("-------------下一个------------")
# # print(input_newnew == c)

# # kernel = [[1, 0, -1]]
# # kernel_t = 0.5 * mindspore.Tensor(kernel) * -1.
# # unsqueeze = mindspore.ops.ExpandDims()
# # kernel_t = unsqueeze(kernel_t, 0)
# # kernel_t = unsqueeze(kernel_t, 0)
# # kernel_t = tile(kernel_t, ([4, 1, 1, 1]))
# # #print(kernel_t.ndim)
# #
# # shape = (2, 2, 3)
# # input_x = mindspore.Tensor(np.array([1, 2, 3]).astype(np.float32))
# # broadcast_to = mindspore.ops.BroadcastTo(shape)
# # output = broadcast_to(input_x)
# # print("输出")
# # #print(output)
# #
# #
# from loss import JointEdgeSegLoss


# def bce2d(input, target):
#     n, c, h, w = input.shape
#     #contigus没有用
#     input_perm = (0, 2, 3, 1)
#     transpose = mindspore.ops.Transpose()
#     log_p = transpose(input, input_perm).view(1, -1)
#     target_t = transpose(target, input_perm).view(1, -1)


#     # target_trans = target_t.clone()

#     pos_index = (target_t == 1)
#     neg_index = (target_t == 0)
#     ignore_index = (target_t > 1)

#     # target_trans[pos_index] = 1
#     # target_trans[neg_index] = 0

#     pos_index = pos_index.asnumpy().astype(bool)
#     neg_index = neg_index.asnumpy().astype(bool)
#     ignore_index = ignore_index.asnumpy().astype(bool)

#     fill = mindspore.ops.Fill()
#     weight = fill(mindspore.float32, log_p.shape, 0)

#     weight = weight.asnumpy()
#     pos_num = pos_index.sum()
#     neg_num = neg_index.sum()
#     sum_num = pos_num + neg_num
#     print(weight.shape)
#     print(pos_index.shape)
#     weight[pos_index] = neg_num * 1.0 / sum_num
#     weight[neg_index] = pos_num * 1.0 / sum_num

#     weight[ignore_index] = 0

#     weight = mindspore.Tensor.from_numpy(weight)


#     sigmoid = nn.Sigmoid()
#     log_p = sigmoid(log_p)
#     binary_cross_entropy = mindspore.ops.BinaryCrossEntropy()
#     loss = binary_cross_entropy(log_p, target_t, weight)

#     return loss


# class JointEdgeSegLoss_new(nn.Cell):
#     def __init__(self, classes, weight=None, reduction='mean', ignore_index=255,
#                  norm=False, upper_bound=1.0, mode='train',
#                  edge_weight=1, seg_weight=1, att_weight=1, dual_weight=1, edge='none'):
#         super(JointEdgeSegLoss_new, self).__init__()
#         self.num_classes = classes
#         # if mode == 'train':
#         #     self.seg_loss = ImageBasedCrossEntropyLoss2d(
#         #         classes=classes, ignore_index=ignore_index, upper_bound=upper_bound)
#         # elif mode == 'val':
#         #     self.seg_loss = mindspore.nn.SoftmaxCrossEntropyWithLogits()

#         self.edge_weight = edge_weight
#         self.seg_weight = seg_weight
#         self.att_weight = att_weight
#         self.dual_weight = dual_weight

#         # self.dual_task = DualTaskLoss()
#         self.input_perm = (0, 2, 3, 1)
#         self.transpose = mindspore.ops.Transpose()
#         self.fill = mindspore.ops.Fill()
#         self.binary_cross_entropy = mindspore.ops.BinaryCrossEntropy()

#     def construct(self, input, target):
#         n, c, h, w = input.shape
#         # contigus没有用
#         """contigus"""
#         # input = mindspore.Tensor(input, dtype=mindspore.float32)
#         print("1")
#         input = input.astype(dtype=mindspore.float32)
#         #target = mindspore.Tensor(target, dtype=mindspore.int32)
#         #input_perm = (0, 2, 3, 1)
#         print("开始")
#         transpose = mindspore.ops.Transpose()
#         log_p = self.transpose(input, self.input_perm).view(1, -1)
#         target_t = self.transpose(target, self.input_perm).view(1, -1)
#         print("结束")
#         #target_t = mindspore.Tensor(target_t, dtype=mindspore.float32)
#         # target_trans = target_t.clone()
#         target_t_new = target_t
#         pos_index = (target_t_new == 1)
#         neg_index = (target_t_new == 0)
#         ignore_index = (target_t_new > 1)

#         # target_trans[pos_index] = 1
#         # target_trans[neg_index] = 0
#         #
#         # pos_index = pos_index.asnumpy().astype(bool)
#         # neg_index = neg_index.asnumpy().astype(bool)
#         # ignore_index = ignore_index.asnumpy().astype(bool)
#         #
#         # pos_index = mindspore.Tensor(pos_index, dtype = mindspore.bool_)
#         # neg_index = mindspore.Tensor(neg_index, dtype = mindspore.bool_)
#         # ignore_index = mindspore.Tensor(ignore_index, dtype = mindspore.bool_)

#         # fill = mindspore.ops.Fill()
#         weight = self.fill(mindspore.float32, log_p.shape, 0)


#         # weight = weight.asnumpy()

#         pos_index = pos_index.astype(mindspore.float32)
#         neg_index = neg_index.astype(mindspore.float32)
#         pos_num = pos_index.sum()
#         neg_num = neg_index.sum()
#         sum_num = pos_num + neg_num
#         print(sum_num)
#         pos_index = pos_index.astype(mindspore.int32)
#         neg_index = neg_index.astype(mindspore.int32)
#         print(weight.shape)
#         print(pos_index.shape)
#         shape_new = pos_index.shape[1]
#         print(shape_new)
#         # for i in range(shape_new):
#         #     if pos_index[0][i]:
#         #        weight[0][i] = neg_num * 1.0 / sum_num
#         #     if neg_index[0][i]:
#         #         weight[0][i] = pos_num * 1.0 / sum_num
#         # weight[pos_index] = neg_num * 1.0 / sum_num
#         # weight[neg_index] = pos_num * 1.0 / sum_num

#         oneslike = mindspore.ops.OnesLike()
#         filler1 = oneslike(weight) * neg_num * 1.0 / sum_num
#         filler2 = oneslike(weight) * pos_num * 1.0 / sum_num
#         filler3 = oneslike(weight) * 0.0
#         # print("概率")
#         # print(edge)

#         weight = np_new.where(pos_index, filler1, weight)
#         weight = np_new.where(neg_index, filler2, weight)
#         weight = np_new.where(ignore_index, filler3, weight)



#         #weight[ignore_index] = 0

#         #weight = mindspore.Tensor.from_numpy(weight)
#         #weight = None
#         sigmoid = nn.Sigmoid()
#         log_p = sigmoid(log_p)
#         # binary_cross_entropy = mindspore.ops.BinaryCrossEntropy()
#         #target_t = mindspore.Tensor(target_t, dtype=mindspore.float32)
#         target_t = target_t.astype(dtype=mindspore.float32)

#         loss = self.binary_cross_entropy(log_p, target_t, weight)
#         #loss = 0
#         return loss


# # def bce2d(input, target):
# #         n, c, h, w = input.shape
# #         # contigus没有用
# #         """contigus"""
# #         # input = mindspore.Tensor(input, dtype=mindspore.float32)
# #         print("1")
# #         input = input.astype(dtype=mindspore.float32)
# #         #target = mindspore.Tensor(target, dtype=mindspore.int32)
# #         input_perm = (0, 2, 3, 1)
# #         print("开始")
# #         transpose = mindspore.ops.Transpose()
# #         log_p = transpose(input, input_perm).view(1, -1)
# #         target_t = transpose(target, input_perm).view(1, -1)
# #         print("结束")
# #         #target_t = mindspore.Tensor(target_t, dtype=mindspore.float32)
# #         # target_trans = target_t.clone()
# #         target_t_new = target_t
# #         pos_index = (target_t_new == 1)
# #         neg_index = (target_t_new == 0)
# #         ignore_index = (target_t_new > 1)
# #
# #         # target_trans[pos_index] = 1
# #         # target_trans[neg_index] = 0
# #         #
# #         # pos_index = pos_index.asnumpy().astype(bool)
# #         # neg_index = neg_index.asnumpy().astype(bool)
# #         # ignore_index = ignore_index.asnumpy().astype(bool)
# #         #
# #         # pos_index = mindspore.Tensor(pos_index, dtype = mindspore.bool_)
# #         # neg_index = mindspore.Tensor(neg_index, dtype = mindspore.bool_)
# #         # ignore_index = mindspore.Tensor(ignore_index, dtype = mindspore.bool_)
# #
# #         fill = mindspore.ops.Fill()
# #         weight = fill(mindspore.float32, log_p.shape, 0)
# #
# #
# #         # weight = weight.asnumpy()
# #
# #         pos_index = pos_index.astype(mindspore.float32)
# #         neg_index = neg_index.astype(mindspore.float32)
# #         pos_num = pos_index.sum()
# #         neg_num = neg_index.sum()
# #         sum_num = pos_num + neg_num
# #         print(sum_num)
# #         pos_index = pos_index.astype(mindspore.int32)
# #         neg_index = neg_index.astype(mindspore.int32)
# #         print(pos_index)
# #         weight[pos_index] = neg_num * 1.0 / sum_num
# #         weight[neg_index] = pos_num * 1.0 / sum_num
# #
# #         weight[ignore_index] = 0
# #
# #         #weight = mindspore.Tensor.from_numpy(weight)
# #         #weight = None
# #         sigmoid = nn.Sigmoid()
# #         log_p = sigmoid(log_p)
# #         binary_cross_entropy = mindspore.ops.BinaryCrossEntropy()
# #         #target_t = mindspore.Tensor(target_t, dtype=mindspore.float32)
# #         target_t = target_t.astype(dtype=mindspore.float32)
# #
# #         loss = binary_cross_entropy(log_p, target_t, weight)
# #         #loss = 0
# #         return loss

# # loss1 = JointEdgeSegLoss_new(19)
# # randArray1 = np.random.randint(low= 0, high=1, size=(2, 19, 720, 720))
# # input1 = randArray1
# # input1 = mindspore.Tensor.from_numpy(input1)
# # input1 = mindspore.Tensor(input1,dtype=mindspore.float32)
# # randArray2 = np.random.randint(low= 0, high=18, size=(2,  19, 720, 720))
# # input2 = randArray2
# # input2 = mindspore.Tensor.from_numpy(input2)
# # input2 = mindspore.Tensor(input2,dtype=mindspore.float32)
# #
# # # inputs = mindspore.Tensor(np.ones((2, 3, 24, 24)).astype("float32"))
# # # targets = mindspore.Tensor(np.ones((2, 3, 24, 24)).astype("float32"))
# # # loss = bce2d(input1, input2)
# # loss = loss1(input1, input2)
# # print(loss)



# # class Net(nn.Cell):
# #     def __init__(self):
# #         super(Net, self).__init__()
# #         self.binary_cross_entropy = mindspore.ops.BinaryCrossEntropy()
# #     def construct(self, logits, labels, weight):
# #         result = self.binary_cross_entropy(logits, labels, weight)
# #         return result
# #
# # net = Net()
# # logits = mindspore.Tensor(np.array([0.2, 0.7, 0.1]), mindspore.float32)
# # labels = mindspore.Tensor(np.array([0., 1., 0.]), mindspore.float32)
# # weight = mindspore.Tensor(np.array([1, 2, 2]), mindspore.float32)
# # output = net(a, b, c)
# # print(output)

# loss = JointEdgeSegLoss(classes= 19)
# # input1 = np.load("output1.npy")
# # input2 = np.load("output2.npy")
# # mask = np.load("mask.npy")
# # edge = np.load("edgemap.npy")
# # input1 = mindspore.Tensor.from_numpy(input1)
# # input2 = mindspore.Tensor.from_numpy(input2)
# # mask = mindspore.Tensor.from_numpy(mask)
# # edge = mindspore.Tensor.from_numpy(edge)

# # input1 = np.load("segin.npy")
# # input2 = np.load("edgin.npy")
# # mask = np.load("segmask.npy")
# # edge = np.load("edgemask.npy")
# # input1 = mindspore.Tensor.from_numpy(input1)
# # input2 = mindspore.Tensor.from_numpy(input2)
# # mask = mindspore.Tensor.from_numpy(mask)
# # edge = mindspore.Tensor.from_numpy(edge)

# # np.random.seed(4800000)
# # input1 = mindspore.Tensor(np.random.randn(2, 19, 720, 720).astype("float32"))
# # input2 = mindspore.Tensor(np.random.randn(2, 1, 720, 720).astype("float32"))
# # mask = mindspore.Tensor(np.random.randint(0, 19, (2, 720, 720)).astype("int64"))
# # edge = mindspore.Tensor(np.random.randint(0, 19, (2,  1, 720, 720)).astype("float32"))

# # input1 = mindspore.Tensor(np.ones((2, 19, 720, 720)).astype("float32"))
# # input2 = mindspore.Tensor(np.ones((2, 1, 720, 720)).astype("float32"))
# # mask = mindspore.Tensor(np.ones((2, 720, 720)).astype("int64"))
# # edge = mindspore.Tensor(np.ones((2, 1, 720, 720)).astype("float32"))
# #
# # inputs = []
# # inputs.append(input1)
# # inputs.append(input2)
# # targets = []
# # targets.append(mask)
# # targets.append(edge)
# # out = loss(inputs, targets)
# # print(out)
# # x = mindspore.Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], mindspore.float32)
# # resize_bilinear = nn.ResizeBilinear()
# # result = resize_bilinear(x, size=(5, 5))
# # print("插值前")
# # print(x)
# #
# # print(result)
# #
# # print(result.shape)
# #
# # x = mindspore.Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], mindspore.float32)
# # resize_bilinear = mindspore.ops.ResizeBilinear((5, 5), align_corners=True)
# # output = resize_bilinear(x)
# # print(output)







# # def _sample_gumbel(shape, eps=1e-10):
# #     """
# #     Sample from Gumbel(0, 1)
# #
# #     based on
# #     https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
# #     (MIT license)
# #     """
# #     minval = mindspore.Tensor(0, mindspore.float32)
# #     maxval = mindspore.Tensor(1, mindspore.float32)
# #     U = mindspore.ops.uniform(shape, minval, maxval)
# #     print(U)
# #     log = mindspore.ops.Log()
# #     return - log(eps - log(U + eps))
# #
# #
# # out1234 = _sample_gumbel((1,2,3))
# # print(out1234)

# # input1234 = mindspore.Tensor([[[0.3739, 0.6466, 0.3380],
# #          [0.5271, 0.1576, 0.0796]]])
# # input1234 = mindspore.Tensor(input1234, dtype=mindspore.float32)
# # eps = 1e-10
# # out321 = - np_new.log(eps - np_new.log(input1234 + eps))
# # print(out321)

# def calculateWeights(target):
#     hist, _ = np_new.histogram(target.flatten(), bins = 19, range=((-1.0, 18.0)), density=True)
#     print(hist)
#     hist = ((hist != 0) * 1 * (1 - hist)) + 1
#     return hist
# ght  = mindspore.Tensor(np.ones((2, 3, 24, 24)).astype("float32"))*-1
# out89 = calculateWeights(ght)
# print(out89)


# # net = nn.Conv2d(3, 3, 1, has_bias=False, weight_init="HeUniform").set_train(True)
# # x = mindspore.Tensor(np.ones((1, 3, 1, 1)), mindspore.float32)
# # output = net(x)
# # # net2 = nn.Sigmoid()
# # # output = net2(output)
# # #print("-----------------------HeNormal-----------------------------")
# # #print(output)
# #
# # # x_in = x = mindspore.Tensor(np.array([[[[5.6e-8, 6.4e-6]]]]).astype(np.float32))
# # # net3 = nn.BatchNorm2d(1).set_train(True)
# # # x_in = net3(x_in)
# # # out_in = net2(x_in)
# # # print(out_in)
# #
# # net5 = nn.Dropout()

# # inputs = mindspore.Tensor(np.random.random((2, 4096, 12, 12)).astype("float32"))
# # # case 2: output_size=2
# # print(inputs.shape)
# # # adaptive_avg_pool_2d = mindspore.ops.AdaptiveAvgPool2D(1)
# # # output = adaptive_avg_pool_2d(inputs)
# # # print(output)
# #
# # pool = nn.AvgPool2d(kernel_size=12, stride=1)
# #
# # output = pool(inputs)
# # print(output)
# # print(output.shape)

# # pool = nn.AvgPool2d(kernel_size=3, stride=1)
# # x = mindspore.Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
# # output = pool(x)
# # print(output)
# # print(output.shape)

# class ImageBasedCrossEntropyLoss2d_new(nn.Cell):

#     def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
#                  norm=False, upper_bound=1.0):
#         super(ImageBasedCrossEntropyLoss2d_new, self).__init__()

#         self.num_classes = classes
#         """NLLLoss的一个参数维度没写"""
#         self.nll_loss = mindspore.ops.NLLLoss(reduction="mean")
#         self.norm = norm
#         self.upper_bound = upper_bound


#         self.log_softmax = nn.LogSoftmax()


#     def calculateWeights(self, target):

#         hist, _ = np_new.histogram(target.flatten(), bins = 19, range=((-1.0, 19.0)), density=True)
#         if self.norm:
#             hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
#         else:
#             hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
#         return hist

#     def construct(self, inputs, targets):


#         loss = 0.0
#         #loss_new = 0.0
#         targets_new = targets
#         transpose = mindspore.ops.Transpose()
#         inputs = transpose(inputs, (0, 2, 3, 1))
#         x_size = inputs.shape
#         for i in range(0, x_size[0]):

#             weights = self.calculateWeights(targets_new[i])

#             shape = (x_size[1] * x_size[2], x_size[3])
#             inputs_new = np_new.reshape(inputs[i], shape)
#             shape_new = (x_size[1] * x_size[2])
#             target_new = np_new.reshape(targets[i], shape_new)

#             #print(self.nll_loss(self.log_softmax(inputs_new), target_new, weights)[0])
#             #print(self.nll_loss(self.log_softmax(inputs_new), target_new, weights)[1])
#             loss += self.nll_loss(self.log_softmax(inputs_new), target_new, weights)[0]

#         return loss
# #
# # loss = ImageBasedCrossEntropyLoss2d_new(19)
# # input1 = mindspore.Tensor(np.ones((2, 19, 72, 72)).astype("float32"))
# #
# # mask = mindspore.Tensor(np.ones((2, 72, 72)))
# # mask = mask.astype(mindspore.int32)
# # mask =mask * 2
# #
# # output = loss(input1, mask)
# # print(output)

# milestone = []
# learning_rates = []
# for i in range(1, 175):
#     milestone.append(i)
#     learning_rates.append(math.pow((1 - i / 175), 1.0))
# lr = nn.piecewise_constant_lr(milestone, learning_rates)
# print(lr)