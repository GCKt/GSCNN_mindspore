import mindspore.nn as nn
import mindspore.ops as ops
import mindspore
from mindspore import save_checkpoint
import time
import os
import time
import numpy as np
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops import composite as C
from mindspore import ParameterTuple
from mindspore.train.callback import Callback
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
class TrainOneStep(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStep, self).__init__(network, optimizer, sens)
        # self.network = network #定义前向网络
        # self.network.set_grad() #构建反向网络
        # self.optimizer = optimizer #定义优化器
        # self.weights = self.optimizer.parameters
        # self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        # self._loss = loss_fn


    def construct(self, inp, mask, edgemap, newcanny, label_panduan):
   

        loss = self.network(inp, mask, edgemap, newcanny, label_panduan)

        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        grads = self.grad(self.network, self.weights)(inp, mask, edgemap, newcanny, label_panduan,  sens)
        #print("tidu:", grads)
        grads = self.grad_reducer(grads)
        #print("ccccccccccccccccccccccccccccccccccccccccc")
        loss = ops.depend(loss, self.optimizer(grads))
        # total_loss /= j
        return loss


# class TrainOneStep(nn.Cell):
#     """
#     Network training package class.

#     Append an optimizer to the training network after that the construct function
#     can be called to create the backward graph.

#     Args:
#         network (Cell): The training network.
#         optimizer (Cell): Optimizer for updating the weights.
#         sens (Number): The adjust parameter. Default value is 1.0.
#         reduce_flag (bool): The reduce flag. Default value is False.
#         mean (bool): Allreduce method. Default value is False.
#         degree (int): Device number. Default value is None.
#     """
#     def __init__(self, network, optimizer, sens=1.0, reduce_flag=False, mean=True, degree=None):
#         super(TrainOneStep, self).__init__(auto_prefix=False)
#         self.network = network
#         self.network.set_grad()
#         self.weights = ParameterTuple(network.trainable_params())
#         self.optimizer = optimizer
#         self.grad = C.GradOperation(get_by_list=True,
#                                     sens_param=True)
#         self.sens = Tensor((np.ones((1,)) * sens).astype(np.float16))
#         self.reduce_flag = reduce_flag
#         self.hyper_map = C.HyperMap()
#         if reduce_flag:
#             self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

#     def construct(self, inp, mask, edgemap, newcanny, label_panduan):
#         weights = self.weights
#         loss = self.network(inp, mask, edgemap, newcanny, label_panduan)
#         grads = self.grad(self.network, weights)(inp, mask, edgemap, newcanny, label_panduan, self.sens)
#         if self.reduce_flag:
#             grads = self.grad_reducer(grads)
#         self.optimizer(grads)
#         return loss