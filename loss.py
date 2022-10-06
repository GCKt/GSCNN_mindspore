
import math
from config import cfg
import mindspore
from mindspore import nn as nn, context
import logging
import numpy as np
from my_functionals.SoftmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss
from mindspore import numpy as np_new

from my_functionals.DualTaskLoss import DualTaskLoss
#from my_functionals.DualTaskLoss import DualTaskLoss
from mindspore.ops import stop_gradient

def get_loss(args):
    '''
    Get the criterion based on the loss function
    args:
    return: criterion
    '''
    '''训练时传过来的参数img_wt_loss为False
    参数joint_edgeseg_loss为True'''
    if args.img_wt_loss:
        criterion = ImageBasedCrossEntropyLoss2d(
            classes=args.dataset_cls.num_classes, size_average=True,
            ignore_index=args.dataset_cls.ignore_label,
            upper_bound=args.wt_bound).cuda()
    elif args.joint_edgeseg_loss:
        criterion = JointEdgeSegLoss(classes=args.dataset_cls.num_classes,
                                     ignore_index=args.dataset_cls.ignore_label, upper_bound=args.wt_bound,
                                     edge_weight=args.edge_weight, seg_weight=args.seg_weight,
                                     att_weight=args.att_weight, dual_weight=args.dual_weight).cuda()

    else:
        criterion = mindspore.nn.SoftmaxCrossEntropyWithLogits().cuda()

    criterion_val = JointEdgeSegLoss(classes=args.dataset_cls.num_classes, mode='val',
                                     ignore_index=args.dataset_cls.ignore_label, upper_bound=args.wt_bound,
                                     edge_weight=args.edge_weight, seg_weight=args.seg_weight).cuda()

    return criterion, criterion_val


class JointEdgeSegLoss(nn.Cell):
    def __init__(self, classes, weight=None, reduction='mean', ignore_index=255,
                 norm=False, upper_bound=1.0, mode='train',
                 edge_weight=1, seg_weight=1, att_weight=1, dual_weight=1, edge='none'):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        
        self.seg_loss = ImageBasedCrossEntropyLoss2dNew(
                classes=classes, ignore_index=ignore_index, upper_bound=upper_bound)
        # self.seg_loss = SoftmaxCrossEntropyLoss(
        #         num_cls=classes, ignore_label=ignore_index)
        self.edge_loss =ImageBasedCrossEntropyLoss2d(
                classes=classes, ignore_index=ignore_index, upper_bound=upper_bound)
        
        
        self.new_loss = SoftmaxCrossEntropyLoss(num_cls=19, ignore_label=255)

        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.dual_weight = dual_weight

        self.dual_task = DualTaskLoss()
        self.input_perm = (0, 2, 3, 1)
        self.transpose = mindspore.ops.Transpose()
        self.fill = mindspore.ops.Fill()
        self.binary_cross_entropy = mindspore.ops.BinaryCrossEntropy()
        self.sigmoid = nn.Sigmoid()
        self.oneslike = mindspore.ops.OnesLike()
        self.prelu = nn.PReLU()
        self.nllloss = mindspore.ops.NLLLoss(reduction="mean")
        self.newloss = mindspore.ops.SoftmaxCrossEntropyWithLogits()
        self.ones =  mindspore.ops.Ones()
        self.zeros = mindspore.ops.Zeros()
        # self.newweight = mindspore.ops.Ones((1, 518400), mindspore.float32)
        # self.bce_loss = mindspore.nn.BCEWithLogitsLoss(weight = self.newweight)

    def bce2d(self, input, target):
        n, c, h, w = input.shape
        # contigus没有用
        """contigus"""
        # input = mindspore.Tensor(input, dtype=mindspore.float32)

        input = input.astype(dtype=mindspore.float32)
        # target = stop_gradient(target)
        # input = stop_gradient(input)
        #target = mindspore.Tensor(target, dtype=mindspore.int32)
        #input_perm = (0, 2, 3, 1)

        #transpose = mindspore.ops.Transpose()
        log_p = self.transpose(input, self.input_perm).view(1, -1)
        target_t = self.transpose(target, self.input_perm).view(1, -1)
        #print("log_p.shape", log_p.shape)
        # log_p = input
        # target_t = target

        #target_t = mindspore.Tensor(target_t, dtype=mindspore.float32)
        # target_trans = target_t.clone()
        target_t_new = target_t
        pos_index = (target_t_new == 1)
        neg_index = (target_t_new == 0)
        ignore_index = (target_t_new > 1)

        
        weight = self.fill(mindspore.float32, log_p.shape, 0.0)
        #weight = self.ones((1, 518400), mindspore.float32)

        

        pos_index = pos_index.astype(mindspore.float32)
        neg_index = neg_index.astype(mindspore.float32)
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num

        

        
        filler1 = self.oneslike(weight) * neg_num * 1.0 / sum_num
        filler2 = self.oneslike(weight) * pos_num * 1.0 / sum_num
        filler3 = self.oneslike(weight) * 0.0
        
        pos_index = pos_index.astype(mindspore.bool_)
        neg_index = neg_index.astype(mindspore.bool_)
        ignore_index = ignore_index.astype(mindspore.bool_)
        weight = np_new.where(pos_index, filler1, weight)
        weight = np_new.where(neg_index, filler2, weight)
        weight = np_new.where(ignore_index, filler3, weight)
        #print("leixing",filler1.dtype)
        # weight.masked_fill(pos_index, filler1)
        # weight.masked_fill(neg_index, filler2)
        # weight.masked_fill(ignore_index, 0.0)
        #weight = None
       
        
        #log_p = self.prelu(log_p) 
        log_p = self.sigmoid(log_p)
        
        target_t = target_t.astype(dtype=mindspore.float32)
        #target_t = target_t.astype(dtype=mindspore.int32)
        loss = self.binary_cross_entropy(log_p, target_t,weight)
        
        # print(self.bce_loss.weight)
        # print(self.bce_loss.weight.shape)
        # loss = self.bce_loss(log_p, target_t)
        #loss = self.newloss(log_p, target_t)[0]
       

        return loss

    def edge_attention(self, input, target, edge, label_panduan):
        n, c, h, w = input.shape
        # 返回一个填充了标量值1的张量，其大小与之相同 input。torch.ones_like(input)相当于 。torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)
        oneslike = mindspore.ops.OnesLike()
        filler = oneslike(target) * 19
        
        label_panduan = label_panduan.astype(dtype=mindspore.int32)
        newtarget = np_new.where(edge.max(1)[0] > 0.8, target, filler)
        #newtarget = np_new.where(label_panduan, newtarget, label_panduan)
        
        newtarget = newtarget.astype(dtype=mindspore.int32)
        return self.edge_loss(input, newtarget)

    def construct(self, inputs, targets):
        segin, edgein = inputs
        segmask, edgemask, labelpanduan= targets


       
        segmask = segmask.astype(dtype=mindspore.int32)
        edgemask = edgemask.astype(dtype=mindspore.int32)

        losses = {}

        losses['seg_loss'] = self.seg_weight * self.seg_loss(segin, segmask)
        
        
        print("seg_loss",losses['seg_loss'])
        losses['edge_loss'] = self.edge_weight  * self.bce2d(edgein, edgemask)*20
        print("edge_loss",losses['edge_loss'])
        losses['att_loss'] = self.att_weight * self.edge_attention(segin, segmask, edgein, labelpanduan)
        print("att_loss",losses['att_loss'])
        # #print(losses)
        losses['dual_loss'] = self.dual_weight * self.dual_task(segin, segmask)
        print("dual_loss",losses['dual_loss'])
        # print(losses)
        return        losses['seg_loss'] + losses['edge_loss'] + losses['att_loss'] + losses['dual_loss']
        #main_loss = losses['seg_loss'] 
        #main_loss =  self.bce2d(edgein, edgemask)
        #main_loss = main_loss.mean()
        #print(losses)
        #return main_loss
        #return losses['seg_loss']



        


# Img Weighted Loss
class ImageBasedCrossEntropyLoss2d(nn.Cell):

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        """NLLLoss的一个参数维度没写"""
        self.nll_loss = mindspore.ops.NLLLoss(reduction="mean")
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING
        # 不确定axis的值
        self.log_softmax = nn.LogSoftmax()
        self.cross_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        #self.weights = weight

    def calculateWeights(self, target):

        hist, _ = np_new.histogram(target.flatten(), bins = 19, range=((0.0, 18.0)), density=True)
        # print("20edge_loss")
        # print(hist)
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def construct(self, inputs, targets):
        loss = 0.0
        #loss_new = 0.0
        targets_new = targets
        transpose = mindspore.ops.Transpose()
        concat = mindspore.ops.Concat(axis=1)
        #newnum = np_new.zeros((1,1, 720, 720))
        #inputs = concat((inputs,newnum))
        inputs = transpose(inputs, (0, 2, 3, 1))
        x_size = inputs.shape
        #print(x_size)
        for i in range(0, x_size[0]):
            
            weights = self.calculateWeights(targets_new[i])
            
            num = np_new.ones((1))*1e-20
            newconcat = mindspore.ops.Concat(axis=0)
            newweight = newconcat((weights,num))
            #newweight = np_new.ones((20))
            shape = (x_size[1] * x_size[2], x_size[3])
            inputs_new = np_new.reshape(inputs[i], shape)
            
            shape_new = (x_size[1] * x_size[2])
            target_new = np_new.reshape(targets[i], shape_new)
            
            #print(newweight)
            #self.nll_loss.weight = weights
            #print(self.nll_loss.weight)
            #print(self.nll_loss(self.log_softmax(inputs_new), target_new))
            #print(self.nll_loss(self.log_softmax(inputs_new), target_new, weights)[1])
            inputs_new = self.log_softmax(inputs_new)
            newnum = np_new.zeros((720*720, 1))
            inputs_new = concat((inputs_new,newnum))
            loss += self.nll_loss(inputs_new, target_new, newweight)[0] 


        return loss


class ImageBasedCrossEntropyLoss2dNew(nn.Cell):

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2dNew, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        """NLLLoss的一个参数维度没写"""
        self.nll_loss = mindspore.ops.NLLLoss(reduction="mean")
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING
        self.cross_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="sum")
        # 不确定axis的值
        self.log_softmax = nn.LogSoftmax()
        #self.weights = weight

    def calculateWeights(self, target):

        hist, _ = np_new.histogram(target.flatten(), bins = 19, range=((0.0, 19.0)), density=True)
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def construct(self, inputs, targets):

        if self.batch_weights:
            weights = self.calculateWeights(targets)
            #self.weights = weights
        loss = 0.0
        # loss_new = 0.0
        targets_new = targets
        transpose = mindspore.ops.Transpose()
        inputs = transpose(inputs, (0, 2, 3, 1))
        x_size = inputs.shape
        for i in range(0, x_size[0]):
                weights = self.calculateWeights(targets_new[i])

                shape = (x_size[1] * x_size[2], x_size[3])
                inputs_new = np_new.reshape(inputs[i], shape)
                shape_new = (x_size[1] * x_size[2])
                target_new = np_new.reshape(targets[i], shape_new)

                s = self.log_softmax(inputs_new)
                #print("权重:",weights)
                #print(self.nll_loss(self.log_softmax(inputs_new), target_new, weights)[0])
                loss += self.nll_loss(self.log_softmax(inputs_new), target_new, weights)[0]
        
        return loss


# class NewLoss(nn.Cell):

#     def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
#                  norm=False, upper_bound=1.0):
#         super(NewLoss, self).__init__()
#         logging.info("Using Per Image based weighted loss")
#         self.num_classes = classes
#         """NLLLoss的一个参数维度没写"""
#         self.nll_loss = mindspore.ops.NLLLoss(reduction="mean")
#         self.norm = norm
#         self.upper_bound = upper_bound
#         self.batch_weights = cfg.BATCH_WEIGHTING
#         self.cross_loss = SoftmaxCrossEntropyLoss()
#         # 不确定axis的值
#         self.log_softmax = nn.LogSoftmax()
#         #self.weights = weight

#     def calculateWeights(self, target):

#         hist, _ = np_new.histogram(target.flatten(), bins = 19, range=((0.0, 19.0)), density=True)
#         if self.norm:
#             hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
#         else:
#             hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
#         return hist

#     def construct(self, inputs, targets):

        
#         loss = 0.0
#         # loss_new = 0.0
#         targets_new = targets
#         transpose = mindspore.ops.Transpose()
#         inputs = transpose(inputs, (0, 2, 3, 1))
#         x_size = inputs.shape
#         for i in range(0, x_size[0]):
               

#             shape = (x_size[1] * x_size[2], x_size[3])
#             inputs_new = np_new.reshape(inputs[i], shape)
#             shape_new = (x_size[1] * x_size[2])
#             target_new = np_new.reshape(targets[i], shape_new)

#             s = self.log_softmax(inputs_new)
            
                
#             loss += self.cross_loss(inputs_new, target_new)[0]
        
#         return loss




# Cross Entroply NLL Loss
class CrossEntropyLoss2d(nn.Cell):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = mindspore.ops.NLLLoss(reduction="mean")
        self.log_softmax = nn.LogSoftmax()
        self.weight = weight
    def construct(self, inputs, targets):
        return self.nll_loss(self.log_softmax(inputs), targets, self.weight)


#context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
# input = mindspore.Tensor(np.random.randn(3, 3), mindspore.float32)
# labels = mindspore.Tensor([1, 0, 1], mindspore.int32)
#
# loss3 = ImageBasedCrossEntropyLoss2d(3)
#
# print(loss3(input, labels))

# input1 = np.load("output1.npy")
# mask = np.load("mask.npy")
# input1 = mindspore.Tensor.from_numpy(input1)
# mask = mindspore.Tensor.from_numpy(mask)
# mask = mindspore.Tensor(mask, dtype=mindspore.int32)
#loss3 = ImageBasedCrossEntropyLoss2d(19)
#print(loss3(input1, mask))

# a = np.random.rand(100)
# print(a)
# hist,bins = np.histogram(a ,range=(5,9))
# print(hist)
# print(bins)

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""loss unit"""
from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P


class SoftmaxCrossEntropyLoss(nn.Cell):
    """SoftmaxCrossEntropyLoss"""
    def __init__(self, num_cls=19, ignore_label=-1):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        """construct"""
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss



