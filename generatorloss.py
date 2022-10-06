import mindspore.nn  as nn
from my_functionals.SoftmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss
from loss import JointEdgeSegLoss
import mindspore

class Generatorloss(nn.Cell):
    def __init__(self, generator):
        super(Generatorloss, self).__init__()
        # 下面定义需要用到的loss
        self.generator = generator
        self.my_loss = loss = JointEdgeSegLoss(classes=19, ignore_index=255,
                                           upper_bound=1.0,
                         edge_weight=1.0, seg_weight=1.0,
                         att_weight=1.0, dual_weight=1.0)
        #self.my_loss  = SoftmaxCrossEntropyLoss(num_cls = 19 ,ignore_label=255)




    def construct(self, inp, mask, edgemap, newcanny, label_panduan):
        # 下面是计算loss的流程
        # prediction = self.generator(maxture)
        seg_out, edge_out= self.generator(inp, mask, edgemap, newcanny, label_panduan)
        mask = mask.astype(dtype=mindspore.int32)
        edgemap = edgemap.astype(dtype=mindspore.int32)
        inputs = (seg_out, edge_out)
        targets = (mask, edgemap, label_panduan)
        #print("网络结果")
        #print("seg_out",seg_out.max())
        #print("edge_out",edge_out.max())
        loss = self.my_loss(inputs, targets)
        # print("8888888888888888888888888888888888888888")
        return loss