import network.mynn as  mynn
import mindspore.nn as nn
import mindspore
from mindspore.common.initializer import HeNormal, Constant
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                      has_bias=False, pad_mode = "pad", padding=1,weight_init="HeNormal")

class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = mynn.Norm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = mynn.Norm2d(planes)
        #self.downsample = nn.SequentialCell()
        if downsample is not None:
             self.downsample = downsample
        else :
             self.downsample = nn.SequentialCell()
        self.stride = stride
        # for m in self.cells():
        #     if isinstance(m, nn.Conv2d):
        #         HeNormal(m.weight, mode='fan_out', nonlinearity='relu')
        #     """batchnorm需不需要权重初始化"""
            # elif isinstance(m, nn.BatchNorm2d):
            #     Constant(m.weight, 1)
            #     Constant(m.bias, 0)


    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


# inputs = mindspore.Tensor(np.ones((2, 64, 24, 24)).astype("float32"))
# resnet = BasicBlock(64, 64, stride=1, downsample=None)
# #print(resnet)
# outputs = resnet(inputs)
#print(outputs.shape)