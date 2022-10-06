import mindspore
import mindspore.nn as nn
from mindspore import context
import mindspore.ops as ops
import network.mynn as mynn
import sys
from collections import OrderedDict
from functools import partial
import numpy as np


def bnrelu(channels):
    '''ReLu没有torch中的inpace属性，没有大问题，只是计算会变慢，占用多一点的内存'''
    return nn.SequentialCell(mynn.Norm2d(channels),
                         nn.ReLU())
    # return nn.SequentialCell(nn.BatchNorm2d(channels),
    #                      nn.ReLU())


class GlobalAvgPool2d(nn.Cell):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def construct(self, inputs):
        in_size = inputs.shape
        return inputs.view((in_size[0], in_size[1], -1)).mean(axis=2)



class IdentityResidualBlock(nn.Cell):

    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_act=bnrelu,
                 dropout=None,
                 dist_bn=False
                 ):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps.
            Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions,
            otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups.
            This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        dist_bn: Boolean
            A variable to enable or disable use of distributed BN
        """
        super(IdentityResidualBlock, self).__init__()
        self.dist_bn = dist_bn

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn


        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                ("conv1", nn.Conv2d(in_channels,
                                    channels[0],
                                    3,
                                    stride=stride,
                                    weight_init="heuniform",
                                    pad_mode = "pad",
                                    padding = dilation,
                                    has_bias=False,
                                    dilation=dilation)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1],
                                    3,
                                    stride=1,
                                    pad_mode = "pad",
                                    padding = dilation,
                                    weight_init="heuniform",
                                    has_bias=False,
                                    dilation=dilation))
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                ("conv1",
                 nn.Conv2d(in_channels,
                           channels[0],
                           1,
                           stride=stride,
                           weight_init="heuniform",
                           pad_mode = "pad",
                           has_bias=False)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0],
                                    channels[1],
                                    3, stride=1,
                                    weight_init="heuniform",
                                    pad_mode = "pad",
                                    padding = dilation,
                                    has_bias=False,
                                    group=groups,
                                    dilation=dilation)),
                ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2],
                                    1, stride=1, weight_init="heuniform", pad_mode = "pad", padding = 0,has_bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.SequentialCell(OrderedDict(layers))
        self.proj_conv = nn.SequentialCell()
        if need_proj_conv:
            self.proj_conv = nn.SequentialCell(
                [norm_act(in_channels),
                nn.Conv2d(
                in_channels, channels[-1], 1, stride=stride, has_bias=False, pad_mode = "pad", padding = 0, weight_init="heuniform")])
            # self.proj_conv = nn.SequentialCell(
            #     [nn.Conv2d(in_channels, channels[-1], 1, stride=stride, has_bias=False, weight_init="ones")])
    def construct(self, x):
        """
        This is the standard forward function for non-distributed batch norm
        """
        identity = x

        x = self.bn1(x)
        shortcut = self.proj_conv(identity)
        #shortcut = self.proj_conv(x)
        #print("天明")
        out = self.convs(x)
        out = out + shortcut
        return out


class WiderResNetA2(nn.Cell):

    def __init__(self,
                 structure,
                 norm_act=bnrelu,
                 classes=0,
                 dilation=False,
                 dist_bn=False
                 ):
        """Wider ResNet with pre-activation (identity mapping) blocks

        This variant uses down-sampling by max-pooling in the first two blocks and \
         by strided convolution in the others.

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer
            \with `classes` outputs at the end
            of the network.
        dilation : bool
            If `True` apply dilation to the last three modules and change the
            \down-sampling factor from 32 to 8.
        """
        super(WiderResNetA2, self).__init__()
        self.dist_bn = dist_bn

        # If using distributed batch norm, use the encoding.nn as oppose to torch.nn


        #nn.Dropout = nn.Dropout2d
        '''bnrelu : 自己定义的norm和relu'''
        #norm_act = bnrelu
        self.structure = structure
        self.dilation = dilation

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        # Initial layers
        self.mod1 = nn.SequentialCell(OrderedDict([
            ("conv1", nn.Conv2d(3, 64, 3, stride=1,  has_bias=False, pad_mode = "pad", padding = 1,weight_init="heuniform"))
        ]))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                    (1024, 2048, 4096)]
        mod_id = 0
        num = structure[0]
        blocks = []
        for block_id in range(num):
            if not dilation:
                dil = 1
                stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
            else:
                if mod_id == 3:
                    dil = 2
                elif mod_id > 3:
                    dil = 4
                else:
                    dil = 1
                stride = 2 if block_id == 0 and mod_id == 2 else 1

            if mod_id == 4:
                drop = partial(nn.Dropout, keep_prob=0.3)
            elif mod_id == 5:
                drop = partial(nn.Dropout, keep_prob=0.5)
            else:
                drop = None

            blocks.append((
                "block%d" % (block_id + 1),
                IdentityResidualBlock(in_channels,
                                      channels[mod_id], norm_act=norm_act,
                                      stride=stride, dilation=dil,
                                      dropout=drop, dist_bn=self.dist_bn)
            ))

            # Update channels and p_keep
            in_channels = channels[mod_id][-1]

        self.mod2 = (nn.SequentialCell(OrderedDict(blocks)))
        self.pool2 = nn.MaxPool2d(3, stride=2, pad_mode = "same")

        mod_id = 1
        num = structure[1]
        blocks = []
        for block_id in range(num):
            if not dilation:
                dil = 1
                stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
            else:
                if mod_id == 3:
                    dil = 2
                elif mod_id > 3:
                    dil = 4
                else:
                    dil = 1
                stride = 2 if block_id == 0 and mod_id == 2 else 1

            if mod_id == 4:
                drop = partial(nn.Dropout, keep_prob=0.3)
            elif mod_id == 5:
                drop = partial(nn.Dropout, keep_prob=0.5)
            else:
                drop = None

            blocks.append((
                "block%d" % (block_id + 1),
                IdentityResidualBlock(in_channels,
                                      channels[mod_id], norm_act=norm_act,
                                      stride=stride, dilation=dil,
                                      dropout=drop, dist_bn=self.dist_bn)
            ))

            # Update channels and p_keep
            in_channels = channels[mod_id][-1]

        self.mod3 = (nn.SequentialCell(OrderedDict(blocks)))
        self.pool3 = nn.MaxPool2d(3, stride=2, pad_mode = "same")

        mod_id = 2
        num = structure[2]
        blocks = []
        for block_id in range(num):
            if not dilation:
                dil = 1
                stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
            else:
                if mod_id == 3:
                    dil = 2
                elif mod_id > 3:
                    dil = 4
                else:
                    dil = 1
                stride = 2 if block_id == 0 and mod_id == 2 else 1

            if mod_id == 4:
                drop = partial(nn.Dropout, keep_prob=0.3)
            elif mod_id == 5:
                drop = partial(nn.Dropout, keep_prob=0.5)
            else:
                drop = None

            blocks.append((
                "block%d" % (block_id + 1),
                IdentityResidualBlock(in_channels,
                                      channels[mod_id], norm_act=norm_act,
                                      stride=stride, dilation=dil,
                                      dropout=drop, dist_bn=self.dist_bn)
            ))

            # Update channels and p_keep
            in_channels = channels[mod_id][-1]

        self.mod4 = (nn.SequentialCell(OrderedDict(blocks)))

        mod_id = 3
        num = structure[3]
        blocks = []
        for block_id in range(num):
            if not dilation:
                dil = 1
                stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
            else:
                if mod_id == 3:
                    dil = 2
                elif mod_id > 3:
                    dil = 4
                else:
                    dil = 1
                stride = 2 if block_id == 0 and mod_id == 2 else 1

            if mod_id == 4:
                drop = partial(nn.Dropout, keep_prob=0.3)
            elif mod_id == 5:
                drop = partial(nn.Dropout, keep_prob=0.5)
            else:
                drop = None

            blocks.append((
                "block%d" % (block_id + 1),
                IdentityResidualBlock(in_channels,
                                      channels[mod_id], norm_act=norm_act,
                                      stride=stride, dilation=dil,
                                      dropout=drop, dist_bn=self.dist_bn)
            ))

            # Update channels and p_keep
            in_channels = channels[mod_id][-1]

        self.mod5 = (nn.SequentialCell(OrderedDict(blocks)))

        mod_id = 4
        num = structure[4]
        blocks = []
        for block_id in range(num):
            if not dilation:
                dil = 1
                stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
            else:
                if mod_id == 3:
                    dil = 2
                elif mod_id > 3:
                    dil = 4
                else:
                    dil = 1
                stride = 2 if block_id == 0 and mod_id == 2 else 1

            if mod_id == 4:
                drop = partial(nn.Dropout, keep_prob=0.3)
            elif mod_id == 5:
                drop = partial(nn.Dropout, keep_prob=0.5)
            else:
                drop = None

            blocks.append((
                "block%d" % (block_id + 1),
                IdentityResidualBlock(in_channels,
                                      channels[mod_id], norm_act=norm_act,
                                      stride=stride, dilation=dil,
                                      dropout=drop, dist_bn=self.dist_bn)
            ))

            # Update channels and p_keep
            in_channels = channels[mod_id][-1]

        self.mod6 = (nn.SequentialCell(OrderedDict(blocks)))

        mod_id = 5
        num = structure[5]
        blocks = []
        for block_id in range(num):
            if not dilation:
                dil = 1
                stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
            else:
                if mod_id == 3:
                    dil = 2
                elif mod_id > 3:
                    dil = 4
                else:
                    dil = 1
                stride = 2 if block_id == 0 and mod_id == 2 else 1

            if mod_id == 4:
                drop = partial(nn.Dropout, keep_prob=0.3)
            elif mod_id == 5:
                drop = partial(nn.Dropout, keep_prob=0.5)
            else:
                drop = None

            blocks.append((
                "block%d" % (block_id + 1),
                IdentityResidualBlock(in_channels,
                                      channels[mod_id], norm_act=norm_act,
                                      stride=stride, dilation=dil,
                                      dropout=drop, dist_bn=self.dist_bn)
            ))

            # Update channels and p_keep
            in_channels = channels[mod_id][-1]

        self.mod7 = (nn.SequentialCell(OrderedDict(blocks)))
        # for mod_id, num in enumerate(structure):
        #     # Create blocks for module
        #     blocks = []
        #     for block_id in range(num):
        #         if not dilation:
        #             dil = 1
        #             stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
        #         else:
        #             if mod_id == 3:
        #                 dil = 2
        #             elif mod_id > 3:
        #                 dil = 4
        #             else:
        #                 dil = 1
        #             stride = 2 if block_id == 0 and mod_id == 2 else 1
        #
        #         if mod_id == 4:
        #             drop = partial(nn.Dropout, keep_prob=0.3)
        #         elif mod_id == 5:
        #             drop = partial(nn.Dropout, keep_prob=0.5)
        #         else:
        #             drop = None
        #
        #         blocks.append((
        #             "block%d" % (block_id + 1),
        #             IdentityResidualBlock(in_channels,
        #                                   channels[mod_id], norm_act=norm_act,
        #                                   stride=stride, dilation=dil,
        #                                   dropout=drop, dist_bn=self.dist_bn)
        #         ))
        #
        #         # Update channels and p_keep
        #         in_channels = channels[mod_id][-1]
        #
        #     # Create module
        #     if mod_id < 2:
        #         if mod_id == 1:
        #
        #             self.mod3=(nn.SequentialCell(OrderedDict(blocks)))
        #             self.pool3=nn.MaxPool2d(3, stride=2)
        #     self.insert_child_to_cell("mod%d" % (mod_id + 2), nn.SequentialCell(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        self.classifier = nn.SequentialCell()
        if classes != 0:
            self.classifier = nn.SequentialCell(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Dense(in_channels, classes))
            ]))

    def construct(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(out)
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        out = self.bn_out(out)

        print("out:",out)
        out = self.classifier(out)

        return out



# inputs = mindspore.Tensor(np.ones((2, 3, 24, 24)).astype("float32"))
# wide_resnet = WiderResNetA2(structure = [3, 3, 6, 3, 1, 1],classes=1000, dilation=True)
# #print(wide_resnet)
# outputs = wide_resnet(inputs)
# #print(outputs.shape)