import mindspore.nn as nn
import mindspore
from mindspore.ops import Conv2D, Concat
from mindspore._checkparam import twice
from mindspore.nn.layer.conv import _Conv

import numpy as np
import math
import network.mynn as mynn
import my_functionals.custom_functional as myF


class GatedSpatialConv2d(_Conv):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, has_bias=False,
                 bias_init='zeros', data_format='NCHW'):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """

        kernel_size = twice(kernel_size)
        stride = twice(stride)
        padding = twice(padding)
        dilation = twice(dilation)
        """权重的初始化"""
        weight_shape = (out_channels, in_channels, 1, 1)
        weight = mindspore.Tensor(self.xavier_normal(weight_shape))
        
        super(GatedSpatialConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode='same',
            padding=padding,
            dilation=dilation,
            group=groups,
            has_bias=has_bias,
            #weight_init=weight,
            weight_init='xavieruniform',
            #weight_init="ones",
            bias_init=bias_init,
            data_format='NCHW',
            transposed=False,
        )

        self._gate_conv = nn.SequentialCell(
            mynn.Norm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1,pad_mode = "pad",padding=0, weight_init='HeUniform'),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1, pad_mode = "pad", padding = 0, weight_init='HeUniform'),
            mynn.Norm2d(1),
            nn.Sigmoid()
        )

        #print(self.weight==weight)



    def construct(self, input_features, gating_features):
        """

        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        concat_op = Concat(axis=1)
        
        alphas = self._gate_conv(concat_op((input_features, gating_features)))
        #alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

      
        #print("alphas",alphas)
        input_features = (input_features * (alphas + 1))
        """Conv2D相当于F.conv2d"""
        fconv2d = Conv2D(self.out_channels, kernel_size=1, stride=self.stride,
                         dilation=self.dilation, pad_mode = "pad", pad = 0,group=self.group)

        return fconv2d(input_features, self.weight)



    def extend_repr(self):
        s = 'input_channels={}, output_channels={}, kernel_size={}, ' \
            'stride={}, pad_mode={}, padding={}, dilation={}, ' \
            'group={}, has_bias={}, ' \
            'weight_init={}, bias_init={}, format={}'.format(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.pad_mode,
                self.padding,
                self.dilation,
                self.group,
                self.has_bias,
                self.weight_init,
                self.bias_init,
                self.format)
        return s

    def xavier_normal(self, inputs_shape, gain=1.):
        def _calculate_fan_in_and_fan_out(tensor):
            """
            _calculate_fan_in_and_fan_out
            """
            dimensions = len(tensor)

            if dimensions < 2:
                raise ValueError("Fan in and fan out can not be computed for tensor"
                                 " with fewer than 2 dimensions")
            if dimensions == 2:  # Linear
                fan_in = tensor[1]
                fan_out = tensor[0]
            else:
                num_input_fmaps = tensor[1]
                num_output_fmaps = tensor[0]
                receptive_field_size = 1
                if dimensions > 2:
                    receptive_field_size = tensor[2] * tensor[3]
                fan_in = num_input_fmaps * receptive_field_size
                fan_out = num_output_fmaps * receptive_field_size

            return fan_in, fan_out

        fan_in, fan_out = _calculate_fan_in_and_fan_out(inputs_shape)

        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        #std = gain / math.sqrt(fan_in + fan_out)
        weight = np.random.normal(0, std, size=inputs_shape).astype(np.float32)

        return weight

# net = GatedSpatialConv2d(64, 64)
# cs = mindspore.Tensor(np.ones((2, 64, 24, 24)).astype("float32"))
# s3 = mindspore.Tensor(np.ones((2, 1, 24, 24)).astype("float32"))
# output = net(cs, s3)
#print(output.shape)


