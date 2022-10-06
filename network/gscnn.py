import mindspore

from mindspore import nn as nn, context

import network.Resnet as Resnet
from loss import JointEdgeSegLoss
from network.wider_resnet import WiderResNetA2
from config import cfg
from network.mynn import initialize_weights, Norm2d
import argparse

from my_functionals import GatedSpatialConv as gsc
from mindspore.ops.functional import stop_gradient
import cv2
import numpy as np


# parser = argparse.ArgumentParser(description='GSCNN')
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--arch', type=str, default='network.gscnn.GSCNN')
# parser.add_argument('--dataset', type=str, default='cityscapes')
# parser.add_argument('--cv', type=int, default=0,
#                     help='cross validation split')
# parser.add_argument('--joint_edgeseg_loss', action='store_true', default=True,
#                     help='joint loss')
# parser.add_argument('--img_wt_loss', action='store_true', default=False,
#                     help='per-image class-weighted loss')
# parser.add_argument('--batch_weighting', action='store_true', default=False,
#                     help='Batch weighting for class')
# parser.add_argument('--eval_thresholds', type=str, default='0.0005,0.001875,0.00375,0.005',
#                     help='Thresholds for boundary evaluation')
# parser.add_argument('--rescale', type=float, default=1.0,
#                     help='Rescaled LR Rate')
# parser.add_argument('--repoly', type=float, default=1.5,
#                     help='Rescaled Poly')
#
# parser.add_argument('--edge_weight', type=float, default=1.0,
#                     help='Edge loss weight for joint loss')
# parser.add_argument('--seg_weight', type=float, default=1.0,
#                     help='Segmentation loss weight for joint loss')
# parser.add_argument('--att_weight', type=float, default=1.0,
#                     help='Attention loss weight for joint loss')
# parser.add_argument('--dual_weight', type=float, default=1.0,
#                     help='Dual loss weight for joint loss')
#
# parser.add_argument('--evaluate', action='store_true', default=False)
#
# parser.add_argument("--local_rank", default=0, type=int)
#
# parser.add_argument('--sgd', action='store_true', default=True)
# parser.add_argument('--sgd_finetuned',action='store_true',default=False)
# parser.add_argument('--adam', action='store_true', default=False)
# parser.add_argument('--amsgrad', action='store_true', default=False)
#
# parser.add_argument('--trunk', type=str, default='resnet101',
#                     help='trunk model, can be: resnet101 (default), resnet50')
# parser.add_argument('--max_epoch', type=int, default=175)
# parser.add_argument('--start_epoch', type=int, default=0)
# parser.add_argument('--color_aug', type=float,
#                     default=0.25, help='level of color augmentation')
# parser.add_argument('--rotate', type=float,
#                     default=0, help='rotation')
# parser.add_argument('--gblur', action='store_true', default=True)
# parser.add_argument('--bblur', action='store_true', default=False)
# parser.add_argument('--lr_schedule', type=str, default='poly',
#                     help='name of lr schedule: poly')
# parser.add_argument('--poly_exp', type=float, default=1.0,
#                     help='polynomial LR exponent')
# parser.add_argument('--bs_mult', type=int, default=1)
# parser.add_argument('--bs_mult_val', type=int, default=2)
# parser.add_argument('--crop_size', type=int, default=720,
#                     help='training crop size')
# parser.add_argument('--pre_size', type=int, default=None,
#                     help='resize image shorter edge to this before augmentation')
# parser.add_argument('--scale_min', type=float, default=0.5,
#                     help='dynamically scale training images down to this size')
# parser.add_argument('--scale_max', type=float, default=2.0,
#                     help='dynamically scale training images up to this size')
# parser.add_argument('--weight_decay', type=float, default=1e-4)
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--snapshot', type=str, default=None)
# parser.add_argument('--restore_optimizer', action='store_true', default=False)
# parser.add_argument('--exp', type=str, default='default',
#                     help='experiment directory name')
# parser.add_argument('--tb_tag', type=str, default='',
#                     help='add tag to tb dir')
# parser.add_argument('--ckpt', type=str, default='logs/ckpt')
# parser.add_argument('--tb_path', type=str, default='logs/tb')
# parser.add_argument('--syncbn', action='store_true', default=True,
#                     help='Synchronized BN')
# parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
#                     help='Synchronized BN')
# parser.add_argument('--test_mode', action='store_true', default=False,
#                     help='minimum testing (1 epoch run ) to verify nothing failed')
# parser.add_argument('-wb', '--wt_bound', type=float, default=1.0)
# parser.add_argument('--maxSkip', type=int, default=0)
# args = parser.parse_args()
# args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
#                         'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}





class AtrousSpatialPyramidPoolingModule(nn.Cell):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.SequentialCell(nn.Conv2d(in_dim, reduction_dim, kernel_size=1,pad_mode = "pad",padding= 0, has_bias=False, weight_init="heuniform"),
                          Norm2d(reduction_dim), nn.ReLU()))
        # other rates
        for r in rates:
            self.features.append(nn.SequentialCell(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, has_bias=False, pad_mode = "pad", padding = r,weight_init="heuniform"),
                Norm2d(reduction_dim),
                nn.ReLU()
            ))
        self.features = mindspore.nn.CellList(self.features)

        # img level features
        #adaptive_avg_pool_2d = mindspore.ops.AdaptiveAvgPool2D(1)
        #self.img_pooling = mindspore.ops.AdaptiveAvgPool2D(1)
        self.img_pooling = nn.AvgPool2d(kernel_size=90, stride=90)
        # self.img_pooling = nn.MaxPool2d(kernel_size=3, stride=3, pad_mode="same")
        # self.img_pooling1 = nn.MaxPool2d(kernel_size=3, stride=5, pad_mode="same")
        # self.img_pooling2 = nn.MaxPool2d(kernel_size=3, stride=6, pad_mode="same")
        self.img_conv = nn.SequentialCell(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, pad_mode = "pad", padding = 0,has_bias=False, weight_init="heuniform"),
            Norm2d(reduction_dim), nn.ReLU())
        self.edge_conv = nn.SequentialCell(
            nn.Conv2d(1, reduction_dim, kernel_size=1, pad_mode = "pad",  padding = 0,has_bias=False, weight_init="heuniform"),
            Norm2d(reduction_dim), nn.ReLU())

    def construct(self, x, edge):
        x_size = x.shape
        '''x : torch.size([2, 4096, 12, 12 ])'''
        '''edge : torch.size([2, 1, 96, 96 ])'''

        img_features = self.img_pooling(x)
        #print(img_features.shape)
        # img_features = self.img_pooling1(img_features)
        #
        # img_features = self.img_pooling2(img_features)

        '''img_features : torch.size([2, 4096, 1, 1 ])'''
        img_features = self.img_conv(img_features)
        '''img_features : torch.size([2, 256, 1, 1 ])'''
        resize_bilinear = nn.ResizeBilinear()

        img_features = resize_bilinear(img_features, x_size[2:])
        '''img_features : torch.size([2, 256, 12, 12 ])'''
        out = img_features

        edge_features = resize_bilinear(edge, x_size[2:])
        '''edge_features : torch.size([2, 1, 12, 12 ])'''
        edge_features = self.edge_conv(edge_features)
        '''edge_features : torch.size([2, 256, 12, 12 ])'''
        concat_op = mindspore.ops.Concat(axis=1)

        out = concat_op((out, edge_features))
        '''out : torch.size([2, 512, 12, 12 ])'''

        for f in self.features:
            y = f(x)
            out = concat_op((out, y))
        return out


class GSCNN(nn.Cell):
    '''
    Wide_resnet version of DeepLabV3
    mod1
    pool2
    mod2 str2
    pool3
    mod3-7

      structure: [3, 3, 6, 3, 1, 1]
      channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                  (1024, 2048, 4096)]
    '''

    def __init__(self, num_classes, trunk=None, criterion=None):
        '''
        num_classes : 类别
        cirterion : 损失函数
        '''
        super(GSCNN, self).__init__()
        self.criterion = criterion
        self.num_classes = num_classes
        '''wide_resnet : 规则流的骨架'''
        wide_resnet = WiderResNetA2(structure=[3, 3, 6, 3, 1, 1], classes=1000, dilation=True)
        # wide_resnet = wider_resnet38_a2(classes=1000, dilation=True)
        # 调用多个GPU来进行训练
        #wide_resnet = torch.nn.DataParallel(wide_resnet)

        #wide_resnet = wide_resnet.cells()
        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        self.interpolate = mindspore.nn.ResizeBilinear()
        del wide_resnet

        self.dsn1 = nn.Conv2d(64, 1, 1, weight_init="heuniform",pad_mode = "pad", padding = 0,has_bias=True)
        self.dsn3 = nn.Conv2d(256, 1, 1, weight_init="heuniform", pad_mode = "pad", padding = 0,has_bias=True)
        self.dsn4 = nn.Conv2d(512, 1, 1, weight_init="heuniform", pad_mode = "pad", padding = 0,has_bias=True)
        self.dsn7 = nn.Conv2d(4096, 1, 1, weight_init="heuniform", pad_mode = "pad", padding = 0,has_bias=True)

        self.res1 = Resnet.BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1, weight_init="heuniform", pad_mode = "pad", padding = 0,has_bias=True)
        self.res2 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1, weight_init="heuniform", pad_mode = "pad", padding = 0,has_bias=True)
        self.res3 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1, weight_init="heuniform", pad_mode = "pad", padding = 0,has_bias=True)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, pad_mode = "pad", padding = 0,has_bias=False, weight_init="heuniform")

        self.cw = nn.Conv2d(2, 1, kernel_size=1, pad_mode = "pad", padding = 0,has_bias=False, weight_init="heuniform")

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)

        self.aspp = AtrousSpatialPyramidPoolingModule(4096, 256, output_stride=8)

        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, has_bias=False, pad_mode = "pad", padding = 0,weight_init="heuniform")
        self.bot_aspp = nn.Conv2d(1280 + 256, 256, kernel_size=1, has_bias=False, pad_mode = "pad", padding = 0,weight_init="heuniform")

        self.final_seg = nn.SequentialCell(
            nn.Conv2d(256 + 48, 256, kernel_size=3, has_bias=True, pad_mode = "pad", padding = 1,weight_init='HeNormal'),
            Norm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, has_bias=True, pad_mode = "pad", padding = 1,weight_init='HeNormal'),
            Norm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1, has_bias=True, pad_mode = "pad", padding = 1,weight_init='HeNormal'))

        # nn.Conv2d(256 + 48, 256, kernel_size=3, has_bias=False, weight_init='ones'),
        # Norm2d(256),
        # nn.ReLU(),
        # nn.Conv2d(256, 256, kernel_size=3, has_bias=False, weight_init='ones'),
        # Norm2d(256),
        # nn.ReLU(),
        # nn.Conv2d(256, num_classes, kernel_size=1, has_bias=False, weight_init='ones'))

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.PReLU()
        #initialize_weights(self.final_seg)

    def construct(self, inp, mask, edgemap, newcanny, label_panduan, gts=None):

        x_size = inp.shape
        #print("开始")
        '''x_size : torch.Size([2, 3, 96, 96])'''
        # res 1
        m1 = self.mod1(inp)
        #print("m1")
        '''m1 : torch.Size([2, 64, 96, 96])'''
        #res 2
        m2 = self.mod2(self.pool2(m1))
        #print("m2")
        '''m2 : torch.Size([2, 128, 48, 48])'''
        # res 3
        m3 = self.mod3(self.pool3(m2))
        '''m3 : torch.Size([2, 256, 24, 24])'''
        #print("m3")
        # m2 = self.mod2(m1)
        # '''m2 : torch.Size([2, 128, 48, 48])'''
        # # res 3
        # m3 = self.mod3(m2)
        # '''m3 : torch.Size([2, 256, 24, 24])'''

        # res 4-7
        m4 = self.mod4(m3)
        #print("m4")
        '''m4 : torch.Size([2, 512, 12, 12])'''
        m5 = self.mod5(m4)
        #print("m5")
        '''m5 : torch.Size([2, 1024, 12, 12])'''
        m6 = self.mod6(m5)
        '''m6 : torch.Size([2, 2048, 12, 12])'''
        m7 = self.mod7(m6)
        '''m7 : torch.Size([2, 4096, 12, 12])'''
        # 下采样F.interpolate
        resize_bilinear = nn.ResizeBilinear()
        dsn3 = self.dsn3(m3)
        s3 = resize_bilinear(self.dsn3(m3), x_size[2:], align_corners=True)
        '''s3 : torch.Size([2, 1, 96, 96])'''
        # l = self.dsn3(m3)
        s4 = resize_bilinear(self.dsn4(m4), x_size[2:], align_corners=True)
        '''s4 : torch.Size([2, 1, 96, 96])'''
        s7 = resize_bilinear(self.dsn7(m7), x_size[2:], align_corners=True)
        '''s7 : torch.Size([2, 1, 96, 96])'''

        m1f = resize_bilinear(m1, x_size[2:], align_corners=True)
        '''m1f : torch.Size([2, 64, 96, 96])'''
        #im_arr = inp.asnumpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        '''im_arr : torch.Size([2, 96, 96, 3])'''
        # canny = mindspore.ops.Zeros((x_size[0], 1, x_size[2], x_size[3]))
        # canny = canny.asnumpy()
        '''canny : torch.Size([2, 1, 96, 96])'''
        # canny获得边缘图像
        # cv2.Canny()方法可以获得图像的边缘图像
        canny = stop_gradient(newcanny)
        '''canny : torch.Size([2, 1, 96, 96])'''
        # canny = torch.from_numpy(canny).float()
        cs = self.res1(m1f)
        '''cs : torch.Size([2, 64, 96, 96])'''
        cs = resize_bilinear(cs, x_size[2:], align_corners=True)
        '''cs : torch.Size([2, 64, 96, 96])'''
        cs = self.d1(cs)
        '''cs : torch.Size([2, 32, 96, 96])'''
        cs = self.gate1(cs, s3)
        '''cs : torch.Size([2, 32, 96, 96])'''
        cs = self.res2(cs)
        '''cs : torch.Size([2, 32, 96, 96])'''
        cs = resize_bilinear(cs, x_size[2:], align_corners=True)
        '''cs : torch.Size([2, 32, 96, 96])'''
        cs = self.d2(cs)
        '''cs : torch.Size([2, 16, 96, 96])'''
        cs = self.gate2(cs, s4)
        '''cs : torch.Size([2, 16, 96, 96])'''
        cs = self.res3(cs)
        '''cs : torch.Size([2, 16, 96, 96])'''
        cs = resize_bilinear(cs, x_size[2:], align_corners=True)
        '''cs : torch.Size([2, 16, 96, 96])'''
        cs = self.d3(cs)
        '''cs : torch.Size([2, 8, 96, 96])'''
        cs = self.gate3(cs, s7)
        '''cs : torch.Size([2, 8, 96, 96])'''
        cs = self.fuse(cs)
        '''cs : torch.Size([2, 1, 96, 96])'''
        cs = resize_bilinear(cs, x_size[2:], align_corners=True)
        '''cs : torch.Size([2, 1, 96, 96])'''
        edge_out = self.sigmoid(cs)
        #edge_out = self.relu(cs)
        '''edge_out : torch.Size([2, 1, 96, 96])'''
        concat_op = mindspore.ops.Concat(axis=1)
        cat = concat_op((edge_out, canny))
        '''cat : torch.Size([2, 2, 96, 96])'''
        acts = self.cw(cat)
        '''acts : torch.Size([2, 1, 96, 96])'''
        acts = self.sigmoid(acts)
        '''acts : torch.Size([2, 1, 96, 96])'''
        # aspp
        x = self.aspp(m7, acts)
        '''x : torch.Size([2, 1536, 12, 12])'''

        dec0_up = self.bot_aspp(x)
        '''dec0_up : torch.Size([2, 256, 12, 12])'''
        dec0_fine = self.bot_fine(m2)
        '''dec0_fine : torch.Size([2, 48, 48, 48])'''
        dec0_up = self.interpolate(dec0_up, m2.shape[2:], align_corners=True)
        '''dec0_up : torch.Size([2, 256, 48, 48])'''
        # dec0 = [dec0_fine, dec0_up]
        #
        # dec0 = concat_op((dec0, 1))
        #print("文杰")
        dec0 = concat_op((dec0_fine, dec0_up))
        dec1 = self.final_seg(dec0)
        #print("天明")
        '''dec1 : torch.Size([2, 19, 48, 48])'''
        seg_out = self.interpolate(dec1, x_size[2:])
        '''seg_out : torch.Size([2, 19, 96, 96])'''

        #self.training = False
        # print(self.training)
        # if self.training:
        #     return self.criterion((seg_out, edge_out), gts)
        # else:
        loss = JointEdgeSegLoss(classes=19, ignore_index=255,
                                upper_bound=1.0,
                                edge_weight=1.0, seg_weight=1.0,
                                att_weight=1.0, dual_weight=1.0)

        #return seg_out, edge_out
        # mask = mindspore.Tensor(mask, dtype=mindspore.int32)
        # edgemap = mindspore.Tensor(edgemap, dtype=mindspore.int32)
        mask = stop_gradient(mask.astype(dtype=mindspore.int32))
        edgemap = stop_gradient(edgemap.astype(dtype=mindspore.int32))
        # mask = stop_gradient(mask)
        # edgemap = stop_gradient(edgemap)
        #print("网络结果")
        #print(seg_out.max())
        print("edge:",edge_out.max())
        print("edgemin:",edge_out.min())
        inputs = (seg_out, edge_out)
        targets = (mask, edgemap, label_panduan)
        #main_loss = loss(inputs, targets)
        # print("----------------------------------结束----------------------")
        #print(main_loss)
        #return main_loss
        return seg_out,edge_out


if __name__ == '__main__':
    # img = torch.randn(1, 3, 256, 256)
    # model = get_icnet_resnet50_citys()
    # outputs = model(img)

    # model = GSCNN(19).to(device)
    #context.set_context(mode = context.PYNATIVE_MODE, device_target = "GPU")
    # context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    model = GSCNN(19).set_train(True)
    print(model)
    inputs = mindspore.Tensor(np.ones((2, 3, 24, 24)).astype("float32"))
    im_arr = inputs.asnumpy().transpose((0, 2, 3, 1)).astype(np.uint8)
    '''im_arr : torch.Size([2, 96, 96, 3])'''
    x_size = inputs.shape
    canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
    '''canny : torch.Size([2, 1, 96, 96])'''
    # canny获得边缘图像
    # cv2.Canny()方法可以获得图像的边缘图像
    for i in range(x_size[0]):
        canny[i] = cv2.Canny(im_arr[i], 10, 100)
    canny = mindspore.Tensor.from_numpy(canny)
    canny = mindspore.Tensor(canny, dtype=mindspore.float32)
    new_canny = mindspore.Tensor(np.random.randn(2, 1, 24, 24).astype("float32"))
    output = model(inputs, canny)
    #print(output[0].shape)
    #print(output[1].shape)
    inputs = mindspore.Tensor(inputs, dtype=mindspore.double)

