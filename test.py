import os
# import sys
# import logging

import math

from config import assert_and_infer_cfg
from datasetsnew.setup_data import setup_loaders

import argparse
#import yaml
import mindspore.nn as nn
from mindspore import Model, FixedLossScaleManager
from mindspore import context
# from mindspore import set_seed
# from mindspore.context import ParallelMode
# from mindspore.communication import init
# from mindspore.nn.optim import optimizer
# from mindspore.train.callback import CheckpointConfig
# from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import LossMonitor, TimeMonitor
# from mindspore.train.callback import TimeMonitor




# Argument Parser
from loss import JointEdgeSegLoss
from network.gscnn import GSCNN

#os.environ['CUDA_VISIBLE_DEVICES'] = '4'

context.set_context(device_id=7)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
parser = argparse.ArgumentParser(description='GSCNN')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.gscnn.GSCNN')
parser.add_argument('--dataset', type=str, default='cityscapes')
parser.add_argument('--cv', type=int, default=0,
                    help='cross validation split')
parser.add_argument('--joint_edgeseg_loss', action='store_true', default=True,
                    help='joint loss')
parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class')
parser.add_argument('--eval_thresholds', type=str, default='0.0005,0.001875,0.00375,0.005',
                    help='Thresholds for boundary evaluation')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Rescaled LR Rate')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Rescaled Poly')

parser.add_argument('--edge_weight', type=float, default=1.0,
                    help='Edge loss weight for joint loss')
parser.add_argument('--seg_weight', type=float, default=1.0,
                    help='Segmentation loss weight for joint loss')
parser.add_argument('--att_weight', type=float, default=1.0,
                    help='Attention loss weight for joint loss')
parser.add_argument('--dual_weight', type=float, default=1.0,
                    help='Dual loss weight for joint loss')

parser.add_argument('--evaluate', action='store_true', default=False)

parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--sgd_finetuned',action='store_true',default=False)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=175)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--rotate', type=float,
                    default=0, help='rotation')
parser.add_argument('--gblur', action='store_true', default=True)
parser.add_argument('--bblur', action='store_true', default=False)
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=1)
parser.add_argument('--bs_mult_val', type=int, default=2)
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt')
parser.add_argument('--tb_path', type=str, default='logs/tb')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Synchronized BN')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='minimum testing (1 epoch run ) to verify nothing failed')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0)
parser.add_argument('--maxSkip', type=int, default=0)
args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                        'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}


args.world_size = 1
#Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))


def train():
    assert_and_infer_cfg(args)


    train_dataset, val_dataset = setup_loaders(args)
    net = GSCNN(num_classes=args.dataset_cls.num_classes)
    # loss = JointEdgeSegLoss(classes=args.dataset_cls.num_classes, ignore_index=args.dataset_cls.ignore_label, upper_bound=args.wt_bound,
    #                         edge_weight=args.edge_weight, seg_weight=args.seg_weight,
    #                         att_weight=args.att_weight, dual_weight=args.dual_weight)
    """学习率"""
    milestone = []
    learning_rates = []
    for i in range(1, 175):
        milestone.append(i)
        learning_rates.append(math.pow((1 - i / 175), 1.0))
    lr = nn.piecewise_constant_lr(milestone, learning_rates)
    """优化器"""
    param_groups = net.trainable_params()

    optim = nn.SGD(param_groups, learning_rate=lr, weight_decay=args.weight_decay,
                        momentum=args.momentum, nesterov=False, loss_scale=1.0)
    #optim = nn.Momentum(params=net.trainable_params(), learning_rate=lr, momentum=0.9)

    cb = LossMonitor()
    time_cb = TimeMonitor()
    loss_scale = FixedLossScaleManager(1.0, drop_overflow_update=False)
    model = Model(net, optimizer=optim,  loss_scale_manager=loss_scale)
    model.train(epoch=args.max_epoch, train_dataset=train_dataset, callbacks=cb, dataset_sink_mode=False)


if __name__ == '__main__':

    train()
