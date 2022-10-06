import os
# import sys
# import logging

import math

from config import assert_and_infer_cfg
from datasetsnew.setup_data import setup_loaders
from mindspore import Model, FixedLossScaleManager, load_checkpoint, load_param_into_net
from mindspore import save_checkpoint, context, load_checkpoint, load_param_into_net
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
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
# from mindspore.train.callback import TimeMonitor
from mindspore.common import set_seed
from trainonestep import TrainOneStep
from generatorloss import Generatorloss


# Argument Parser
from loss import JointEdgeSegLoss
from network.gscnn import GSCNN

#os.environ['CUDA_VISIBLE_DEVICES'] = '4'

context.set_context(device_id=1)
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
#context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
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
parser.add_argument('--max_epoch', type=int, default=225)
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
parser.add_argument('--model_type', action='store_true', default="GSCNN")
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Synchronized BN')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='minimum testing (1 epoch run ) to verify nothing failed')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0)
parser.add_argument('--maxSkip', type=int, default=0)
parser.add_argument('--save_folder', default='/home/heu_MEDAI/gongcheng/New19Channel/xiaocheckpoint',
                    help='Location to save epoch models')
parser.add_argument('--model_path', type=str,
                    default="/home/heu_MEDAI/gongcheng/19Channel/xiaocheckpoint/GSCNN_ckpt_57-23_95.ckpt",
                    help='Path to model file created by training')
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

epoch_loss = []

assert_and_infer_cfg(args)

def train(trainoneStep, data):
    trainoneStep.set_train()
    trainoneStep.set_grad()
    
    steps = data.get_dataset_size()

    

    for epoch in range(args.max_epoch):

        total_loss = 0
        j = 0
        for iteration, everydata in enumerate(data.create_dict_iterator(), 1):
            input = everydata["img"]
            mask = everydata["mask"]
            edge = everydata["edgemap"]
            canny = everydata["canny"]
            label_panduan = everydata["label_panduan"]
            # print("''''''''''''准备输出loss''''''''''''''''''''''''")
            loss = trainoneStep(input, mask, edge, canny, label_panduan)
            loss = loss.asnumpy()
            # loss = self.network(mixture, len, source, cross_valid)

            # print("输出loss: ", loss)
            
            # print("第{}次trainonestp共花费时间：".format(), t1 - t0)

            print("epoch[{}]({}/{}),loss:{:.4f},stepTime:{}".format(epoch + 1, j+1, steps, loss.asnumpy()))

            # cb_params.cur_step_num = epoch + 1
            # ckpt_cb.step_end(run_context)

            # if j == (step//2):
            # if j % 1001 == 0:
            #     save_ckpt = os.path.join(args.train_url, 'half{}_{}_gdprnn.ckpt'.format(epoch + 1, j))
            #     save_checkpoint(trainoneStep.network, save_ckpt)
            j = j + 1
            total_loss += loss
        train_loss = total_loss/j
        
        
        
        epoch_loss.append(train_loss)
        print("Epoch {} Complete: Avg. Loss: {:.4f}|| Time: {} min {}s.".format(epoch, train_loss))
        save_ckpt = os.path.join(args.save_folder, '{}_{}.ckpt'.format(epoch, args.model_type))
        save_checkpoint(trainoneStep.network, save_ckpt)
        
        

            


if __name__ == '__main__':
    
    set_seed(42)
    train_dataset, val_dataset = setup_loaders(args)
    print("分类种类",args.dataset_cls.num_classes)
    net = GSCNN(num_classes=args.dataset_cls.num_classes)
    print(net)
    # param_dict = load_checkpoint(args.model_path)
    # load_param_into_net(net, param_dict)
    # loss = JointEdgeSegLoss(classes=args.dataset_cls.num_classes, ignore_index=args.dataset_cls.ignore_label, upper_bound=args.wt_bound,
    #                         edge_weight=args.edge_weight, seg_weight=args.seg_weight,
    #                         att_weight=args.att_weight, dual_weight=args.dual_weight)
    """学习率"""
    milestone = []
    learning_rates = []
    for i in range(1, 10):
        milestone.append(i)
        learning_rates.append(math.pow((1 - i / 175), 1.0)*0.01)
    #print(learning_rates)
    lr = nn.piecewise_constant_lr(milestone, learning_rates)

    def poly_lr(base_lr, epoch_steps, total_steps, end_lr=0.0001, power=0.9):
        for i in range(total_steps):
            if i % epoch_steps == 0
                base_lr = base_lr * 0.98
            yield base_lr
            


    """优化器"""
    param_groups = net.trainable_params()

    optim = nn.SGD(param_groups, learning_rate=0.01, weight_decay=0.0001,
                        momentum=0.9, nesterov=False, loss_scale=1.0)
    #optim = nn.Adam(param_groups, learning_rate=0.01, weight_decay=args.weight_decay)
                                            
    #optim = nn.Momentum(params=net.trainable_params(), learning_rate=lr, momentum=0.9)

    
    lossNetwork = Generatorloss(net)
    trainonestepNet = TrainOneStep(lossNetwork, optim)
    #train(trainonestepNet, train_dataset)
    loss_cb = LossMonitor()
    time_cb = TimeMonitor()
    config_ck = CheckpointConfig(save_checkpoint_steps=95, keep_checkpoint_max=2)
    # config_ck = CheckpointConfig(save_checkpoint_steps=num_steps, keep_checkpoint_max=1)^M
    ckpt_cb = ModelCheckpoint(prefix='GSCNN_ckpt',
                              directory=args.save_folder,
                              config=config_ck)
    cb = [loss_cb, ckpt_cb]
    #loss_scale = FixedLossScaleManager(0.5, drop_overflow_update=False)^M
    #model = Model(net, optimizer=optim,  loss_scale_manager=loss_scale)^M
    
    model = Model(trainonestepNet)
    model.train(epoch=args.max_epoch, train_dataset=train_dataset, callbacks=cb, dataset_sink_mode=False)

    
