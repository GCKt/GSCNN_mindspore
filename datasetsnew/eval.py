import os

import mindspore

from newutils import AverageMeter, fast_hist, evaluate_eval

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context
from config import assert_and_infer_cfg
from datasetsnew.setup_data import setup_loaders
import numpy as np

# Argument Parser
from network.NewGSCNN import NewGSCNN
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
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
parser.add_argument('--save_folder', default='/home/heu_MEDAI/gongcheng/19Channel/checkpoint',
                    help='Location to save epoch models')
parser.add_argument('--model_path', type=str,
                    default="/home/heu_MEDAI/gongcheng/19Channel/checkpoint/GSCNN_ckpt_15-2_2826.ckpt",
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



def evaluate_eval(hist ):
    '''
    Modified IOU mechanism for on-the-fly IOU calculations ( prevents memory overflow for
    large dataset) Only applies to eval/eval.py
    '''
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))


    freq = hist.sum(axis=1) / hist.sum()
    mean_iu = np.nanmean(iu)
    return mean_iu


def validate():
    '''
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    '''
    assert_and_infer_cfg(args)

    train_dataset, val_dataset = setup_loaders(args)

    net = NewGSCNN(num_classes=args.dataset_cls.num_classes)
    net.set_train(mode=False)
    param_dict = load_checkpoint(args.model_path)
    load_param_into_net(net, param_dict)
    val_loss = AverageMeter()
    mf_score = AverageMeter()
    IOU_acc = 0
    dump_images = []
    heatmap_images = []
    vi = 0
    for data in val_dataset.create_dict_iterator():
        input = data["img"]
        mask = data["mask"]
        edge = data["edgemap"]
        canny = data["canny"]

        # mask_shape = mask.shape
        # h = mask_shape[1]
        # w = mask_shape[2]
        #
        # input_shape = input.shape
        # batch_pixel_size = input_shape[0]*input_shape[2]*input_shape[3]



        seg_out, edge_out = net(input, mask,edge,canny)    # output = (1, 19, 713, 713)



        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions

        argmax = mindspore.ops.ArgMaxWithValue(axis=1)
        seg_predictions, output = argmax(seg_out)
        print("开始")
        print(seg_predictions.sum())
        # edge_predictions = edge_out.max(1)[0]
        #
        # #Logging
        #
        # _edge = edge.max(1)[0]

        #Image Dumps

        print("估算准确率")
        IOU_acc += fast_hist(seg_predictions.asnumpy().flatten(), mask.asnumpy().flatten())
        print("输出IOU")

        mean_iu = evaluate_eval(IOU_acc)
        print(mean_iu)
    # if args.local_rank == 0:
    #     evaluate_eval(args, net, optimizer, val_loss, mf_score, IOU_acc, dump_images, heatmap_images,
    #             writer, curr_epoch, args.dataset_cls)

    return mean_iu


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    validate()