# import mindspore
# import mindspore.ops as ops
# from mindspore import Tensor
# import numpy as np
# from mindspore import nn as nn
# from mindspore.dataset import context

# from loss import JointEdgeSegLoss
# from network.mynn import initialize_weights, Norm2d
# import cv2
# import argparse
# from datasetsnew import setup_data
# from network.gscnn import GSCNN

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

# parser.add_argument('--edge_weight', type=float, default=1.0,
#                     help='Edge loss weight for joint loss')
# parser.add_argument('--seg_weight', type=float, default=1.0,
#                     help='Segmentation loss weight for joint loss')
# parser.add_argument('--att_weight', type=float, default=1.0,
#                     help='Attention loss weight for joint loss')
# parser.add_argument('--dual_weight', type=float, default=1.0,
#                     help='Dual loss weight for joint loss')

# parser.add_argument('--evaluate', action='store_true', default=False)

# parser.add_argument("--local_rank", default=0, type=int)

# parser.add_argument('--sgd', action='store_true', default=True)
# parser.add_argument('--sgd_finetuned',action='store_true',default=False)
# parser.add_argument('--adam', action='store_true', default=False)
# parser.add_argument('--amsgrad', action='store_true', default=False)

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

# if __name__ == '__main__':
#     context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=1)
#     dataset, valset = setup_data.setup_loaders(args)
#     # data = trainset.__getitem__(0)
#     # print(len(trainset.__getitem__(0)))
#     # dataset, val, train_data = setup_data.setup_loaders(args)
#     # data = train_data.__getitem__(36)
#     # print(data[0])
#     # print(data[1])
#     # print(data[2])
#     # print(data[3])
#     i = 0
#     for data in dataset.create_dict_iterator():
#         #print(data["img"], data["canny"])
#         inputs = data["img"]
#         canny = data["canny"]
#         mask = data["mask"]
#         edgemap = data["edgemap"]
#         # print(inputs)
#         # print(canny)
#         # print(mask)
#         # edge = np.load("edgenew.npy")
#         # edge = mindspore.Tensor(edge)
#         # masknew = np.load("masknew.npy")
#         # masknew = mindspore.Tensor(masknew)
#         # sk = (edge == edgemap)
#         # print(edge == edgemap)
#         # print("下一个")
#         # print(mask == masknew)
#         model = GSCNN(19).set_train(True)
#         #model = GSCNN(19)
#         output1 = model(inputs, mask, edgemap, canny)
#         # output1, output2 = model(inputs, canny)
#         # output1 = output1.asnumpy()
#         # output2 = output2.asnumpy()
#         # mask = mask.asnumpy()
#         # edgemap = edgemap.asnumpy()
#         # np.save("output1.npy", output1)
#         # np.save("output2.npy", output2)
#         # np.save("mask.npy", mask)
#         # np.save("edgemap.npy", edgemap)
#         i = i + 1
#         print("-------------------------------开始----------------------------------------------")
#         print("----------------------------------输出loss-----------------------------------------")
#         print(i)
#         print(output1)

#         print("----------------------------------------------------------------------------")