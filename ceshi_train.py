# import os
# import sys
# import logging
# import argparse
# from collections import OrderedDict
# import numpy
# import yaml
# import mindspore.common.initializer as weight_init
# import transforms.joint_transforms as joint_transforms
# from network.wider_resnet import WiderResNetA2
# from transforms import transforms as extended_transforms
# import mindspore
# import mindspore.dataset.vision.py_transforms as py_transforms
# import mindspore.dataset as ds
# import mindspore.dataset.transforms.py_transforms as tc
# import datasetsnew.data_transform as newTransform
# import os
# import numpy as np
# import mindspore
# from PIL import Image
# from collections import defaultdict
# import math
# import logging
# import datasetsnew.cityscapes_labels as cityscapes_labels
# import json
# from config import cfg
# import mindspore.dataset.vision.py_transforms as py_transforms
# import mindspore.dataset.transforms.py_transforms as tc
# import datasetsnew.edge_utils as edge_utils
# import cv2

# from datasetsnew import cityscapes, setup_data
# from datasetsnew.cityscapes import make_dataset, colorize_mask

# trainid_to_name = cityscapes_labels.trainId2name
# print(trainid_to_name)
# id_to_trainid = cityscapes_labels.label2trainid
# print(id_to_trainid)
# num_classes = 19
# ignore_label = 255
# root = cfg.DATASET.CITYSCAPES_DIR

# palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
#            153, 153, 153, 250, 170, 30,
#            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
#            255, 0, 0, 0, 0, 142, 0, 0, 70,
#            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)

# import mindspore.nn as nn
# from mindspore import Model
# from mindspore import context
# from mindspore import set_seed
# from mindspore.context import ParallelMode
# from mindspore.communication import init
# from mindspore.nn.optim import optimizer
# from mindspore.train.callback import CheckpointConfig
# from mindspore.train.callback import ModelCheckpoint
# from mindspore.train.callback import LossMonitor
# from mindspore.train.callback import TimeMonitor
# from config import assert_and_infer_cfg
# import math
# from datasetsnew.setup_data import setup_loaders
# import cv2

# # Argument Parser
# from loss import JointEdgeSegLoss
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


# args.world_size = 1
# #Test Mode run two epochs with a few iterations of training and val
# if args.test_mode:
#     args.max_epoch = 2

# if 'WORLD_SIZE' in os.environ:
#     args.world_size = int(os.environ['WORLD_SIZE'])
#     print("Total world size: ", int(os.environ['WORLD_SIZE']))


# trainid_to_name = cityscapes_labels.trainId2name
# print(trainid_to_name)
# id_to_trainid = cityscapes_labels.label2trainid
# print(id_to_trainid)
# num_classes = 20
# ignore_label = 255
# root = cfg.DATASET.CITYSCAPES_DIR

# palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
#            153, 153, 153, 250, 170, 30,
#            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
#            255, 0, 0, 0, 0, 142, 0, 0, 70,
#            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)

# def colorize_mask(mask):
#     # mask: numpy array of the mask
#     new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
#     new_mask.putpalette(palette)
#     return new_mask


# def add_items(items, aug_items, cities, img_path, mask_path, mask_postfix, mode, maxSkip):
#     for c in cities:
#         c_items = [name.split('_leftImg8bit.png')[0] for name in
#                    os.listdir(os.path.join(img_path, c))]
#         for it in c_items:
#             item = (os.path.join(img_path, c, it + '_leftImg8bit.png'),
#                     os.path.join(mask_path, c, it + mask_postfix))
#             items.append(item)


# def make_cv_splits(img_dir_name):
#     '''
#     Create splits of train/val data.
#     A split is a lists of cities.
#     split0 is aligned with the default Cityscapes train/val.
#     '''
#     trn_path = os.path.join(root, img_dir_name, 'leftImg8bit', 'train')
#     val_path = os.path.join(root, img_dir_name, 'leftImg8bit', 'val')

#     trn_cities = ['train/' + c for c in os.listdir(trn_path)]
#     val_cities = ['val/' + c for c in os.listdir(val_path)]

#     # want reproducible randomly shuffled
#     trn_cities = sorted(trn_cities)

#     all_cities = val_cities + trn_cities
#     num_val_cities = len(val_cities)
#     num_cities = len(all_cities)

#     cv_splits = []
#     for split_idx in range(cfg.DATASET.CV_SPLITS):
#         split = {}
#         split['train'] = []
#         split['val'] = []
#         offset = split_idx * num_cities // cfg.DATASET.CV_SPLITS
#         for j in range(num_cities):
#             if j >= offset and j < (offset + num_val_cities):
#                 split['val'].append(all_cities[j])
#             else:
#                 split['train'].append(all_cities[j])
#         cv_splits.append(split)

#     return cv_splits


# def make_split_coarse(img_path):
#     '''
#     Create a train/val split for coarse
#     return: city split in train
#     '''
#     all_cities = os.listdir(img_path)
#     all_cities = sorted(all_cities)  # needs to always be the same
#     val_cities = []  # Can manually set cities to not be included into train split

#     split = {}
#     split['val'] = val_cities
#     split['train'] = [c for c in all_cities if c not in val_cities]
#     return split


# def make_test_split(img_dir_name):
#     test_path = os.path.join(root, img_dir_name, 'leftImg8bit', 'test')
#     test_cities = ['test/' + c for c in os.listdir(test_path)]

#     return test_cities


# def make_dataset(quality, mode, maxSkip=0, fine_coarse_mult=6, cv_split=0):
#     '''
#     创建数据集
#     Assemble list of images + mask files

#     fine -   modes: train/val/test/trainval    cv:0,1,2
#     coarse - modes: train/val                  cv:na

#     path examples:
#     leftImg8bit_trainextra/leftImg8bit/train_extra/augsburg
#     gtCoarse/gtCoarse/train_extra/augsburg
#     '''
#     items = []
#     aug_items = []

#     if quality == 'fine':
#         assert mode in ['train', 'val', 'test', 'trainval']
#         img_dir_name = 'leftImg8bit_trainvaltest'
#         img_path = os.path.join(root, img_dir_name, 'leftImg8bit')
#         mask_path = os.path.join(root, 'gtFine_trainvaltest', 'gtFine')
#         mask_postfix = '_gtFine_labelIds.png'
#         cv_splits = make_cv_splits(img_dir_name)
#         if mode == 'trainval':
#             modes = ['train', 'val']
#         else:
#             modes = [mode]
#         for mode in modes:
#             if mode == 'test':
#                 cv_splits = make_test_split(img_dir_name)
#                 add_items(items, cv_splits, img_path, mask_path,
#                           mask_postfix)
#             else:
#                 logging.info('{} fine cities: '.format(mode) + str(cv_splits[cv_split][mode]))

#                 add_items(items, aug_items, cv_splits[cv_split][mode], img_path, mask_path,
#                           mask_postfix, mode, maxSkip)
#     else:
#         raise 'unknown cityscapes quality {}'.format(quality)
#     logging.info('Cityscapes-{}: {} images'.format(mode, len(items) + len(aug_items)))
#     return items, aug_items


# class CityScapes_new():

#     def __init__(self, quality, mode, maxSkip=0, joint_transform=None, sliding_crop=None,
#                  transform=None, target_transform=None, dump_images=False,
#                  cv_split=None, eval_mode=False,
#                  eval_scales=None, eval_flip=False):
#         self.quality = quality
#         self.mode = mode
#         self.maxSkip = maxSkip
#         self.joint_transform = joint_transform
#         self.sliding_crop = sliding_crop
#         self.transform = transform
#         self.target_transform = target_transform
#         self.dump_images = dump_images
#         self.eval_mode = eval_mode
#         self.eval_flip = eval_flip
#         self.eval_scales = None
#         if eval_scales != None:
#             self.eval_scales = [float(scale) for scale in eval_scales.split(",")]
#         # 交叉验证cv_split
#         if cv_split:
#             self.cv_split = cv_split
#             assert cv_split < cfg.DATASET.CV_SPLITS, \
#                 'expected cv_split {} to be < CV_SPLITS {}'.format(
#                     cv_split, cfg.DATASET.CV_SPLITS)
#         else:
#             self.cv_split = 0
#         self.imgs, _ = make_dataset(quality, mode, self.maxSkip, cv_split=self.cv_split)
#         if len(self.imgs) == 0:
#             raise RuntimeError('Found 0 images, please check the data set')

#         self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#     def _eval_get_item(self, img, mask, scales, flip_bool):
#         return_imgs = []
#         for flip in range(int(flip_bool) + 1):
#             imgs = []
#             if flip:
#                 img = img.transpose(Image.FLIP_LEFT_RIGHT)
#             for scale in scales:
#                 w, h = img.size
#                 target_w, target_h = int(w * scale), int(h * scale)
#                 resize_img = img.resize((target_w, target_h))
#                 tensor_img = py_transforms.ToTensor()(resize_img)
#                 final_tensor = py_transforms.Normalize(*self.mean_std)(tensor_img)
#                 imgs.append(tensor_img)
#             return_imgs.append(imgs)
#         return return_imgs, mask

#     def __getitem__(self, index):

#         img_path, mask_path = self.imgs[index]

#         # img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         img = img[:, :, ::-1]
#         mask = Image.open(mask_path)
#         img_name = os.path.splitext(os.path.basename(img_path))[0]
#         print("图片名字")
#         print(img_name)
#         # print(mask_path)

#         mask = np.array(mask)

#         mask_copy = mask.copy()
#         for k, v in id_to_trainid.items():
#             mask_copy[mask == k] = v

#         mask = mask_copy.astype(np.uint8)
#         # mask = Image.fromarray(mask_copy.astype(np.uint8))
#         # img = cv2.merge()
#         # cv2.imshow("IMG", mask)
#         #
#         # # Image Transformations

#         img, mask = self.joint_transform(img, mask)
#         if self.transform is not None:
#             img, mask = self.transform(img, mask)
#         if self.target_transform is not None:
#             mask = self.target_transform(mask)

#         _edgemap = mask

#         _edgemap = edge_utils.mask_to_onehot(_edgemap, num_classes)

#         _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)

#         _edgemap = _edgemap.astype(np.int32)
#         # edgemap = mindspore.Tensor.from_numpy(_edgemap)
#         # edgemap = mindspore.Tensor(edgemap, dtype=mindspore.float32)

#         arr1 = np.array(img, dtype=float)
#         x_size = arr1.shape
#         img = arr1.astype(np.float32)
#         # img = mindspore.Tensor.from_numpy(arr1.astype("float32"))
#         im_arr = arr1.transpose((1, 2, 0)).astype(np.uint8)
#         '''im_arr : torch.Size([2, 96, 96, 3])'''
#         # canny = np.zeros((1,  x_size[1], x_size[2]))
#         '''canny : torch.Size([2, 1, 96, 96])'''
#         # canny获得边缘图像
#         # cv2.Canny()方法可以获得图像的边缘图像

#         canny = cv2.Canny(im_arr, 10, 100)
#         canny = np.reshape(canny, (-1, x_size[1], x_size[2]))
#         canny = canny.astype(np.float32)
#         # canny = mindspore.Tensor.from_numpy(canny)
#         # canny = mindspore.Tensor(canny, dtype=mindspore.float32)
#         mask_shape = mask.shape
#         label_panduan = np.ones(mask_shape)
#         label_panduan[0][0] = 0
#         # Debug

#         # print(img_name)
#         return img, mask, _edgemap, canny, label_panduan
#         # return img, mask, _edgemap

#     def __len__(self):
#         return len(self.imgs)





# def train():
#     '''
#         input: argument passed by the user
#         return:  training data loader, validation data loader loader,  train_set
#         '''

#     if args.dataset == 'cityscapes':
#         '''调用了ciytscapes.py文件,args.dataset_cls就相当于文件了，可以任意调用
#         里面包含的类和函数'''
#         args.dataset_cls = cityscapes
#         # args.train_batch_size = args.bs_mult * args.ngpu
#         args.train_batch_size = 1
#         args.val_batch_size = 1
#         # if args.bs_mult_val > 0:
#         #     args.val_batch_size = args.bs_mult_val * args.ngpu
#         # else:
#         #     args.val_batch_size = args.bs_mult * args.ngpu
#     else:
#         raise
#     args.num_workers = 1
#     # args.num_workers = 4 * args.ngpu
#     if args.test_mode:
#         args.num_workers = 0  # 1

#     mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#     # Geometric image transformations
#     # 几何图像变换


#     train_joint_transform_list = [
#         # 随机形状以及裁剪
#         # joint_transforms.RandomSizeAndCrop(args.crop_size,
#         #                                    False,
#         #                                    pre_size=args.pre_size,
#         #                                    scale_min=args.scale_min,
#         #                                    scale_max=args.scale_max,
#         #                                    ignore_index=args.dataset_cls.ignore_label),
#         # 改变形状
#         newTransform.Resize(args.crop_size),
#         # 水平翻转
#         # joint_transforms.RandomHorizontallyFlip()
#     ]

#     # if args.rotate:
#     #    train_joint_transform_list += [joint_transforms.RandomRotate(args.rotate)]

#     train_joint_transform = newTransform.Compose(train_joint_transform_list)

#     # Image appearance transformations
#     train_input_transform = []
#     # if args.color_aug:
#     #     train_input_transform += [
#     #         # 随机改变图像的亮度、对比度和饱和度
#     #         mindspore.dataset.vision.c_transforms.RandomColorAdjust(
#     #             brightness=args.color_aug,
#     #             contrast=args.color_aug,
#     #             saturation=args.color_aug,
#     #             hue=args.color_aug)]
#     #
#     # if args.bblur:
#     #     train_input_transform += [extended_transforms.RandomBilateralBlur()]
#     # elif args.gblur:
#     #     train_input_transform += [extended_transforms.RandomGaussianBlur()]
#     # else:
#     #     pass

#     value_scale = 255
#     mean = [0.485, 0.456, 0.406]
#     mean = [item * value_scale for item in mean]
#     std = [0.229, 0.224, 0.225]
#     std = [item * value_scale for item in std]

#     train_input_transform += [
#         newTransform.Normalize(mean=mean, std=std)
#         # mindspore.dataset.vision.c_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     ]
#     train_input_transform = newTransform.Compose(train_input_transform)

#     val_input_transform = newTransform.Compose([
#         newTransform.Normalize(mean=mean, std=std)
#     ])

#     target_transform = None

#     target_train_transform = None

#     city_mode = 'train'  ## Can be trainval
#     city_quality = 'fine'
#     train_set = args.dataset_cls.CityScapes(
#         city_quality, city_mode, 0,
#         joint_transform=train_joint_transform,
#         transform=train_input_transform,
#         target_transform=target_train_transform,
#         dump_images=args.dump_augmentation_images,
#         cv_split=args.cv)
#     val_set = args.dataset_cls.CityScapes('fine', 'val', 0,
#                                           transform=val_input_transform,
#                                           target_transform=target_transform,
#                                           cv_split=args.cv)
#     train_sampler = None
#     val_sampler = None

#     data = train_set.__getitem__(0)
#     print(data[0])
#     print(data[1])
#     print(data[2])
#     print(data[3])
#     s = data[2]
#     print(data[4])

#     train_dataset = ds.GeneratorDataset(train_set, column_names=["img", "mask", "edgemap", "canny", "label_panduan"],
#                                         shuffle=True)

#     train_dataset = train_dataset.batch(args.train_batch_size, drop_remainder=False)
#     for data in train_dataset.create_dict_iterator():
#         input = data["img"]
#         mask = data["mask"]
#         edge = data["edgemap"]
#         canny = data["canny"]
#         label_panduan = data["label_panduan"]
#         print(input.shape)
#     # dataset, trainset,test = setup_data.setup_loaders(args)
#     # data = test.__getitem__(0)
#     # print(len(test.__getitem__(0)))
#     # dataset, val, train_data = setup_data.setup_loaders(args)
#     # data = train_data.__getitem__(0)
#     # print(data[0].max())
#     # print(data[1])
#     # print(data[2])
#     # print(data[3])
#     #
#     #
#     #
#     # i = 0
#     # class model_new(nn.Cell):
#     #     def __init__(self):
#     #         super(model_new, self).__init__()
#     #         #wide_resnet = WiderResNetA2(structure=[3, 3, 6, 3, 1, 1], classes=1000, dilation=True)
#     #
#     #
#     #         #self.mod1 =  nn.Conv2d(3, 64, 3, stride=1, has_bias=False, weight_init= "ones")
#     #
#     #         self.b1 = nn.BatchNorm2d(3, eps=1e-05, momentum=0.9)
#     #         self.re = nn.ReLU()
#     #
#     #     def construct(self, x):
#     #         # m1 = self.mod1(x)
#     #         # m1 = self.b1(m1)
#     #         x = self.b1(x)
#     #         x = self.re(x)
#     #         return x
#     #
#     # zeros = mindspore.ops.Zeros()
#     #
#     # m = nn.BatchNorm2d(3, affine=False)
#     # input = mindspore.Tensor(np.random.randn(2, 3, 1, 3), dtype=mindspore.float32)
#     # print(input)
#     # output1 = m(input)
#     # print("结果")
#     # print(output1)



#     #weights = zeros((19), mindspore.float32)
#     # i = 0
#     # for data in dataset.create_dict_iterator():
#     #     # print(data["img"], data["canny"])
#     #     inputs = data["img"]
#     #     canny = data["canny"]
#     #     mask = data["mask"]
#     #     edgemap = data["edgemap"]
#     #     print(inputs.shape)
#     #     print(canny.shape)
#     #     print(mask.shape)
#     #     print(edgemap.shape)
#     #     # print(inputs)
#     #     # print(canny)
#     #     # print(mask)
#     #     # edge = np.load("edgenew.npy")
#     #     # edge = mindspore.Tensor(edge)
#     #     # masknew = np.load("masknew.npy")
#     #     # masknew = mindspore.Tensor(masknew)
#     #     # sk = (edge == edgemap)
#     #     # print(edge == edgemap)
#     #     # print("下一个")
#     #     # print(mask == masknew)
#     #     # model = GSCNN(19).set_train(True)
#     #     # # model = GSCNN(19)
#     #     # output1 = model(inputs, mask, edgemap, canny)
#     #     # # output1, output2 = model(inputs, canny)
#     #     # # output1 = output1.asnumpy()
#     #     # # output2 = output2.asnumpy()
#     #     # # mask = mask.asnumpy()
#     #     # # edgemap = edgemap.asnumpy()
#     #     # # np.save("output1.npy", output1)
#     #     # # np.save("output2.npy", output2)
#     #     # # np.save("mask.npy", mask)
#     #     # # np.save("edgemap.npy", edgemap)
#     #     # i = i + 1
#     #     # print("-------------------------------开始----------------------------------------------")
#     #     # print("----------------------------------输出loss-----------------------------------------")
#     #     # print(i)
#     #     # print(output1)
#     #
#     #     print("----------------------------------------------------------------------------")




# if __name__ == '__main__':
#     train()