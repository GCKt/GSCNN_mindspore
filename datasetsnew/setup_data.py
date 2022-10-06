"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


# import transforms.joint_transforms as joint_transforms
# import transforms.transforms as extended_transforms
import mindspore
import mindspore.dataset.vision.py_transforms as py_transforms
import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms as tc
from datasetsnew import cityscapes

import datasetsnew.data_transform as newTransform


def setup_loaders(args,rank_size, rank_id):
    '''
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    '''

    if args.dataset == 'cityscapes':
        '''调用了ciytscapes.py文件,args.dataset_cls就相当于文件了，可以任意调用
        里面包含的类和函数'''
        args.dataset_cls = cityscapes
        # args.train_batch_size = args.bs_mult * args.ngpu
        args.train_batch_size = 1
        args.val_batch_size = 1
        # if args.bs_mult_val > 0:
        #     args.val_batch_size = args.bs_mult_val * args.ngpu
        # else:
        #     args.val_batch_size = args.bs_mult * args.ngpu
    else:
        raise
    args.num_workers = 1
    # args.num_workers = 4 * args.ngpu
    if args.test_mode:
        args.num_workers = 0  # 1

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Geometric image transformations
    # 几何图像变换


    train_joint_transform_list = [
        # 随机形状以及裁剪
        # joint_transforms.RandomSizeAndCrop(args.crop_size,
        #                                    False,
        #                                    pre_size=args.pre_size,
        #                                    scale_min=args.scale_min,
        #                                    scale_max=args.scale_max,
        #                                    ignore_index=args.dataset_cls.ignore_label),
        # 改变形状
        newTransform.Resize(args.crop_size),
        # 水平翻转
        # joint_transforms.RandomHorizontallyFlip()
    ]

    # if args.rotate:
    #    train_joint_transform_list += [joint_transforms.RandomRotate(args.rotate)]

    train_joint_transform = newTransform.Compose(train_joint_transform_list)

    # Image appearance transformations
    train_input_transform = []
    # if args.color_aug:
    #     train_input_transform += [
    #         # 随机改变图像的亮度、对比度和饱和度
    #         mindspore.dataset.vision.c_transforms.RandomColorAdjust(
    #             brightness=args.color_aug,
    #             contrast=args.color_aug,
    #             saturation=args.color_aug,
    #             hue=args.color_aug)]
    #
    # if args.bblur:
    #     train_input_transform += [extended_transforms.RandomBilateralBlur()]
    # elif args.gblur:
    #     train_input_transform += [extended_transforms.RandomGaussianBlur()]
    # else:
    #     pass

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_input_transform += [
        newTransform.Normalize(mean=mean, std=std)
        # mindspore.dataset.vision.c_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
    train_input_transform = newTransform.Compose(train_input_transform)

    val_input_transform = newTransform.Compose([
        newTransform.Normalize(mean=mean, std=std)
    ])

    target_transform = None

    target_train_transform = None

    city_mode = 'train'  ## Can be trainval
    city_quality = 'fine'
    train_set = args.dataset_cls.CityScapes(
        city_quality, city_mode, 0,
        joint_transform=train_joint_transform,
        transform=train_input_transform,
        target_transform=target_train_transform,
        dump_images=args.dump_augmentation_images,
        cv_split=args.cv)
    val_set = args.dataset_cls.CityScapes('fine', 'val', 0,
                                          joint_transform=train_joint_transform,
                                          transform=val_input_transform,
                                          target_transform=target_transform,
                                          cv_split=args.cv)
    train_sampler = None
    val_sampler = None

    train_dataset = ds.GeneratorDataset(train_set, column_names=["img", "mask", "edgemap",  "canny", "label_panduan"],
                                  shuffle=True, num_parallel_workers=2,num_shards=rank_size, shard_id=rank_id)

    train_dataset = train_dataset.batch(args.train_batch_size, drop_remainder=False)

    val_dataset = ds.GeneratorDataset(val_set, column_names=["img", "mask", "edgemap",  "canny", "label_panduan"],
                                  shuffle=True, num_parallel_workers=2,num_shards=rank_size, shard_id=rank_id)
    val_dataset = val_dataset.batch(args.train_batch_size, drop_remainder=False)

    # train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
    #                           num_workers=args.num_workers, shuffle=(train_sampler is None), drop_last=True,
    #                           sampler=train_sampler)
    # val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
    #                         num_workers=args.num_workers // 2, shuffle=False, drop_last=False, sampler=val_sampler)

    #return train_dataset, val_dataset, train_set
    return train_dataset, val_dataset

