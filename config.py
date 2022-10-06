

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# from utils.AttrDict import AttrDict
from uuu.AttrDict import AttrDict
import mindspore
import mindspore.nn as nn

__C = AttrDict()
# Consumers can get config by:
# from fast_rcnn_config import cfg
cfg = __C
__C.EPOCH = 0
__C.CLASS_UNIFORM_PCT=0.0
__C.BATCH_WEIGHTING=False
__C.BORDER_WINDOW=1
__C.REDUCE_BORDER_EPOCH= -1
__C.STRICTBORDERCLASS= None

__C.DATASET = AttrDict()
#__C.DATASET.CITYSCAPES_DIR='/mass_data/Cityscapes'
__C.DATASET.CITYSCAPES_DIR='/home/work/user-job-dir/data/NewCityscapes'

__C.DATASET.CV_SPLITS=3

__C.MODEL = AttrDict()
__C.MODEL.BN = 'regularnorm'
__C.MODEL.BNFUNC = mindspore.nn.BatchNorm2d
__C.MODEL.BIGMEMORY = False

def assert_and_infer_cfg(args, make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """

    if args.batch_weighting:
        __C.BATCH_WEIGHTING=True

    if args.syncbn:
        #import encoding
        __C.MODEL.BN = 'syncnorm'
        '''重点'''
        #__C.MODEL.BNFUNC = encoding.nn.BatchNorm2d
        #__C.MODEL.BNFUNC = mindspore.nn.SyncBatchNorm
        __C.MODEL.BNFUNC = mindspore.nn.BatchNorm2d
        print('Using regular batch norm')
    else:
        __C.MODEL.BNFUNC = mindspore.nn.BatchNorm2d
        #__C.MODEL.BNFUNC = mindspore.nn.SyncBatchNorm
        print('Using regular batch norm')

    if make_immutable:
        cfg.immutable(True)
