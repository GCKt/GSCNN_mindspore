import mindspore
import mindspore.nn as nn
from mindspore.ops.functional import stop_gradient
from mindspore import numpy as np_new
from mindspore import Tensor
import numpy as np
from my_functionals.custom_functional import compute_grad_mag


def perturbate_input_(input, n_elements=200):
    N, C, H, W = input.shape
    assert N == 1
    c_ = np.random.random_integers(0, C - 1, n_elements)
    h_ = np.random.random_integers(0, H - 1, n_elements)
    w_ = np.random.random_integers(0, W - 1, n_elements)
    for c_idx in c_:
        for h_idx in h_:
            for w_idx in w_:
                input[0, c_idx, h_idx, w_idx] = 1
    return input


def _sample_gumbel(shape, eps=1e-10):
    """
    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    # minval = 0
    # minval = minval.astype(dtype=mindspore.int32)
    # minval = Tensor(0, mindspore.float32)
    # maxval = Tensor(1, mindspore.float32)
    zeros = mindspore.ops.Zeros()
    minval = zeros(1, mindspore.float32)

    ones = mindspore.ops.Ones()
    maxval = ones(1, mindspore.float32)


    U = mindspore.ops.uniform(shape, minval, maxval)

    log = mindspore.ops.Log()

    #k = - log(eps - log(U + eps))
    return - log(eps - log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    #assert logits.ndim == 3
    gumbel_noise = _sample_gumbel(logits.shape, eps=eps)
    y = logits + gumbel_noise
    softmax = mindspore.ops.Softmax(axis= 1)
    return softmax(y / tau)


def _one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    '''torch.eye这个函数主要是为了生成对角线全1，其余部分全0的二维数组'''
    eye = mindspore.ops.Eye()
    y = eye(num_classes, num_classes, mindspore.int32)
    #y = torch.eye(num_classes).cuda()
    '''torch.permute这个函数主要是将tensor的维度换位。'''
    transpose = mindspore.ops.Transpose()
    y = transpose(y[labels], (0, 3, 1, 2))
    return y


class DualTaskLoss(nn.Cell):
    def __init__(self, cuda=False):
        super(DualTaskLoss, self).__init__()
        self._cuda = cuda
        return

    def construct(self, input_logits, gts, ignore_pixel=255):
        """
        :param input_logits: NxCxHxW
        :param gt_semantic_masks: NxCxHxW
        :return: final loss
        """
        N, C, H, W = input_logits.shape
        th = 1e-8  # 1e-10
        eps = 1e-10
        zeros = mindspore.ops.Zeros()
        shape = (N, 19, H, W)
        broadcast_to = mindspore.ops.BroadcastTo(shape)
        ignore_mask = (gts == ignore_pixel)
        ignore_mask = ignore_mask.astype(dtype=mindspore.int32)
        ignore_mask = stop_gradient(ignore_mask)
        ignore_mask_new = broadcast_to(ignore_mask.view(N, 1, H, W))
        ignore_mask_new = ignore_mask_new.astype(dtype=mindspore.bool_)
        ignore_mask = ignore_mask.astype(dtype=mindspore.bool_)
        """修改int64为int32"""
        input_logits = np_new.where(ignore_mask_new,
                                   zeros((N, C, H, W), mindspore.int64),
                                   input_logits)

        gt_semantic_masks = gts
        gt_semantic_masks = stop_gradient(gt_semantic_masks)
        gt_semantic_masks = np_new.where(ignore_mask, zeros((N, H, W), mindspore.int32), gt_semantic_masks)
        gt_semantic_masks = _one_hot_embedding(gt_semantic_masks,19)
        gt_semantic_masks = stop_gradient(gt_semantic_masks)

        g = _gumbel_softmax_sample(input_logits.view(N, C, -1), tau=0.5)

        reshape = mindspore.ops.Reshape()
        g = reshape(g, (N, C, H, W))
        g = compute_grad_mag(g, cuda=self._cuda)

        g_hat = compute_grad_mag(gt_semantic_masks, cuda=self._cuda)

        g = g.view(N, -1)
        g_hat = reshape(g_hat, (N, -1))
        l1loss = nn.L1Loss(reduction='none')
        loss_ewise = l1loss(g, g_hat)



        p_plus_g_mask = (g >= th)
        p_plus_g_mask = stop_gradient(p_plus_g_mask)
        loss_p_plus_g = ((loss_ewise * p_plus_g_mask).astype(dtype=mindspore.float32)).sum() / ((p_plus_g_mask).astype(dtype=mindspore.float32).sum() + eps)

        p_plus_g_hat_mask = (g_hat >= th)
        p_plus_g_hat_mask = stop_gradient(p_plus_g_hat_mask)

        loss_p_plus_g_hat = ((loss_ewise * p_plus_g_hat_mask).astype(dtype=mindspore.float32)).sum() / ((p_plus_g_hat_mask.astype(dtype=mindspore.float32)).sum() + eps)

        total_loss = 0.5 * loss_p_plus_g + 0.5 * loss_p_plus_g_hat

        return total_loss