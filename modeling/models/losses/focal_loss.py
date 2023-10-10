# pylint: disable-all
from typing import Optional, List  # <-Change:  Add List

import torch
import torch.nn as nn
import torch.nn.functional as F


# Source: https://github.com/kornia/kornia/blob/f4f70fefb63287f72bc80cd96df9c061b1cb60dd/kornia/losses/focal.py


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> kornia.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError(
            "Input labels type is not a torch.Tensor. Got {}".format(
                type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0],
                          num_classes,
                          *shape[1:],
                          device=device,
                          dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


# def focal_loss(input: torch.Tensor,
#                target: torch.Tensor,
#                alpha: torch.Tensor,  # float,
#                gamma: float = 2.0,
#                reduction: str = 'none',
#                eps: float = 1e-8) -> torch.Tensor:
#     r"""Function that computes Focal loss.
#     See :class:`~kornia.losses.FocalLoss` for details.
#     """
#     if not torch.is_tensor(input):
#         raise TypeError("Input type is not a torch.Tensor. Got {}".format(
#             type(input)))
#
#     if not len(input.shape) >= 2:
#         raise ValueError(
#             "Invalid input shape, we expect BxCx*. Got: {}".format(
#                 input.shape))
#
#     if input.size(0) != target.size(0):
#         raise ValueError(
#             'Expected input batch_size ({}) to match target batch_size ({}).'.
#             format(input.size(0), target.size(0)))
#
#     n = input.size(0)
#     out_size = (n, ) + input.size()[2:]
#     if target.size()[1:] != input.size()[2:]:
#         raise ValueError('Expected target size {}, got {}'.format(
#             out_size, target.size()))
#
#     if not input.device == target.device:
#         raise ValueError(
#             "input and target must be in the same device. Got: {} and {}".
#             format(input.device, target.device))
#
#     # compute softmax over the classes axis
#     input_soft: torch.Tensor = F.softmax(input, dim=1) + eps
#
#     # create the labels one hot tensor
#     target_one_hot: torch.Tensor = one_hot(target,
#                                            num_classes=input.shape[1],
#                                            device=input.device,
#                                            dtype=input.dtype)
#
#     # compute the actual focal loss
#     weight = torch.pow(-input_soft + 1., gamma)
#
#     focal = -alpha * weight * torch.log(input_soft)
#     loss_tmp = torch.sum(target_one_hot * focal, dim=1)
#
#     if reduction == 'none':
#         loss = loss_tmp
#     elif reduction == 'mean':
#         loss = torch.mean(loss_tmp)
#     elif reduction == 'sum':
#         loss = torch.sum(loss_tmp)
#     else:
#         raise NotImplementedError(
#             "Invalid reduction mode: {}".format(reduction))
#     return loss
#
#
# class FocalLoss(nn.Module):
#     r"""Criterion that computes Focal loss.
#     According to [1], the Focal loss is computed as follows:
#     .. math::
#         \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
#     where:
#        - :math:`p_t` is the model's estimated probability for each class.
#     Arguments:
#         alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
#         gamma (float): Focusing parameter :math:`\gamma >= 0`.
#         reduction (str, optional): Specifies the reduction to apply to the
#          output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
#          ‘mean’: the sum of the output will be divided by the number of elements
#          in the output, ‘sum’: the output will be summed. Default: ‘none’.
#     Shape:
#         - Input: :math:`(N, C, *)` where C = number of classes.
#         - Target: :math:`(N, *)` where each value is
#           :math:`0 ≤ targets[i] ≤ C−1`.
#     Examples:
#         >>> N = 5  # num_classes
#         >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
#         >>> loss = kornia.losses.FocalLoss(**kwargs)
#         >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
#         >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
#         >>> output = loss(input, target)
#         >>> output.backward()
#     References:
#         [1] https://arxiv.org/abs/1708.02002
#     """
#     def __init__(self,
#                  alpha: float,
#                  gamma: float = 2.0,
#                  reduction: str = 'none') -> None:
#         super(FocalLoss, self).__init__()
#         # self.alpha: float = alpha
#         self.gamma: float = gamma
#         self.reduction: str = reduction
#         self.eps: float = 1e-6
#         self.register_buffer('alpha', torch.tensor([[1 - alpha], [alpha]], dtype=torch.float32))
#
#     def forward(  # type: ignore
#             self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return focal_loss(input, target, self.alpha, self.gamma,
#                           self.reduction, self.eps)


def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alphas: Optional[List[float]],  # <-Change:  rename to alphas and to a list of floats
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Focal loss.

    See :class:`~kornia.losses.FocalLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}".format(
                input.device, target.device))

    ### New addition ###
    alphas = torch.tensor(alphas, dtype=input.dtype, device=input.device).view(1, -1, *[1] * (input.ndim - 2))
    if alphas.size(1) != input.size(1):
        raise ValueError("Invalid alphas shape. we expect{} alpha values. Got: {}"
                         .format(input.size(1), alphas.size(1)))

    # Normalize alphas to sum up 1
    alphas.div_(alphas.sum())

    # Original code:

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alphas * weight * torch.log(input_soft)  # <-Change:  alpha -> alphas
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = kornia.losses.FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alphas: Optional[List[float]], gamma: float = 2.0,  # <- Change:  alpha to alphas
                 reduction: str = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alphas: Optional[List[float]] = alphas  # <- Change:  alpha to alphas
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alphas, self.gamma, self.reduction,
                          self.eps)  # <- Change:  alpha to alphas


class FocalLossV2(torch.nn.Module):
    def __init__(self, gamma=2, beta=4, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, gt, mask=None):
        positive_mask = abs(gt - 1) < 5e-2
        negative_mask = abs(gt - 1) > 5e-2
        loss_positive = ((1 - pred) * positive_mask) ** self.gamma * torch.log(pred * positive_mask + 1e-9)
        loss_negative = (
            ((1 - gt) * negative_mask) ** self.beta
            * (pred * negative_mask) ** self.gamma
            * torch.log((1 - pred) * negative_mask + 1e-9)
        )
        if self.reduction == 'mean':
            loss = -torch.mean(loss_negative + loss_positive)
        else:
            loss = -(loss_negative + loss_positive)
        return loss


# class FocalLossV2(torch.nn.Module):
#     def __init__(self, gamma=2, beta=4, reduction='mean'):
#         super().__init__()
#         self.gamma = gamma
#         self.beta = beta
#         self.reduction = reduction
#
#     def forward(self, pred, gt, mask=None):
#         pred = pred.clamp(min=0.0001, max=1.0)
#         positive_mask = abs(gt - 1) < 5e-2
#         negative_mask = abs(gt - 1) > 5e-2
#         loss_positive = ((1 - pred) * positive_mask) ** self.gamma * torch.log((pred * positive_mask).clamp(min=0.0001, max=1.0))
#         loss_negative = (
#             ((1 - gt) * negative_mask) ** self.beta
#             * (pred * negative_mask) ** self.gamma
#             * torch.log(((1 - pred) * negative_mask).clamp(min=0.0001, max=1.0))
#         )
#         if self.reduction == 'mean':
#             loss = -torch.mean(loss_negative + loss_positive)
#         else:
#             loss = -(loss_negative + loss_positive)
#         return loss


def sigmoid_focal_loss(
    logits,
    targets,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        logits: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as logits. Stores the binary
                 classification label for each element in logits
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss
)  # type: torch.jit.ScriptModule


def sigmoid_focal_loss_star(
    logits,
    targets,
    alpha: float = -1,
    gamma: float = 1,
    reduction: str = "none",
):
    """
    FL* described in RetinaNet paper Appendix: https://arxiv.org/abs/1708.02002.
    Args:
        logits: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as logits. Stores the binary
                 classification label for each element in logits
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Gamma parameter described in FL*. Default = 1 (no weighting).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    shifted_logits = gamma * (logits * (2 * targets - 1))
    loss = -F.logsigmoid(shifted_logits) / gamma

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss *= alpha_t

    if reduction == "mean":
        loss = loss.mean()  # pyre-ignore
    elif reduction == "sum":
        loss = loss.sum()  # pyre-ignore

    return loss


sigmoid_focal_loss_star_jit = torch.jit.script(
    sigmoid_focal_loss_star
)  # type: torch.jit.ScriptModule
