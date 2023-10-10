import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice_loss import BinaryDiceLoss


class BinarySegLossKD(nn.Module):

    def __init__(self, num_classes):
        super(BinarySegLossKD, self).__init__()
        self.binary_dice_loss = BinaryDiceLoss(smooth=1, p=2, reduction='mean')
        self.avg_pool = nn.AdaptiveAvgPool2d(8)
        self.num_classes = num_classes

    def binary_seg_loss(self, pred_S, pred_T, target):
        bce_loss = 0
        dic_loss = 0
        for i in range(self.num_classes):
            # binary cross entropy
            pos_weight = (target[:, i] == 0).float().sum() / \
                         (target[:, i] == 1).float().sum().clamp(min=1.0)
            bce_loss = bce_loss + F.binary_cross_entropy_with_logits(
                pred_S[:, i], target[:, i].float(), reduction='mean', pos_weight=pos_weight)
            bce_loss = bce_loss + F.binary_cross_entropy_with_logits(
                pred_S[:, i], pred_T[:, i].sigmoid(), reduction='mean', pos_weight=pos_weight)
            # binary dice loss
            dic_loss = dic_loss + self.binary_dice_loss(pred_S[:, i].sigmoid(), target[:, i].float())
            dic_loss = dic_loss + self.binary_dice_loss(pred_S[:, i].sigmoid(), pred_T[:, i].sigmoid())
        seg_loss = bce_loss + dic_loss
        return seg_loss

    def similarity_loss(self, feat_S, feat_T, target):
        target = target[:, :self.num_classes]                                            # (N, M, H, W)
        target = torch.cat([target, 1 - target.any(dim=1, keepdim=True)], dim=1)         # (N, M, H, W)
        target = F.interpolate(target.float(), size=feat_S.shape[-2:], mode='nearest')   # (N, M, H, W)
        target = target.flatten(2)                                                       # (N, M, H * W)
        number = target.sum(dim=2).clamp(min=1.0).unsqueeze(dim=1)                       # (N, 1, M)
        feat_S = feat_S.flatten(2)                                                       # (N, C, H * W)
        feat_T = feat_T.flatten(2)                                                       # (N, C, H * W)
        mean_S = feat_S.bmm(target.permute(0, 2, 1)) / number                            # (N, C, M)
        mean_T = feat_T.bmm(target.permute(0, 2, 1)) / number                            # (N, C, M)

        # variance loss
        vari_S = feat_S[:, :, :, None] - mean_S[:, :, None, :]                           # (N, C, H * W, M)
        vari_T = feat_T[:, :, :, None] - mean_T[:, :, None, :]                           # (N, C, H * W, M)
        vari_S = vari_S * vari_S                                                         # (N, C, H * W, M)
        vari_T = vari_T * vari_T                                                         # (N, C, H * W, M)
        vari_S = vari_S.sum(dim=2) / number                                              # (N, C, M)
        vari_T = vari_T.sum(dim=2) / number                                              # (N, C, M)
        vari_S = F.normalize(vari_S, p=2, dim=1)                                         # (N, C, M)
        vari_T = F.normalize(vari_T, p=2, dim=1)                                         # (N, C, M)
        simi_S = vari_S.permute(0, 2, 1).bmm(vari_S)                                     # (N, M, M)
        simi_T = vari_T.permute(0, 2, 1).bmm(vari_T)                                     # (N, M, M)
        vari_loss = F.mse_loss(simi_S, simi_T, reduction='mean')

        # mean loss
        mean_S = F.normalize(mean_S, p=2, dim=1)                                         # (N, C, M)
        mean_T = F.normalize(mean_T, p=2, dim=1)                                         # (N, C, M)
        simi_S = mean_S.permute(0, 2, 1).bmm(mean_S)                                     # (N, M, M)
        simi_T = mean_T.permute(0, 2, 1).bmm(mean_T)                                     # (N, M, M)
        mean_loss = F.mse_loss(simi_S, simi_T, reduction='mean')

        # attention loss
        attn_S = feat_S * feat_S                                                         # (N, C, H * W)
        attn_T = feat_T * feat_T                                                         # (N, C, H * W)
        attn_S = attn_S.mean(dim=1)                                                      # (N, H * W)
        attn_T = attn_T.mean(dim=1)                                                      # (N, H * W)
        attn_S = F.normalize(attn_S, p=2, dim=1)                                         # (N, H * W)
        attn_T = F.normalize(attn_T, p=2, dim=1)                                         # (N, H * W)
        attn_loss = F.mse_loss(attn_S, attn_T, reduction='mean')

        # pair loss
        feat_S = self.avg_pool(feat_S)                                                   # (N, C, H, W)
        feat_T = self.avg_pool(feat_T)                                                   # (N, C, H, W)
        feat_S = feat_S.flatten(2)                                                       # (N, C, H * W)
        feat_T = feat_T.flatten(2)                                                       # (N, C, H * W)
        feat_S = F.normalize(feat_S, p=2, dim=1)                                         # (N, C, H * W)
        feat_T = F.normalize(feat_T, p=2, dim=1)                                         # (N, C, H * W)
        simi_S = feat_S.permute(0, 2, 1).bmm(feat_S)                                     # (N, H * W, H * W)
        simi_T = feat_T.permute(0, 2, 1).bmm(feat_T)                                     # (N, H * W, H * W)
        pair_loss = F.mse_loss(simi_S, simi_T, reduction='mean')

        simi_loss = vari_loss + mean_loss + attn_loss + pair_loss
        return simi_loss

    def _forward_G(self, pred_S, pred_T, feat_S, feat_T, fake_D, target):
        # Segmentation Loss
        seg_loss = self.binary_seg_loss(pred_S, pred_T, target)

        # pair-wise loss
        sim_loss = self.similarity_loss(feat_S, feat_T, target)
        
        # adversarial loss
        gen_loss = -torch.mean(fake_D)
        return seg_loss, sim_loss, gen_loss

    def _forward_D(self, fake_D, real_D):
        loss_real = torch.mean(F.relu(1. - real_D))
        loss_fake = torch.mean(F.relu(1. + fake_D))
        return loss_real, loss_fake

    def forward_G(self, pred_S, pred_T, feat_S, feat_T, fake_D, target):
        if isinstance(pred_S, (list, tuple)):
            seg_loss, sim_loss, gen_loss = 0, 0, 0
            for _, (p, x, d) in enumerate(zip(pred_S, feat_S, fake_D)):
                seg_loss_, sim_loss_, gen_loss_ = self._forward_G(p, pred_T, x, feat_T, d, target)
                seg_loss = seg_loss + seg_loss_
                sim_loss = sim_loss + sim_loss_
                gen_loss = gen_loss + gen_loss_
        else:
            seg_loss, sim_loss, gen_loss = self._forward_G(pred_S, pred_T, feat_S, feat_T, fake_D, target)
        return seg_loss, sim_loss, gen_loss

    def forward_D(self, fake_D, real_D):
        if isinstance(fake_D, (list, tuple)):
            dis_loss_real = 0
            dis_loss_fake = 0
            for x in fake_D:
                loss_real, loss_fake = self._forward_D(x, real_D)
                dis_loss_real = dis_loss_real + loss_real
                dis_loss_fake = dis_loss_fake + loss_fake
        else:
            dis_loss_real, dis_loss_fake = self._forward_D(fake_D, real_D)
        return dis_loss_real, dis_loss_fake
