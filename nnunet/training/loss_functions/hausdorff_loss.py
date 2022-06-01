import numpy as np
import torch

from torch import nn
from scipy.ndimage import distance_transform_edt as distance

class HausdorffLoss(nn.Module):
    def __init__(self, apply_nonlin=None):
        super(HausdorffLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
    
    def compute_dtm(self, img, img_shape):
        fg_dtm = np.zeros(img_shape)
        for b in range(img_shape[0]): # batch size
            for c in range(1, img_shape[1]):
                posmask = img[b].astype(np.bool)
                if posmask.any():
                    posdis = distance(posmask)
                    # print("posdis")
                    # print(posdis.shape)
                    # print(fg_dtm[b][c].shape)
                    # print(posdis)
                    fg_dtm[b][c] = posdis
        return fg_dtm

    def compute_dtm01(self, img, img_shape):
        normalized_dtm = np.zeros(img_shape)
        for b in range(img_shape[0]): # batch size
                # ignore background
            for c in range(1, img_shape[1]):
                posmask = img[b].astype(np.bool)
                if posmask.any():
                    posdis = distance(posmask)
                    normalized_dtm[b][c] = posdis/np.max(posdis)

        return normalized_dtm

    def hd_loss(self, seg_soft, gt, seg_dtm, gt_dtm):
        delta_s = (seg_soft[:,1,...] - gt.float()) ** 2
        s_dtm = seg_dtm[:,1,...] ** 2
        g_dtm = gt_dtm[:,1,...] ** 2
        dtm = s_dtm + g_dtm
        multipled = torch.einsum('bxy, bxy->bxy', delta_s, dtm)
        hd_loss = multipled.mean()

        return hd_loss

    def forward(self, output, target):
        if self.apply_nonlin:
            output = self.apply_nonlin(output)
        with torch.no_grad():
            gt_dtm_npy = self.compute_dtm01(target.cpu().numpy()[:, 0, :, :], output.shape)
            gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(output.device.index)
            seg_dtm_npy = self.compute_dtm01(output[:, 1, :, :].cpu().numpy()>0.5, output.shape)
            seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(output.device.index)
        
        loss_hd = self.hd_loss(output, target[:, 0, :, :], seg_dtm, gt_dtm)
        return loss_hd

