import numpy as np
import torch

from torch import nn

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()

# class SpeckleLoss(nn.Module):
#     def __init__(self, apply_nonlin=None):
#         super(SpeckleLoss, self).__init__()
#         self.apply_nonlin = apply_nonlin
    
#     def compute_img_dist(self, targets, imgs):
#         print("IMG DIST")
#         print(targets.shape)
#         print(imgs.shape)
#         pixels = targets * imgs
#         print(pixels.shape)
#         N = torch.sum(targets, dim=(1, 2), keepdim=True)
#         print(N)
#         pix2 = torch.square(pixels)
#         pix4 = torch.square(pix2)
#         e_x2 = torch.sum(pix2) / N
#         e_x4 = torch.sum(pix4) / N
#         scale = e_x2
#         shape = e_x2**2 / ( e_x4 - (e_x2**2))
#         scale[torch.where(scale.isnan())] = 0.0
#         shape[torch.where(shape.isnan())] = 0.0
#         return scale, shape
    
#     def compute_pred_dist(self, outputs, imgs):
#         pixels = outputs * imgs
#         N = torch.sum(outputs, dim=(1, 2), keepdim=True)
#         pix2 = torch.square(pixels)
#         pix4 = torch.square(pix2)
#         e_x2 = torch.sum(pix2) / N
#         e_x4 = torch.sum(pix4) / N
#         scale = e_x2
#         shape = e_x2**2 / ( e_x4 - (e_x2**2))
#         return scale, shape

#     def forward(self, outputs, targets, imgs):
#         if self.apply_nonlin:
#             outputs = self.apply_nonlin(outputs)
#         print(outputs.shape)
#         print(targets.shape)
#         print(targets.dtype)
#         print(torch.unique(targets))
#         print(imgs.shape)
#         masks = torch.nn.functional.one_hot(targets.long())
#         masks = torch.squeeze(masks)
#         masks = masks.permute(0,3,1,2)
#         print(masks.shape)
#         print(torch.unique(masks))
#         img_scale, img_shape = self.compute_img_dist(masks, imgs)
#         pred_scale, pred_shape = self.compute_pred_dist(outputs, imgs)
#         print("new params")
#         # print(img_scale, img_shape)
#         # print(pred_scale, pred_shape)
#         loss = 0.5 * (torch.sum(torch.square(img_scale-pred_scale)) + torch.sum(torch.square(img_shape-pred_shape)))
#         print(loss)
#         return loss 
        

class SpeckleLoss(nn.Module):
    def __init__(self, apply_nonlin=None):
        super(SpeckleLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.rl = torch.nn.ReLU()
            
    def heavyside_approx(self, x, coef=1, shift=0):
        return 1 / (1 + torch.exp(-coef * (x-shift)))

    def compute_img_dist(self, target, img):
        # print(img.shape)
        # print(target.shape)
        if (torch.sum(target)==0):
            return torch.tensor([0.0, 0.0]).cuda()
        pixels = target * img
        # print("pixels", pixels.requires_grad)
        N = torch.sum(target)
        # print("N", N.requires_grad)
        pix2 = torch.square(pixels)
        # print("pix2", pix2.requires_grad)
        pix4 = torch.square(pix2)
        # print("pix4", pix4.requires_grad)
        e_x2 = torch.sum(pix2) / N
        # print("e_x2", e_x2.requires_grad)
        e_x4 = torch.sum(pix4) / N
        # print("e_x4", e_x4.requires_grad)
        scale = e_x2
        if(( e_x4 - (e_x2**2)) == 0):
            shape = 0
        else:
            shape = e_x2**2 / ( e_x4 - (e_x2**2))
    
        return shape, scale

    def forward(self, outputs, targets, imgs, component_weight=0.95):
        loss = torch.tensor(0.0).cuda()
        if self.apply_nonlin:
            outputs = self.apply_nonlin(outputs)
            # outputs = 2 * self.rl(outputs-0.5)
            outputs = self.heavyside_approx(outputs, coef=10, shift=0.5)

        for i in range(imgs.shape[0]):
            img = imgs[i]
            out = outputs[i]
            target = targets[i]
            # print("speckle shapes")
            # print(torch.unique(targets))
            # print(img.shape)
            # print(out.shape)
            # print(target.shape)
            # print(imgs.shape)
            # print(outputs.shape)
            # print(targets.shape)
            for j in range(1, 4):
                mask = target == j
                # print("target", target.requires_grad)
                # print("img", img.requires_grad)
                # print("out", out.requires_grad)
                # print(out[j].shape)
                # print(out[j].requires_grad)
                target_shape, target_scale = self.compute_img_dist(torch.squeeze(mask), torch.squeeze(img))
                # print("target_shape", target_shape)
                # print("target_scale", target_scale)
                # print("Computing output image")
                out_shape, out_scale = self.compute_img_dist(out[j], torch.squeeze(img))
                # print("out_shape", out_shape)
                # print("out_scale", out_scale)
                # print("target_dist", target_dist.requires_grad)
                # print("out_dist", out_dist.requires_grad)
                # print("ditst")
                # print(target_dist)
                # print(out_dist)
                # print(loss)
                # print(torch.sum(torch.square(target_dist - out_dist)) / 2)
                # print("loss1", loss.requires_grad)
                loss += component_weight  * torch.square(target_shape - out_shape) + (1-component_weight) * torch.square(target_scale - out_scale)
                # print(component_weight  * torch.square(target_shape - out_shape), (1-component_weight) * torch.square(target_scale - out_scale))
                # print(torch.square(target_shape - out_shape) / torch.square(target_scale - out_scale))
                # print("loss2", loss.requires_grad)
        # print("loss", loss.requires_grad)
        return loss    





            