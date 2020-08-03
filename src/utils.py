# -*- coding: utf-8 -*-
"""
    Created on Thursday, Jul 16 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Tuesday, Jul 28 2020

Southeast University, College of Automation, 211189 Nanjing China
"""

import torch
import torch.nn as nn
import cv2
import numpy as np

NEAR_0 = 1e-10


def padding(image, width):
    ih, iw, ic = image.shape
    padding1 = np.zeros((ih, width, ic))
    padding2 = np.zeros((width, iw + width*2, ic))
    padded_img = np.concatenate((padding1, image, padding1), axis=1)
    padded_img = np.concatenate((padding2, padded_img, padding2), axis=0)
    return padded_img.astype('uint8')


def fill_hole(image):

    im_floodfill = image.copy()

    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    isbreak = False
    seedPoint = None
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if im_floodfill[i][j] == 0:
                seedPoint = (i, j)
                isbreak = True
                break
        if isbreak:
            break
    if not isbreak:
        return image

    cv2.floodFill(im_floodfill, mask, seedPoint, 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    im_out = image | im_floodfill_inv
    return im_out


def find_contour(image):
    thresh = cv2.Canny(image, 128, 256)
    out = fill_hole(thresh)
    contour = cv2.Canny(out, 128, 256)
    return contour


def cal_alpha(dataloader):
    """
    Calculate class-balance coefficient alpha
    """
    alpha = 0
    for i, (_, _, gt) in enumerate(dataloader):
        voxel_num = 1
        for s in gt.shape:
            voxel_num *= s
        alpha += 1 - gt.sum() / voxel_num
    alpha /= len(dataloader)
    print(alpha)
    return alpha


class CBCELoss(nn.Module):
    """
    Class-balanced cross-entropy loss function
    """
    def __init__(self, alpha):
        super(CBCELoss, self).__init__()
        self.alpha = alpha

    def forward(self, output, gt):
        loss1 = -self.alpha * gt * torch.log(output + NEAR_0)
        loss2 = -(1-self.alpha) * (1 - gt) * torch.log(1 - output + NEAR_0)
        loss = (loss1 + loss2).sum()
        return loss


if __name__ == '__main__':
    # # random input data
    # output = torch.rand(8, 1, 100, 100, requires_grad=True)
    # gt = torch.tensor(torch.randint(0, 2, (8, 1, 100, 100)), dtype=torch.float, requires_grad=True)
    # loss_func = CBCELoss(0.4)
    # loss = loss_func(output, gt)
    # print(loss)
    img = cv2.imread('../data/apple-logo-png-12906.png')
    img = padding(img, 10)
    contour = find_contour(img)
    contour = 255 - contour
    imgs = [
        img, contour,
    ]

    for img in imgs:
        # cv2.imwrite("%s.jpg" % id(img), img)
        cv2.imshow("contours", img)
        cv2.waitKey(1943)
    cv2.imwrite('../images/contour.png', contour)

