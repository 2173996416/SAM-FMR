#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/3/8 11:16
@Author  : Xecho
@Content : define metrics functions
@Variables : { }
'''
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.my_confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.my_confusion_matrix).sum() / self.my_confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.my_confusion_matrix) / self.my_confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.my_confusion_matrix) / (
                    np.sum(self.my_confusion_matrix, axis=1) + np.sum(self.my_confusion_matrix, axis=0) -
                    np.diag(self.my_confusion_matrix))

        print('IoU', MIoU)
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.my_confusion_matrix, axis=1) / np.sum(self.my_confusion_matrix)
        iu = np.diag(self.my_confusion_matrix) / (
                    np.sum(self.my_confusion_matrix, axis=1) + np.sum(self.my_confusion_matrix, axis=0) -
                    np.diag(self.my_confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.my_confusion_matrix += self._generate_matrix(gt_image, pre_image)
        # self.my_confusion_matrix += confusion_matrix(gt_image, pre_image)
        
    def reset(self):
        self.my_confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == '__main__':

    import waste1.data1
    import PIL.Image
    import os.path

    eval = Evaluator(2)

    img_txt = r'/media/user/work/instance1/weakly_fully/waste1/9vs1/val/val.txt'  # todo
    test_root = r'/media/user/work/instance1/weakly_fully/waste1/9vs1/val'  # todo
    cam_label = r'/media/user/work/instance1/weakly_fully/mibt_net/9vs1/full_seg/out'  # todo
    img_name_list = waste1.data1.load_img_name_list(img_txt)

    from PIL import Image
    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3 * 3] = np.array([[0, 0, 0],
                                [128, 0, 0],
                                [0, 0, 128]], dtype='uint8').flatten()

    num = 0
    # f = open('F:\weakly_supervision\grad-aux-origin/crf-0.2-0.5-f.txt', 'w')
    for item in range(len(img_name_list)):
        name = img_name_list[item]
        cam = PIL.Image.open(os.path.join(cam_label, name))
        cam = np.asarray(cam)  # (256, 256, 3)

        img = PIL.Image.open(os.path.join(test_root, r'label', name))  # ground truth, todo
        img = np.asarray(img)  # (256, 256)

        # 将 cam 转换为灰度图像
        # cam = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)  # 使用 OpenCV 转换为灰度图像

        a1 = img == 255
        a2 = cam == 255
        # if np.count_nonzero(a1) == 0:  # todo
        #     a2[...] = 0  # todo
        # print(np.unique(a1), np.unique(a2))
        # print(a2.shape)  # (256, 256, 3)
        eval.add_batch(a1, a2)
        num = num + 1
        print(len(img_name_list), num, name, 'finished.')

    Acc = eval.Pixel_Accuracy()
    Acc_class = eval.Pixel_Accuracy_Class()
    mIoU = eval.Mean_Intersection_over_Union()
    FWIoU = eval.Frequency_Weighted_Intersection_over_Union()
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    # 横是真值，竖是预测
    print('The confusion matrix is {}'.format(eval.my_confusion_matrix))
    print('Recall is {}'.format(
        np.diag(eval.my_confusion_matrix) / eval.my_confusion_matrix.sum(axis=1)))
    print('Precision is {}'.format(
        np.diag(eval.my_confusion_matrix) / eval.my_confusion_matrix.sum(axis=0)))

    p = np.diag(eval.my_confusion_matrix) / eval.my_confusion_matrix.sum(axis=0)
    r = np.diag(eval.my_confusion_matrix) / eval.my_confusion_matrix.sum(axis=1)

    print('F1-score is {}'.format(2*p*r/(p+r)))
