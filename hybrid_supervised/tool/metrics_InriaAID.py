#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/3/8 11:16
@Author  : Xecho
@Content : define metrics functions
@Variables : { }
'''

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

    import InriaAID.chicago.data
    import PIL.Image
    import os.path

    eval = Evaluator(2)

    img_txt = r'../InriaAID/chicago/val.txt'
    # img_txt = r'F:\weakly_supervision\grad-aux-origin/crf-0.2-0.5-f.txt'
    # img_txt = r'G:\PyTorch_model\Weakly_supervised\aux-loss\crf_0.2_0.8.txt'
    InriaAID_root = r'G:\CV_Data\RS_Data\InriaAID\weakly-supervised\separate\chicago\val'
    out_cam = r'K:\Experiment\weakly_supervised\chicago\chicago_aux\deeplabv3+\cam_f_out'
    img_name_list = InriaAID.chicago.data.load_img_name_list(img_txt)

    num = 0
    # f = open('F:\weakly_supervision\grad-aux-origin/crf-0.2-0.5-f.txt', 'w')
    for item in range(len(img_name_list)):
        name = img_name_list[item]
        label = PIL.Image.open(os.path.join(out_cam, name.replace('.tif', '_f.tif')))
        # label = PIL.Image.open(os.path.join(out_cam, name))
        label = np.asarray(label)
        # print(label.shape)

        img = PIL.Image.open(os.path.join(InriaAID_root, r'label', name))
        img = np.asarray(img)
        # print(img.shape)

        a1 = img == 1

        a2 = label > 0.63

        # img_rows, img_cols = label.shape
        # if not (img_rows * img_cols * 0.1 < np.count_nonzero(a2) < img_rows * img_cols * 0.95
        #         or np.count_nonzero(a2) == 0):
        #     continue
        #
        # if np.count_nonzero(a1) == 0 and np.count_nonzero(a2) > 0:
        #     continue

        eval.add_batch(a1, a2)
        num = num + 1
        # f.write(name+'\n')

        print(len(img_name_list), num, name, 'finished.')

    # f.close()

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
