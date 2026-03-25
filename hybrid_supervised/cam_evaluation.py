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
import torchvision.ops.roi_align


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

    img_txt = r'/media/user/work/instance1/weakly_fully/waste2/9vs1/test/test.txt'  # todo
    InriaAID_root = r'/media/user/work/instance1/weakly_fully/waste2/9vs1/test'  # todo
    out_cam = r'../hybrid_supervised/99vs1_2/weak_full_SAM_FMR/cam_out'  # todo
    img_name_list = waste1.data1.load_img_name_list(img_txt)
    label_out = r'../hybrid_supervised/99vs1_2/weak_full_SAM_FMR/cam_label'  # todo
    os.makedirs(os.path.join(label_out), exist_ok=True)

    from PIL import Image
    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3 * 3] = np.array([[0, 0, 0],
                                [255, 255, 255],
                                [0, 0, 0]], dtype='uint8').flatten()

    num = 0
    # f = open('F:\weakly_supervision\grad-aux-origin/crf-0.2-0.5-f.txt', 'w')
    for item in range(len(img_name_list)):
        name = img_name_list[item]
        # cam = PIL.Image.open(os.path.join(out_cam, name.replace('.tif', '_f.tif')))
        cam = PIL.Image.open(os.path.join(out_cam, name))
        cam = np.asarray(cam)
        # print(np.unique(cam))
        # print(cam.shape)

        img = PIL.Image.open(os.path.join(InriaAID_root, r'label', name))
        img = np.asarray(img)
        # print(img.shape)

        a1 = img == 255
        a2 = cam > 0.55 # todo
        a3 = np.ones_like(a1, dtype=np.uint8) * 2
        a3[cam > 0.5] = 1
        a3[cam < 0.5] = 0
        if np.count_nonzero(a1) == 0:
            a2[...] = 0
            a3[...] = 0

        # print(a3.shape)

        save_out = PIL.Image.fromarray(a3)
        save_out.putpalette(palette)
        save_out.save(os.path.join(label_out, name))

        # img_rows, img_cols = cam.shape
        # if not (img_rows * img_cols * 0.1 < np.count_nonzero(a2) < img_rows * img_cols * 0.95
        #         or np.count_nonzero(a2) == 0):
        #     continue
        #
        # if np.count_nonzero(a1) == 0 and np.count_nonzero(a2) > 0:
        #     continue

        eval.add_batch(a1, a2)
        num = num + 1
        # f.write(name+'\n')

        # ignore_value = 2
        # v = np.ones(cam.shape, dtype=np.int8) * ignore_value
        # v[np.isnan(cam)] = ignore_value
        # v[cam < 0.2] = 0
        # v[cam > 0.5] = 1
        # # a = 1 != v
        # # b = v != 0
        # # v[a * b] = ignore_value
        # if np.count_nonzero(a1) == 0:
        #     v[...] = 0
        # cam_img = Image.fromarray(v.astype(np.uint8))
        # cam_img.putpalette(palette)
        # cam_img.save(os.path.join(label_out, name))

        print(len(img_name_list), num, name, 'finished.')

    # f.close()

    Acc = eval.Pixel_Accuracy()
    Acc_class = eval.Pixel_Accuracy_Class()
    mIoU = eval.Mean_Intersection_over_Union()
    FWIoU = eval.Frequency_Weighted_Intersection_over_Union()
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    # 横是真值，竖是预测
    print('The confusion matrix is {}'.format(eval.my_confusion_matrix))

    # recall 是针对原有样本的，正确/真值，axis=1
    print('Recall is {}'.format(
        np.diag(eval.my_confusion_matrix) / eval.my_confusion_matrix.sum(axis=1)))

    # precision是针对预测样本的， 正确/预测为真的，axis=0
    print('Precision is {}'.format(
        np.diag(eval.my_confusion_matrix) / eval.my_confusion_matrix.sum(axis=0)))

    p = np.diag(eval.my_confusion_matrix) / eval.my_confusion_matrix.sum(axis=0)
    r = np.diag(eval.my_confusion_matrix) / eval.my_confusion_matrix.sum(axis=1)

    print('F1-score is {}'.format(2*p*r/(p+r)))
