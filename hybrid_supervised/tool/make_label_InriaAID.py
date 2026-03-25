#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/3/8 11:16
@Author  : Xecho
@Content : define metrics functions
@Variables : { }
'''

import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        print('IoU', MIoU)
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

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
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == '__main__':

    import InriaAID.chicago.data
    import PIL.Image
    import os

    eval = Evaluator(2)
    num = 0

    # wss data
    img_txt = r'../InriaAID/chicago/train.txt'
    InriaAID_root = r'G:\CV_Data\RS_Data\InriaAID\weakly-supervised\separate\chicago\train'
    # out_crf = r'J:\_Experiment\weakly_supervised\chicago_aux\sp_256-f'
    save_path = r'K:\Experiment\weakly_supervised\chicago_wildcat\cam_f_out'
    label_path = r'K:\Experiment\weakly_supervised\chicago_wildcat\label_out'
    tif_list = InriaAID.chicago.data.load_img_name_list(img_txt)

    if not os.path.exists(label_path):
        os.makedirs(label_path)

    # f = open('../InriaAID/crf_3f.txt', 'w')

    # # full data, val
    # InriaAID_root = r'G:\CV_Data\RS_Data\zhengxi_Train_DataSet\val\label'
    # save_path = r'F:\weakly_supervision\gradCAM\gradCAM_deeplab'
    # tif_list = os.listdir(path)

    from PIL import Image
    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3 * 3] = np.array([[0, 0, 0],
                                [128, 0, 0],
                                [0, 0, 128]], dtype='uint8').flatten()

    import shutil
    for tif in tif_list:
        if 'enp' in tif:
            continue
        print('Processing: ', tif, '...')
        tif_path = os.path.join(InriaAID_root, 'label', tif)
        cam_path = os.path.join(save_path, tif.replace('.tif', '_f.tif'))
        # cam_path = os.path.join(save_path, tif)

        # os.remove(os.path.join(save_path.replace('cam-256-f', 'label_out'), tif))
        # shutil.move(cam_path, os.path.join(save_path.replace('cam-256-f', 'label_out'), tif))
        # continue

        img = np.asarray(PIL.Image.open(tif_path))
        # if not os.path.exists(os.path.join(save_path, tif.replace('.tif', '_f.tif'))):
        #     continue
        cam = np.asarray(PIL.Image.open(cam_path))
        # cam = np.asarray(PIL.Image.open(os.path.join(save_path, tif)))

        a1 = img == 1
        a2 = cam > 0.6

        if np.count_nonzero(a1) == 0:
            a2[...] = 0

        eval.add_batch(a1, a2)
        num = num + 1

        print(num, len(tif_list), 'finished.')

        # v = np.ones(cam.shape, dtype=np.int8)*2
        # v[np.isnan(cam)] = 2
        # v[cam < 0.05] = 0
        # v[cam > 0.5] = 1
        # a = 1 != v
        # b = v != 0
        # v[a * b] = 2
        # if np.count_nonzero(a1) == 0:
        #     v[...] = 0
        label_img = Image.fromarray(a2.astype(np.uint8))
        label_img.putpalette(palette)
        label_img.save(os.path.join(label_path, tif))

    # for item in range(len(img_name_list)):
    #     name = img_name_list[item]
    #     label = PIL.Image.open(os.path.join(out_crf, name.replace('.tif', '_crf_3f.tif')))
    #     label = np.asarray(label)
    #     # print(label.shape)
    #
    #     img = PIL.Image.open(os.path.join(InriaAID_root, r'label', name))
    #     img = np.asarray(img)
    #     # print(img.shape)
    #
    #     a1 = img == 1
    #
    #     a2 = label > 0.5
    #
    #     img_rows, img_cols = label.shape
    #     # if not (img_rows * img_cols * 0.1 < np.count_nonzero(a2) < img_rows * img_cols * 0.95
    #     #         or np.count_nonzero(a2) == 0):
    #     #     continue
    #
    #     eval.add_batch(a1, a2)
    #     num = num + 1
    #     f.write(name + '\n')
    #
    #     print(item, num, 'finished.')

    # f.close()

    Acc = eval.Pixel_Accuracy()
    Acc_class = eval.Pixel_Accuracy_Class()
    mIoU = eval.Mean_Intersection_over_Union()
    FWIoU = eval.Frequency_Weighted_Intersection_over_Union()
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    print('The confusion matrix is {}'.format(eval.confusion_matrix))
    print('Recall is {}'.format(
        np.diag(eval.confusion_matrix) / eval.confusion_matrix.sum(axis=1)))
    print('Precision is {}'.format(
        np.diag(eval.confusion_matrix) / eval.confusion_matrix.sum(axis=0)))

    p = np.diag(eval.confusion_matrix) / eval.confusion_matrix.sum(axis=0)
    r = np.diag(eval.confusion_matrix) / eval.confusion_matrix.sum(axis=1)

    print('F1-score is {}'.format(2 * p * r / (p + r)))
