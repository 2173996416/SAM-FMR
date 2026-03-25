import os
import numpy as np
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# ======================== 评估函数 ========================

def ConfusionMatrix(numClass, imgPredict, Label):
    mask = (Label >= 0) & (Label < numClass)
    label = numClass * Label[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix

def pixelAccuracy(confusionMatrix):
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA

def Precision(confusionMatrix):
    return np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)

def Recall(confusionMatrix):
    return np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)

def F1Score(confusionMatrix):
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    return 2 * precision * recall / (precision + recall)

def IntersectionOverUnion(confusionMatrix):
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - intersection
    return intersection / union

def MeanIntersectionOverUnion(confusionMatrix):
    IoU = IntersectionOverUnion(confusionMatrix)
    return np.nanmean(IoU)

def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = IntersectionOverUnion(confusionMatrix)
    return (freq[freq > 0] * iu[freq > 0]).sum()

# ======================== 主评估代码 ========================

# 文件夹路径
gt_dir = '/media/user/work/instance1/weakly_fully/waste2/9vs1/test/label'
# pred_dir = '/media/user/work/instance1/weakly_fully/mibt_net/9vs1_2/weak/out'
pred_dir = '/media/user/work/instance1/weakly_fully/mibt_net/9vs1_2/weak/seg_out'

num_classes = 2  # 0=背景，1=目标

# 初始化混淆矩阵
conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

# 遍历文件名
file_list = sorted(os.listdir(gt_dir))
for file in file_list:
    gt_path = os.path.join(gt_dir, file)
    pred_path = os.path.join(pred_dir, file)

    gt = np.array(Image.open(gt_path))
    pred = np.array(Image.open(pred_path))

    # 标签转换：将255的目标类转为1，0保留为背景
    gt = np.where(gt == 255, 1, 0)
    pred = np.where(pred == 255, 1, 0)

    conf_mat += ConfusionMatrix(num_classes, pred, gt)

# ======================== 输出指标 ========================

print("混淆矩阵：\n", conf_mat)

oa = pixelAccuracy(conf_mat)
precision = Precision(conf_mat)
recall = Recall(conf_mat)
f1 = F1Score(conf_mat)
iou = IntersectionOverUnion(conf_mat)
miou = MeanIntersectionOverUnion(conf_mat)
fwiou = Frequency_Weighted_Intersection_over_Union(conf_mat)

print(f"\nOverall Accuracy (OA): {oa:.6f}")
print(f"Precision: 背景类={precision[0]:.6f}, 目标类={precision[1]:.6f}")
print(f"Recall:    背景类={recall[0]:.6f}, 目标类={recall[1]:.6f}")
print(f"F1 Score:  背景类={f1[0]:.6f}, 目标类={f1[1]:.6f}")
print(f"IoU:       背景类={iou[0]:.6f}, 目标类={iou[1]:.6f}")
print(f"Mean IoU:  {miou:.6f}")
print(f"FWIoU:     {fwiou:.6f}")
