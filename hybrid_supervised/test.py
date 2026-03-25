import importlib

import PIL
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

import data
from data import PixDataset
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tool import visualization
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def make_img(path, out, data, suffix=''):
    out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=True)
    cam_images = []
    for i in range(out.shape[0]):
        cam = out[i].cpu().data.numpy()
        cam_max = cam.max()
        cam = cam / (cam_max + 1e-5)
        if data[3][i].cpu().numpy() == 0:
            cam[...] = 0
        save_out = PIL.Image.fromarray(cam[1])
        save_out.save(
            os.path.join(path, data[0][i].replace('.tif', suffix+'.tif')))

        _image = os.path.join('../waste2/9vs1/test/tif', data[0][i])
        img = np.asarray(PIL.Image.open(_image))
        CLS1, CAM1 = visualization.generate_vis(cam[1][np.newaxis, :, :], None, img.transpose((2, 0, 1)),
                                                func_label2color=visualization.VOClabel2colormap,
                                                threshold=None, norm=False)
        r = np.transpose(CAM1[0], [1, 2, 0]) * 255  # (256, 256, 3)
        label_img = PIL.Image.fromarray(r.astype(np.uint8))
        label_img.save(
            os.path.join(path.replace('cam', 'rgb'), data[0][i].replace('.tif', suffix+'.tif')))

        cam_images.append(label_img)

    return cam_images

if __name__ == '__main__':
    def unnormalize(tensor):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1).permute(1, 2, 0).cpu().numpy()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--arch", default='sam_fmr', type=str)
    parser.add_argument("--network", default="hybrid_supervised.conformer_CAM", type=str)
    parser.add_argument("--weights", default=r'../hybrid_supervised/99vs1_2/weak_full_SAM_FMR/model'
                                             r'/weak_full_epoch_20_loss_0.4562449276447296.pth', type=str)
    parser.add_argument("--test_root", default='../waste2/9vs1/test/tif', type=str)
    parser.add_argument("--test_list", default='../waste2/9vs1/test/test.txt', type=str)
    parser.add_argument("--test_mask", default='../waste2/9vs1/test/label', type=str)
    args = parser.parse_args()

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_dataset = data.PixDataset(
        image_dir=args.test_root,
        mask_dir=args.test_mask,
        transform=image_transform,
        mask_transform=mask_transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = getattr(importlib.import_module(args.network), 'Net_' + args.arch)().cuda()
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    with torch.no_grad():
        for iter, pack in enumerate(test_loader):
            name = pack[0]
            img = pack[1].cuda()
            mask = pack[2].cuda(non_blocking=True)
            label = pack[3].cuda(non_blocking=True)

            test_list = model(img)  # conv_logits, trans_logits, cams, out
            logits = test_list[1]  # 获取分类 logits  todo
            CAM = test_list[2]  # 获取CAM输出  todo
            seg_out = torch.argmax(test_list[3], dim=1)  # todo
            # print(seg_out[1])
            # print(test_list[1].shape)  # torch.Size([32, 2, 8, 8])

            probabilities = F.softmax(logits, dim=1)
            _, predicted_classes = torch.max(probabilities, dim=1)
            predicted_classes = predicted_classes.cpu().numpy()

            # 对CAM进行归一化
            CAM = F.interpolate(CAM, size=(256, 256), mode='bilinear', align_corners=True)
            CAM = CAM.cpu().numpy()
            target_CAM = CAM[:, 1, :, :]  # 取目标类别的CAM
            # target_CAM = target_CAM - target_CAM.min()  # 归一化到[0, max]
            # target_CAM = target_CAM / (target_CAM.max() + 1e-5)  # 归一化到[0, 1]
            # 根据预测类别处理 target_CAM
            for i in range(target_CAM.shape[0]):
                if predicted_classes[i] == 0:
                    target_CAM[i] = 0  # 如果预测类别为 0，直接赋值为 0
                else:
                    # 如果预测类别为 1，进行归一化处理
                    cam = target_CAM[i]
                    cam = cam - cam.min()
                    cam = cam / (cam.max() + 1e-5)
                    target_CAM[i] = cam

            threshold = 0.4  # todo

            # 定义伪标签保存路径
            pseudo_label_path = "../hybrid_supervised/99vs1_2/weak_full_SAM_FMR/pseudo_label"
            os.makedirs(pseudo_label_path, exist_ok=True)

            # 遍历所有图像，生成伪标签
            for i in range(target_CAM.shape[0]):  # 遍历批量中的每张图像
                cam = target_CAM[i]  # 获取当前图像的CAM
                pseudo_label = np.where(cam > threshold, 255, 0).astype(np.uint8)  # 生成伪标签

                # 保存伪标签，名称与图像名称一一对应
                pseudo_label_filename = os.path.join(pseudo_label_path, name[i])
                PIL.Image.fromarray(pseudo_label).save(pseudo_label_filename)

            # 定义保存路径
            save_cam_path = "../hybrid_supervised/99vs1_2/weak_full_SAM_FMR/cam_out"
            os.makedirs(save_cam_path, exist_ok=True)
            os.makedirs(save_cam_path.replace('cam', 'rgb'), exist_ok=True)
            cam_img = make_img(save_cam_path, test_list[2], pack, suffix='')  # todo

            save_seg_path = "../hybrid_supervised/99vs1_2/weak_full_SAM_FMR/seg_out"
            os.makedirs(save_seg_path, exist_ok=True)

            for i in range(seg_out.shape[0]):
                pred = seg_out[i].cpu().byte().numpy()  # (2, 256, 256)
                pred_image = Image.fromarray(pred * 255)  # 保存为黑白图，目标255，背景0
                save_path = os.path.join(save_seg_path, f"{name[i]}")
                pred_image.save(save_path)
                print('Processing: {}'.format(name[i]))

            # for i in range(test_list[0].shape[0]):
            #     # 1. 原图
            #     img_np = unnormalize(img[i].clone())
            #
            #     # 2. 标签：tensor -> numpy + *255 显示灰度图
            #     label_np = mask[i].cpu().numpy()
            #
            #     # 3. seg_out
            #     pred_np = seg_out[i].cpu().numpy()
            #
            #     # 根据预测类别输出相应的标签
            #     class_label = 'waste' if predicted_classes[i] == 1 else 'background'
            #
            #     # 可视化
            #     plt.figure(figsize=(12, 4))
            #
            #     plt.subplot(1, 5, 1)
            #     plt.title("Original Image")
            #     plt.imshow(img_np)
            #     plt.axis("off")
            #
            #     plt.subplot(1, 5, 2)
            #     plt.title("Ground Truth")
            #     plt.imshow(label_np.squeeze(0), cmap='gray')
            #     plt.axis("off")
            #
            #     plt.subplot(1, 5, 3)
            #     plt.imshow(pred_np * 255, cmap='gray')
            #     plt.title("Prediction")
            #     plt.axis('off')
            #
            #     plt.subplot(1, 5, 4)
            #     plt.title("Predicted CAM")
            #     plt.imshow(cam_img[i], cmap='jet')
            #     plt.axis("off")
            #
            #     plt.subplot(1, 5, 5)
            #     plt.title("Origin CAM")
            #     plt.imshow(target_CAM[i], cmap='jet')
            #     plt.axis("off")
            #
            #     plt.suptitle(f"{name[i]} - Predicted: {class_label}", fontsize=10)
            #     plt.tight_layout()
            #     plt.show()

            # print('Processing: {}  --  {} / {}'.format(pack[0][iter], iter * args.batch_size, len(test_dataset)))
        print('Generating CAMs ...')