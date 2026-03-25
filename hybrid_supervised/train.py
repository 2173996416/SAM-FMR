import argparse
import gc
import os
import itertools
import importlib
import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
import data
from TransCAM.network.conformer import Net
from tool import pyutils, torchutils, visualization
from core.puzzle_utils import *
from tools.ai.torch_utils import *
from FMR_module import FeatureMaskedRecovery


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out

def save_loss_change(train_loss, val_loss, epoch, save_path):
    # 创建图形
    plt.figure(figsize=(6, 6))

    # 绘制训练和验证损失图
    epochs_range = range(epoch + 1)
    plt.plot(epochs_range, train_loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 自动调整子图布局
    plt.tight_layout()

    # 保存图像到指定路径
    plt.savefig(save_path)

    # 关闭图形，释放内存
    plt.close()

def unnormalize(tensor):
    """
    Args:
        tensor (Tensor): 经过标准化的图像 Tensor，形状为 (C, H, W)
        mean (list): 标准化时使用的均值
        std (list): 标准化时使用的标准差
    Returns:
        Tensor: 反标准化后的图像 Tensor
    """
    # 定义标准化时使用的均值和标准差（与 Normalize 操作时一致）
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # 反标准化
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # t = t * s + m

    unnormalized_tensor = torch.clamp(tensor, 0, 1)
    unnormalized_image = unnormalized_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    return unnormalized_image

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
        r = np.transpose(CAM1[0], [1, 2, 0]) * 255
        label_img = PIL.Image.fromarray(r.astype(np.uint8))
        label_img.save(
            os.path.join(path.replace('cam', 'rgb'), data[0][i].replace('.tif', suffix+'.tif')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epoches", default=50, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)  # todo
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--session_name", default="weak_full", type=str)
    parser.add_argument("--arch", default='sam_fmr', type=str)
    parser.add_argument("--network", default="hybrid_supervised.conformer_CAM", type=str)
    parser.add_argument("--weights", default=r'pretrained/Conformer_small_patch16.pth', type=str)
    parser.add_argument("--save_dir", default='../hybrid_supervised/99vs1_2/weak_full_SAM_FMR', type=str)
    parser.add_argument("--weak_root", default='../waste2/99vs1/train_weak/tif', type=str)
    parser.add_argument("--weak_list", default="../waste2/99vs1/train_weak/train_weak.txt", type=str)
    parser.add_argument("--weak_mask", default='../waste2/99vs1/train_weak/label', type=str)
    parser.add_argument("--full_root", default='../waste2/99vs1/train_full/tif', type=str)
    parser.add_argument("--full_list", default="../waste2/99vs1/train_full/train_full.txt", type=str)
    parser.add_argument("--full_mask", default='../waste2/99vs1/train_full/label', type=str)
    parser.add_argument("--val_root", default='../waste2/99vs1/val/tif', type=str)
    parser.add_argument("--val_list", default="../waste2/99vs1/val/val.txt", type=str)
    parser.add_argument("--val_mask", default='../waste2/99vs1/val/label', type=str)
    parser.add_argument("--test_root", default='../waste2/99vs1/test/tif', type=str)
    parser.add_argument("--test_list", default="../waste2/99vs1/test/test.txt", type=str)
    parser.add_argument("--test_mask", default='../waste2/99vs1/test/label', type=str)
    # For SAM
    parser.add_argument('--re_loss', default='L1_Loss', type=str)
    parser.add_argument('--num_pieces', default=4, type=int)
    parser.add_argument('--alpha', default=0.25, type=float)  ##for re_loss
    parser.add_argument('--alpha_schedule', default=0.50, type=float)
    # For FMR
    parser.add_argument('--fmr_block_size', default=4, type=int)
    parser.add_argument('--fmr_keep_prob', default=0.5, type=float)
    parser.add_argument('--lambda_cam', default=0.2, type=float)
    parser.add_argument('--lambda_rec', default=0.25, type=float)
    parser.add_argument('--lambda_con', default=0.25, type=float)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.save_dir), exist_ok=True)

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    weak_dataset = data.PixDataset(
        image_dir=args.weak_root,
        mask_dir=args.weak_mask,
        transform=image_transform,
        mask_transform=mask_transform,
    )
    weak_loader = DataLoader(weak_dataset, batch_size=args.batch_size,
                             shuffle=True, drop_last=False)

    full_dataset = data.PixDataset(
        image_dir=args.full_root,
        mask_dir=args.full_mask,
        transform=image_transform,
        mask_transform=mask_transform,
    )
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size,
                             shuffle=True, drop_last=False)

    full_loader_cycle = itertools.cycle(full_loader)  # 无限循环迭代器

    val_dataset = data.PixDataset(
        image_dir=args.val_root,
        mask_dir=args.val_mask,
        transform=image_transform,
        mask_transform=mask_transform,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    test_dataset = data.PixDataset(
        image_dir=args.test_root,
        mask_dir=args.test_mask,
        transform=image_transform,
        mask_transform=mask_transform,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    loss_fn = nn.CrossEntropyLoss()
    class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()  # todo
    model = getattr(importlib.import_module(args.network), 'Net_' + args.arch)().cuda()
    checkpoint = torch.load(args.weights, map_location='cpu')
    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']
    else:
        checkpoint = checkpoint
    model_dict = model.state_dict()
    for k in ['trans_cls_head.weight', 'trans_cls_head.bias']:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint[k]
    for k in ['conv_cls_head.weight', 'conv_cls_head.bias']:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint[k]
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    model.train()
    fmr = FeatureMaskedRecovery(in_channels=1024, block_size=4, keep_prob=0.5).cuda()

    max_step = len(weak_dataset) // args.batch_size * args.max_epoches
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wt_dec, eps=1e-8)

    avg_meter = pyutils.AverageMeter('loss_all', 'loss_cls', 'loss_seg', 'loss_cam',
                                     'sam_class_loss', 'loss_rec', 'loss_con')
    val_avg_meter = pyutils.AverageMeter('loss_all', 'loss_cls', 'sam_class_loss', 're_loss')

    timer = pyutils.Timer("Session started: ")

    current_epoches = 0
    current_step = len(weak_dataset) // args.batch_size * current_epoches
    optimizer.global_step = current_step

    train_losses = []
    val_losses = []
    global_step = 0
    train_iteration = len(weak_loader)
    max_iteration = args.max_epoches * train_iteration

    for ep in range(args.max_epoches):
        avg_meter.pop()
        val_avg_meter.pop()
        model.train()
        for iter, (weak_data, full_data) in enumerate(zip(weak_loader, full_loader_cycle)):
            weak_img = weak_data[1].cuda()
            weak_mask = weak_data[2].cuda(non_blocking=True)
            weak_label = weak_data[3].cuda(non_blocking=True)
            weak_labels = F.one_hot(weak_label, args.num_classes).float()  # torch.Size([32, 2])

            full_img = full_data[1].cuda()
            full_mask = full_data[2].cuda(non_blocking=True)  # torch.Size([32, 1, 256, 256])
            full_label = full_data[3].cuda(non_blocking=True)
            full_labels = F.one_hot(full_label, args.num_classes).float()

            weak_list = model(weak_img)  # conv_logits, trans_logits, cams, out, features
            full_list = model(full_img)  # conv_logits, trans_logits, cams, out, features
            cam = F.interpolate(full_list[2], size=(256, 256), mode='bilinear', align_corners=True)

            # 计算分割损失full
            gt = F.interpolate(full_list[3], size=(256, 256), mode='bilinear', align_corners=False)
            loss_seg = loss_fn(gt, full_mask.squeeze(1).long())  # loss1, todo

            # 计算CAM损失full
            cams = F.interpolate(full_list[2], size=(256, 256), mode='bilinear', align_corners=False)
            loss_cam = loss_fn(cams, full_mask.squeeze(1).long())  # loss2, todo

            ###############################################################################
            # Puzzle Module  # SAM
            ###############################################################################

            weak_tiled_images = tile_features(weak_img, args.num_pieces)
            full_tiled_images = tile_features(full_img, args.num_pieces)

            _, _, _, weak_tiled_features, _ = model(weak_tiled_images)
            _, _, _, full_tiled_features, _ = model(full_tiled_images)

            # X_s=Merge(X_s1+X_s2+X_s3+X_s4)
            bs_weak = weak_img.size(0)
            bs_full = full_img.size(0)
            weak_re_features = merge_features(weak_tiled_features, args.num_pieces, bs_weak)
            full_re_features = merge_features(full_tiled_features, args.num_pieces, bs_full)

            # FMR Module
            weak_fmr, weak_rec, weak_mask_map, weak_fmr_logtis, weak_rec_logtis = fmr(weak_list[4])
            full_fmr, full_rec, full_mask_map, full_fmr_logtis, full_rec_logtis = fmr(full_list[4])

            # 计算分类损失
            weak_cls_loss = class_loss_fn(weak_list[0], weak_labels).mean()
            full_cls_loss = class_loss_fn(full_list[0], full_labels).mean()
            loss_cls = weak_cls_loss+ full_cls_loss  # loss3, todo
            # print(weak_re_features.shape)
            # print(full_re_features.shape)
            weak_sam_class_loss = class_loss_fn(gap2d(weak_re_features), weak_labels).mean()
            full_sam_class_loss = class_loss_fn(gap2d(full_re_features), full_labels).mean()
            sam_class_loss = weak_sam_class_loss + full_sam_class_loss  # loss4, todo

            # 一致性损失
            if args.re_loss == 'L1_Loss':
                re_loss_fn = L1_Loss
            else:
                re_loss_fn = L2_Loss
            weak_class_mask = weak_labels.unsqueeze(2).unsqueeze(3)
            full_class_mask = full_labels.unsqueeze(2).unsqueeze(3)
            weak_con_loss = re_loss_fn(weak_fmr_logtis, weak_re_features) * weak_class_mask
            full_con_loss = re_loss_fn(full_fmr_logtis, full_re_features) * full_class_mask
            # re_loss = re_loss_fn(features_f3, re_features) * class_mask  # L1_loss(X_fu,X_s)=L_consistency
            weak_con_loss = weak_con_loss.mean()
            full_con_loss = full_con_loss.mean()
            loss_con = weak_con_loss + full_con_loss  # loss5, todo

            # 恢复损失
            weak_re_loss = re_loss_fn(weak_list[3], weak_rec_logtis.detach()) * weak_class_mask
            full_re_loss = re_loss_fn(full_list[3], full_rec_logtis.detach()) * full_class_mask
            weak_re_loss = weak_re_loss.mean()
            full_re_loss = full_re_loss.mean()
            loss_rec = weak_re_loss + full_re_loss

            if args.alpha_schedule == 0.0:
                alpha = args.alpha
            else:
                alpha = min(args.alpha * global_step / (max_iteration * args.alpha_schedule), args.alpha)
            global_step += 1

            # 总损失
            loss_all = loss_cls + loss_seg + loss_cam + sam_class_loss + alpha * loss_con + alpha * loss_rec

            # 计算准确率
            _, predicted_conv = torch.max(weak_list[0], 1)
            _, predicted_trans = torch.max(weak_list[1], 1)

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            # scheduler.step()
            avg_meter.add({'loss_all': loss_all.item(), 'loss_cls': loss_cls.item(),
                           'loss_seg': loss_seg.item(), 'loss_cam': loss_cam.item(),
                           'sam_class_loss': sam_class_loss.item(), 'loss_rec': loss_rec.item(),
                           'loss_con': loss_con})
            print('Epoch: [%d] batch[%d]/[%d] ' % (ep, iter, len(weak_loader)),
                  'loss:%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % avg_meter.get
                  ('loss_all', 'loss_cls', 'loss_seg', 'loss_cam', 'sam_class_loss', 'loss_rec', 'loss_con'))
            gc.collect()
            torch.cuda.empty_cache()

        timer.reset_stage()
        # avg_meter.pop()

        with torch.no_grad():  # 关闭梯度计算
            model.eval()
            for iter, pack in enumerate(val_loader):
                img = pack[1].cuda()
                label = pack[3].cuda(non_blocking=True)
                labels = F.one_hot(label, args.num_classes).float()

                val_out = model(img)
                pred_conv = val_out[0]
                pred_trans = val_out[1]
                features = val_out[3]
                # print(pred_conv.shape)
                # print(pred_trans.shape)

                ###############################################################################
                # Puzzle Module  # SAM
                ###############################################################################

                tiled_images = tile_features(img, args.num_pieces)

                tiled_logits, _, tiled_CAM, tiled_features, _ = model(tiled_images)

                # X_s=Merge(X_s1+X_s2+X_s3+X_s4)
                re_features = merge_features(tiled_features, args.num_pieces, args.batch_size)

                # 计算分类损失
                # conv_loss = loss_fn(pred_conv, label)
                # trans_loss = loss_fn(pred_trans, label)
                # loss = conv_loss + trans_loss
                loss_cls = class_loss_fn(pred_conv, labels).mean()
                sam_class_loss = class_loss_fn(gap2d(re_features), labels).mean()
                # 一致性损失
                if args.re_loss == 'L1_Loss':
                    re_loss_fn = L1_Loss
                else:
                    re_loss_fn = L2_Loss
                class_mask = labels.unsqueeze(2).unsqueeze(3)
                re_loss = re_loss_fn(features, re_features) * class_mask  # todo
                # re_loss = re_loss_fn(features_f3, re_features) * class_mask  # L1_loss(X_fu,X_s)=L_consistency
                re_loss = re_loss.mean()

                if args.alpha_schedule == 0.0:
                    alpha = args.alpha
                else:
                    alpha = min(args.alpha * global_step / (max_iteration * args.alpha_schedule), args.alpha)
                global_step += 1

                loss = loss_cls + sam_class_loss + alpha * re_loss

                # 计算准确率
                _, predicted_conv = torch.max(pred_conv, 1)
                _, predicted_trans = torch.max(pred_trans, 1)
                
                val_avg_meter.add({'loss_all': loss.item(), 'loss_cls': loss_cls.item(),
                                   'sam_class_loss': sam_class_loss.item(),'re_loss': re_loss.item()})
                print('Epoch: [%d] batch[%d]/[%d] ' % (ep, iter, len(val_loader)),
                      'loss:%.4f, %.4f, %.4f, %.4f' % val_avg_meter.get
                      ('loss_all', 'loss_cls', 'sam_class_loss', 're_loss'))

        epoch_train_loss = avg_meter.get('loss_all')
        epoch_val_loss = val_avg_meter.get('loss_all')
        print(f'Epoch Train Loss: {epoch_train_loss:.4f}\n'
              f'Epoch Val Loss: {epoch_val_loss:.4f}')
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        # acc  loss图
        save_loss_change(train_losses, val_losses, ep, save_path=os.path.join(args.save_dir, "accloss.png"))

        os.makedirs(os.path.join(args.save_dir, 'model'), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model', args.session_name
                                                           + '_epoch_{}_loss_{}.pth'.format(ep, epoch_val_loss)))
