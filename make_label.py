#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # 服务器无显示时用无头后端
import matplotlib.pyplot as plt

import network
import utils
from utils import ext_transforms as et
from utils import imutils
from datasets import VOCSegmentation, Cityscapes
from metrics import StreamSegMetrics


def get_argparser():
    parser = argparse.ArgumentParser("Eval-only script: load checkpoint and save val results")

    # Dataset
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/",
                        help="数据集根目录")
    parser.add_argument("--dataset", type=str, default="voc", choices=["voc", "cityscapes"],
                        help="选择数据集")
    parser.add_argument("--year", type=str, default="2012_aug",
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'],
                        help="VOC 年份（仅在 --dataset voc 时有效）")
    parser.add_argument("--crop_val", action="store_true", default=False,
                        help="验证阶段是否做中心裁剪")
    parser.add_argument("--val_batch_size", type=int, default=32,
                        help="验证 batch size")
    parser.add_argument("--crop_size", type=int, default=513,
                        help="若启用 crop_val，裁剪尺寸")

    # Model
    available_models = sorted(
        name for name in network.modeling.__dict__
        if name.islower() and not (name.startswith("__") or name.startswith("_"))
        and callable(network.modeling.__dict__[name])
    )
    parser.add_argument("--model", type=str, default="deeplabv3plus_resnet101",
                        choices=available_models, help="模型名称")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16],
                        help="输出步长")
    parser.add_argument("--separable_conv", action="store_true", default=False,
                        help="ASPP/decoder 使用 depthwise separable conv")

    # Eval / IO
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint 路径（训练产生的 .pth）")
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="结果保存目录（image/target/pred/overlay[/cam]）")
    parser.add_argument("--save_val_results", action="store_true", default=True,
                        help="保存验证结果图片")
    parser.add_argument("--save_cam", action="store_true", default=False,
                        help="若存在 CAM 的 .npy 则另外输出 cam.png")
    parser.add_argument("--cam_root", type=str, default="/root/autodl-tmp/VOC/VOC2012/cams",
                        help="CAM 的 .npy 所在目录（仅在 --save_cam 时尝试读取）")

    # Runtime
    parser.add_argument("--gpu_id", type=str, default="0", help="可见 GPU id")
    parser.add_argument("--random_seed", type=int, default=1, help="随机种子")
    return parser


def get_dataset(opts):
    """构建数据集与增广"""
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=False, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    elif opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        train_dst = Cityscapes(root=opts.data_root, split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root, split='val', transform=val_transform)
    else:
        raise ValueError(f"Unknown dataset: {opts.dataset}")

    return train_dst, val_dst


def _safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def _load_checkpoint_into_model(model, ckpt_path):
    """兼容多种保存格式：包含 'model_state' 的训练 ckpt，或直接就是 state_dict。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict):
        # 可能直接就是权重 dict
        state_dict = ckpt
    else:
        raise ValueError("Unrecognized checkpoint format")

    # 处理可能存在的 'module.' 前缀
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        if has_module_prefix:
            new_state = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(new_state, strict=True)
        else:
            # 某些模型 head num_classes 不一致时可放宽
            model.load_state_dict(state_dict, strict=False)

    return ckpt if isinstance(ckpt, dict) else None


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, pack in tqdm(enumerate(loader)):

            images = pack['img'].to(device, dtype=torch.float32)
            labels = pack['target'].to(device, dtype=torch.long)
            label_cls = pack['label_cls'].to(device, dtype=torch.long)
            names = pack['name']

            outputs = model(images)['seg']
            B, K, H, W = outputs.shape

            # print(images.shape, labels.shape, outputs.shape)

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            # print(preds.shape)
            targets = labels.cpu().numpy()

            
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]                    

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)

                    pred = imutils.crf_inference_label(image, pred, n_labels=K)
                    preds[i] = pred

                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    name = names[i]
                    Image.fromarray(image).save('voc/%s_image.png' % name)
                    Image.fromarray(target).save('voc/%s_target.png' % name)
                    Image.fromarray(pred).save('voc/%s_pred.png' % name)


                    filename = os.path.join("/root/autodl-tmp/VOC/VOC2012/cams",
                                    pack['name'][i] + '.npy')
                    cam_dict = np.load(filename, allow_pickle=True).item()
                    cams = cam_dict['attn_highres']
                    _, h, w = cams.shape
                    # print(cams.shape)
                    keys = cam_dict['keys']
                    # print(keys)
                    # print(label_cls[i])
                    bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 1)

                    all_cams = np.zeros((K - 1, h, w))
                    for j in range(cams.shape[0]):
                        all_cams[keys[j]] = cams[j]

                    all_cams = np.concatenate((bg_score, all_cams), axis=0)
                    prob = all_cams

                    label = np.argmax(prob, axis=0)
                    label = loader.dataset.decode_target(label).astype(np.uint8)

                    Image.fromarray(label).save('voc/%s_cam.png' % name)



                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('voc/%s_overlay.png' % name, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1
                    
                    metrics.single_update(targets[i], preds[i])
        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()

    # num_classes
    if opts.dataset.lower() == 'voc':
        num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        num_classes = 19
    else:
        raise ValueError(f"Unknown dataset: {opts.dataset}")

    # 随机种子
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 数据
    _, val_dst = get_dataset(opts)
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1  # 与原项目一致
    val_loader = torch.utils.data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2
    )
    print(f"Dataset: {opts.dataset}, Val set: {len(val_dst)}")

    # 模型
    model = network.modeling.__dict__[opts.model](
        num_classes=num_classes, output_stride=opts.output_stride
    )
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # 加载权重
    _ = _load_checkpoint_into_model(model, opts.ckpt)
    model = nn.DataParallel(model).to(device)

    # 指标
    metrics = StreamSegMetrics(num_classes)

    # 验证 + 保存
    model.eval()
    score, _ = validate(opts, model, val_loader, device, metrics)

    # 输出指标
    print(utils.scores_to_str(score)) if hasattr(utils, "scores_to_str") else print(score)
    print(f"Saved visualizations to: {os.path.abspath(opts.save_dir)}")


if __name__ == "__main__":
    main()
