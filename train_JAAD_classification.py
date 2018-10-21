import argparse
from os import path
import torch
import torchvision
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import pdb
import network
from dataset import JAADClassificationDataset, JAADCollateClassification
from torch.utils.data import DataLoader
from network import init_weights
import pickle
from src.i3dpt import I3D, Unit3Dpy

def get_scores(sample, model):
        #pdb.set_trace()
        sample_var = torch.autograd.Variable(torch.from_numpy(sample[:,:,0:15,:,:]).cuda())
        out_var, out_logit = model(sample_var)
        #pdb.set_trace()
        out_tensor = out_var.data.cpu()

        top_val, top_idx = torch.sort(out_tensor, 1, descending=True)

        print(
            'Top {} classes and associated probabilities: '.format(args.top_k))
        for i in range(args.top_k):
            print('[{}]: {:.6E}'.format(kinetics_classes[top_idx[0, i]],
                                        top_val[0, i]))
        return out_logit



        print('===== Final predictions ====')
        print('logits proba class '.format(args.top_k))
        for i in range(args.top_k):
            logit_score = out_logit[0, top_idx[0, i]].data[0]
            print('{:.6e} {:.6e} {}'.format(logit_score, top_val[0, i],
                                            kinetics_classes[top_idx[0, i]]))

def normalize_pos(pos, img_sizes):
    pos[:,:,0] = (pos[:,:,0] / img_sizes[:,:,0] - 0.5) * 10.0
    pos[:,:,2] = (pos[:,:,2] / img_sizes[:,:,0] - 0.5) * 10.0
    pos[:,:,1] = (pos[:,:,1] / img_sizes[:,:,1] - 0.5) * 10.0
    pos[:,:,3] = (pos[:,:,3] / img_sizes[:,:,1] - 0.5) * 10.0

def step_lr_scheduler(optimizer, iter_num, step_size, gamma):
    if iter_num % step_size == step_size - 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * gamma
    return optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAAD graph model")
    parser.add_argument('--gpu_id', metavar='GPU_ID', type=int, default=0, help='GPU ID')
    parser.add_argument('--test_interval', type=int, default=500, help='test interval')
    parser.add_argument('--epoch', type=int, default=100, help='train epochs')
    parser.add_argument('--debug', action='store_true', help='debug switch')
    # RGB arguments
    parser.add_argument(
        '--rgb', action='store_true', help='Evaluate RGB pretrained network')
    parser.add_argument(
        '--rgb_weights_path',
        type=str,
        default='model/model_rgb.pth',
        help='Path to rgb model state_dict')
    parser.add_argument(
        '--rgb_sample_path',
        type=str,
        default='data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy',
        help='Path to kinetics rgb numpy sample')

    # Flow arguments
    parser.add_argument(
        '--flow', action='store_true', help='Evaluate flow pretrained network')
    parser.add_argument(
        '--flow_weights_path',
        type=str,
        default='model/model_flow.pth',
        help='Path to flow model state_dict')
    parser.add_argument(
        '--classes_path',
        type=str,
        default='data/kinetic-samples/label_map.txt',
        help='Number of video_frames to use (should be a multiple of 8)')
    parser.add_argument(
        '--top_k',
        type=int,
        default='5',
        help='When display_samples, number of top classes to display')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    cudnn.benchmark = True

    train_dset = JAADClassificationDataset('train', clip_size=30)
    train_loader = DataLoader(train_dset, batch_size=1, pin_memory=True, num_workers=4, shuffle=True, collate_fn=JAADCollateClassification)
    test_dset = JAADClassificationDataset("test", clip_size=30)
    test_loader = DataLoader(test_dset, batch_size=1, pin_memory=True, num_workers=4, shuffle=False, collate_fn=JAADCollateClassification)

    if args.rgb:
        i3d_rgb = I3D(num_classes=400, modality='rgb')
        i3d_rgb.load_state_dict(torch.load(args.rgb_weights_path))
        i3d_rgb.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024*3,
            out_channels=2,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
        init_weights(i3d_rgb.conv3d_0c_1x1)
        i3d_rgb.cuda()
    if args.flow:
        i3d_flow = I3D(num_classes=400, modality='flow')
        i3d_flow.load_state_dict(torch.load(args.flow_weights_path))      
        i3d_flow.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            out_channels=2,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
        init_weights(i3d_flow.conv3d_0c_1x1)
        i3d_flow.cuda()

   
    optimizer = optim.Adam([{"params":i3d_rgb.parameters()}], lr=0.001)
    for epoch in range(args.epoch):
        i3d_rgb.train()
        #i3d_flow.train()
        for batch_id, data in enumerate(train_loader):
            step_lr_scheduler(optimizer, epoch * len(train_loader) + batch_id, 500, 0.3)
            optimizer.zero_grad()
            rgbs, flow_xs, flow_ys, label = data[0], data[1], data[2], data[3].cuda()
            pdb.set_trace()
            label = label.long()
            # Rung RGB model
            if args.rgb:
                rgbs[0] = rgbs[0].permute(0,2,1,3,4).contiguous().cuda()
                rgbs[1] = rgbs[1].permute(0,2,1,3,4).contiguous().cuda()
                rgbs[2] = rgbs[2].permute(0,2,1,3,4).contiguous().cuda()
                out = i3d_rgb(rgbs[0])
                out_related = i3d_rgb(rgbs[2])
                out_ped = i3d_rgb(rgbs[1])
                pooling = nn.AvgPool3d((1,out_ped.size(3),out_ped.size(4))).cuda()
                out_ped = pooling(out_ped)
                out = pooling(out)
                out_related = pooling(out_related)
                out_logits = i3d_rgb.conv3d_0c_1x1(torch.cat((out, out_related, out_ped), dim=1))
                out_logits = out_logits.squeeze(3).squeeze(3).mean(2)
            if args.flow:
                flow = [None, None, None]
                flows[0] = torch.cat((flow_xs[0].permute(0,2,1,3,4).contiguous(), \
                                  flow_ys[0].permute(0,2,1,3,4).contiguous()), dim=1).cuda()
                flows[1] = torch.cat((flow_xs[1].permute(0,2,1,3,4).contiguous(), \
                                  flow_ys[1].permute(0,2,1,3,4).contiguous()), dim=1).cuda()
                flows[2] = torch.cat((flow_xs[2].permute(0,2,1,3,4).contiguous(), \
                                  flow_ys[2].permute(0,2,1,3,4).contiguous()), dim=1).cuda()
                out = i3d_flow(flows[0])
                out_related = i3d_flow(flows[1])
                out_ped = i3d_flow(flows[2])
                out_logits_flow = i3d_flow.conv3d_0c_1x1(torch.cat((out, out_related, out_ped), dim=1))
                out_logits_flow = out_logits_flow.squeeze(3).squeeze(3).mean(2)
            if args.rgb and args.flow:
                out_logits = torch.log((nn.Softmax()(out_logits) + nn.Softmax()(out_logits_flow)) / 2.0)
                loss = nn.NLLLoss()(out_logits, label)
            else:
                loss = nn.CrossEntropyLoss()(out_logits, label)
            loss.backward()
            optimizer.step()
            log_str = "Epoch: {:04d}, Iter: {:05d}, Loss: {:.4f}".format(epoch, batch_id, loss.item())
            print(log_str)
            if (epoch * len(train_loader) + batch_id) % args.test_interval == 0:
              with torch.no_grad():
                i3d_rgb.eval()
                avg_acc = 0
                num_test = 0
                for test_id, data in enumerate(test_loader):
                    imgs, ped_imgs, related_imgs, label = data[0], data[1], data[2], data[3]
                    label = label.long()
                    # Rung RGB model
                    if args.rgb and (not args.flow):
                        imgs = imgs.permute(0,2,1,3,4).contiguous().cuda()
                        related_imgs = related_imgs.permute(0,2,1,3,4).contiguous().cuda()
                        ped_imgs = ped_imgs.permute(0,2,1,3,4).contiguous().cuda()
                        out = i3d_rgb(imgs)
                        out_related = i3d_rgb(related_imgs)
                        out_ped = i3d_rgb(ped_imgs)
                        pooling = nn.AvgPool3d((1,out_ped.size(3),out_ped.size(4))).cuda()
                        out_ped = pooling(out_ped)
                        out = pooling(out)
                        out_related = pooling(out_related)
                        out_logits = i3d_rgb.conv3d_0c_1x1(torch.cat((out, out_related, out_ped), dim=1))
                        out_logits = out_logits.squeeze(3).squeeze(3).mean(2)
                    elif args.flow and (not args.rgb):
                        out = i3d_flow(imgs)
                        out_related = i3d_flow(related_imgs)
                        out_ped = i3d_flow(ped_imgs)
                        out_logits = i3d_flow.conv3d_0c_1x1(torch.cat((out, out_related, out_ped), dim=1))
                    _, ped_pred = torch.max(out_logits, 1)
                    ped_pred = ped_pred.cpu()
                    avg_acc += float(torch.sum(ped_pred.cpu()==label))
                    num_test += imgs.size(0)
                if args.debug:
                    pdb.set_trace()
                print("Epoch: {:04d}, Iter: {:05d}, accuracy: {:.4f}".format(epoch, batch_id, avg_acc/num_test))
                i3d_rgb.train()
