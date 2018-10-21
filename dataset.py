from itertools import chain

import glob
import torch
from PIL import Image
from os import path
from torch.utils.data import Dataset
import os
import os.path as opath
import numpy as np
import pdb
import random
from torchvision import transforms
from PIL import Image
import pycocotools.mask as mask_util
import pickle

def calc_angle(p1, p2):
    eps = 0.0000001
    vec = p1 - p2
    if vec[0] > 0:
        vec = -vec
    vec_len = np.sqrt(np.abs(vec[0]*vec[0]+vec[1]*vec[1]))
    angle = np.arccos(vec[1] / (vec_len+eps))
    return angle

def triangle(p1, p2, p3):
    eps = 0.0000001
    v12 = p2 - p1
    v13 = p3 - p1
    v12_len = np.sqrt(np.abs(v12[0]*v12[0]+v12[1]*v12[1]))
    v13_len = np.sqrt(np.abs(v13[0]*v13[0]+v13[1]*v13[1]))
    return np.arccos(np.sum(v12 * v13) / ((v12_len * v13_len) + eps))

def calc_triangle(p1, p2, p3):
    return triangle(p1, p2, p3), triangle(p2, p1, p3), triangle(p3, p1, p2)

def calc_distance(p1, p2):
    dx = np.abs(p1[0]-p2[0])
    dy = np.abs(p1[1]-p2[1])
    return dx, dy, np.sqrt(dx*dx+dy*dy)

# keypoint connections [1, 2], [1, 0], [2, 0], [2, 4], [1, 3], [6, 8], [8, 10], [5, 7], [7, 9], [12, 14], [14, 16], [11, 13], [13, 15], [6, 5], [12, 11]
def keypoint_feature(keyps, edges):
    triangles = [[0, 1, 2], [0,2,4], [1,2,4], [1,2,3],[6,8,10], [5,7,9], [12,14,16], [11,13,15], [11,12,13], [11,12,14], [5,6,7], [5,6,8]]
    _, _, length = calc_distance(keyps[5,:], keyps[16,:])
    features = []
    for edge in edges:
        dx, dy, dis = calc_distance(keyps[edge[0], :], keyps[edge[1], :])
        dx /= length
        dy /= length
        dis /= length
        features.append(dx)
        features.append(dy)
        features.append(dis)
        features.append(calc_angle(keyps[edge[0], :], keyps[edge[1], :])/np.pi)
    for triangle in triangles:
        angle1, angle2, angle3 = calc_triangle(keyps[triangle[0], :], keyps[triangle[1], :], keyps[triangle[2], :])
        features.append(angle1/np.pi)
        features.append(angle2/np.pi)
        features.append(angle3/np.pi)
    return np.array(features)

class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

def rgb_transform(resize_size=256, crop_size=224):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        ResizeImage(resize_size),
        transforms.ToTensor(),
        normalize
    ])

def flow_transform(resize_size=256, crop_size=224):
  normalize = transforms.Normalize(mean=0.5,
                                   std=1)
  return  transforms.Compose([
        ResizeImage(resize_size),
        transforms.ToTensor(),
        normalize
    ])

def pil_loader(path):
    return Image.open(path)

class SegmentationDataset(Dataset):
    _EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

    def __init__(self, in_dir, threshold, transform):
        super(SegmentationDataset, self).__init__()

        self.in_dir = in_dir
        self.transform = transform

        # Find all images
        self.images = []
        for sub_dir in sorted(os.listdir(self.in_dir)):
         if (threshold - 30) <= int(sub_dir.split(".")[0].split("_")[1]) < threshold:
          for img_path in chain(*(glob.iglob(path.join(self.in_dir, sub_dir, ext)) for ext in SegmentationDataset._EXTENSIONS)):
            _, name_with_ext = path.split(img_path)
            idx, _ = path.splitext(name_with_ext)
            self.images.append({
                "idx": sub_dir.split(".")[0] + "_" + idx,
                "path": img_path
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Load image
        with Image.open(self.images[item]["path"]) as img_raw:
            size = img_raw.size
            img = self.transform(img_raw.convert(mode="RGB"))

        return {"img": img, "meta": {"idx": self.images[item]["idx"], "size": size}}


def segmentation_collate(items):
    imgs = torch.stack([item["img"] for item in items])
    metas = [item["meta"] for item in items]

    return {"img": imgs, "meta": metas}

class JAADClassificationDataset(Dataset):
    def __init__(self, phase, clip_size=-1):
        super(JAADClassificationDataset, self).__init__()
        self.transform = train_transform((640, 360))
        self.phase = phase
        self.videos = {}
        self.info = {"size":{}, 'densepose':{}}
        for dir_ in sorted(os.listdir('/workspace/caozhangjie/inplace_abn/JAAD_processed_data')):
            temp_dict = open('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/image_list.txt').readlines()
            id_list = [[int(line.split()[0].split('/')[-1].split('.')[0].split('_')[-1]), line.split()] for line in temp_dict]
            self.videos[int(dir_.split('_')[1])] = {}
            for val in id_list:
                self.videos[int(dir_.split('_')[1])][val[0]] = val[1]
                self.info['size'][int(dir_.split('_')[1])] = [[int(size_line.split()[0]), int(size_line.split()[1])] for size_line in open('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/size_list.txt').readlines()]
        if self.phase == 'train':
            self.train_list = []
            for dir_ in os.listdir('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'):
                if int(dir_.split('_')[1]) <= 250:
                    temp_list = [val.split() for val in open('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/list_clip_{:d}_base.txt'.format(clip_size), 'r').readlines()]
                    temp_list = [[int(num) for num in val] + [int(dir_.split('_')[1])] for val in temp_list]
                    self.train_list += temp_list
            self.train_dict = {}
            for entries in self.train_list:
                if entries[3] not in self.train_dict:
                    self.train_dict[entries[3]] = []
                self.train_dict[entries[3]].append(entries)
        elif self.phase == 'test':
            self.test_list = []
            for dir_ in os.listdir('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'):
                if int(dir_.split('_')[1]) > 250:
                    temp_list = [val.split() for val in open('/workspace/caozhangjie/inplace_abn/JAAD_processed_data/'+dir_+'/list_clip_{:d}_base.txt'.format(clip_size), 'r').readlines()]
                    temp_list = [[int(num) for num in val] + [int(dir_.split('_')[1])] for val in temp_list]
                    self.test_list += temp_list
                

    def __getitem__(self, index):
      if self.phase == 'train':
        data_cls = sorted(self.train_dict.keys())[int(random.random() * 100000000) % len(self.train_dict)]
        s_frame, e_frame, ped_id, data_cls_new, video_index = self.train_dict[data_cls][int(random.random() * 100000000) % len(self.train_dict[data_cls])]
        assert data_cls_new == data_cls
      elif self.phase == 'test':
        s_frame, e_frame, ped_id, data_cls, video_index = self.test_list[index]
      all_rgbs = []
      all_flow_xs = []
      all_flow_ys = []
      #all_features = []
      #all_ped_pos = []
      all_ped_label = []
      #all_img_size = []
      for i in range(s_frame, e_frame):
          img_path, flow_x_path, flow_y_path, pos_path, \
            label_path, c_label_path, i_label_path, densepose_path \
            = self.videos[video_index][i+1]
          pdb.set_trace()
          size_img = self.info['size'][video_index][i]
          #all_img_size.append(size_img)
          def translate_ped_id(pos, ped_id):
            j = -1
            for i in range(ped_id+1):
                if pos[i][0] > 0:
                    j+=1
            return j

          def load_image(path, id_, pos_, masks_):
            img = pil_loader(path)
            norm_img = self.transform(img)
            img = np.array(img)
            if len(img.shape) == 2:
                related_img = (np.sum(masks_, axis=2) > 0).astype(img.dtype).reshape([masks_.shape[0], masks_.shape[1]])
            else:
                related_img = (np.sum(masks_, axis=2) > 0).astype(img.dtype).reshape([masks_.shape[0], masks_.shape[1], 1])
            related_img = img * related_img
            related_img = self.transform(Image.fromarray(related_img))
            masks = masks_[:,:,translate_ped_id(pos_, id_)]
            if len(img.shape) == 2:
                ped_img = img * masks.astype(img.dtype).reshape([masks_.shape[0], masks_.shape[1]])
            else:
                ped_img = img * masks.astype(img.dtype).reshape([masks_.shape[0], masks_.shape[1], 1])
            ped_img = self.transform(Image.fromarray(ped_img))
            if len(norm_img.size()) == 2:
                norm_img = norm_img.view(1, norm_img.size(0), norm_img.size(1))
                ped_img = ped_img.view(1, norm_img.size(0), norm_img.size(1))
                related_img = related_img.view(1, norm_img.size(0), norm_img.size(1))
            return norm_img, ped_img, related_img

          ## position
          ped_pos = np.load(pos_path)
          ## densepose feature
          densepose = pickle.load(open(densepose_path, 'rb'), encoding='latin1')
          masks_all = mask_util.decode(densepose['masks'])
          #keyps = np.transpose(densepose['keyps'][ped_id][0:2, :])
          #boxes = densepose['boxes']
          #bodys = densepose['bodys'][ped_id]
          #kp_lines = densepose['kp_lines']
          #keyp_features = keypoint_feature(keyps, kp_lines)
          #all_keyp_features.append(keyp_features)
          #all_ped_pos.append(ped_pos)
          rgb = load_image(img_path, ped_id, ped_pos, masks_all)
          flow_x = load_image(flow_x_path, ped_id, ped_pos, masks_all)
          flow_y = load_image(flow_y_path, ped_id, ped_pos, masks_all)
          pdb.set_trace()
          all_rgbs.append(rgb)
          all_flow_xs.append(flow_x)
          all_flow_ys.append(flow_y)
      #all_ped_pos = np.array(all_ped_pos)
      return all_rgbs, all_flow_xs, all_flow_ys, data_cls

    def __len__(self):
        if self.phase == 'train':
            return 2000
        elif self.phase == 'test':
            #return len(self.test_list)
            return 50

def JAADCollateClassification(items):
    rgbs = [item[0] for item in items]
    flow_xs = [item[1] for item in items]
    flow_ys = [item[2] for item in items]
    new_rgbs = [[], [], []]
    new_flow_xs = [[], [], []]
    new_flow_ys = [[], [], []]
    for item in rgbs:
        new_rgbs[0].append(item[0])
        new_rgbs[1].append(item[1])
        new_rgbs[2].append(item[2])
    for item in flow_xs:
        new_flow_xs[0].append(item[0])
        new_flow_xs[1].append(item[1])
        new_flow_xs[2].append(item[2])
    for item in flow_ys:
        new_flow_ys[0].append(item[0])
        new_flow_ys[1].append(item[1])
        new_flow_ys[2].append(item[2])
    new_rgbs[0] = torch.stack(new_rgbs[0])
    new_rgbs[1] = torch.stack(new_rgbs[1])
    new_rgbs[2] = torch.stack(new_rgbs[2])
    new_flow_xs[0] = torch.stack(new_flow_xs[0])
    new_flow_xs[1] = torch.stack(new_flow_xs[1])
    new_flow_xs[2] = torch.stack(new_flow_xs[2])
    new_flow_ys[0] = torch.stack(new_flow_ys[0])
    new_flow_ys[1] = torch.stack(new_flow_ys[1])
    new_flow_ys[2] = torch.stack(new_flow_ys[2])
    label = torch.Tensor([item[3] for item in items])
    return new_rgbs, new_flow_xs, new_flow_ys, label

if __name__ == '__main__':
    train_dset = JAADClassificationDataset('train', clip_size=30)
    #pdb.set_trace()
    print(len(train_dset[0]))
