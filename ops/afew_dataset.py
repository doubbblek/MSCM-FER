import torch.utils.data as data
import pickle
from PIL import Image
import os
import numpy as np
from numpy.random import randint
import torch 
from math import ceil, floor
import torchvision
from .transforms import *


def get_list(path):
    labels =  {'Happy': 0, 'Sad': 1, 'Neutral': 2, 'Angry': 3, 'Surprise': 4, 'Disgust': 5, 'Fear': 6}
    train_list = []
    val_list = []
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')

    for c in labels.keys():
        cls_folder = os.path.join(train_path, c)
        sample_list = os.listdir(cls_folder)

        for s in sample_list:
            if '.DS_Store' in s:
                continue
            sample_folder = os.path.join(cls_folder, s)
            tmp = {'path': sample_folder, 'label': labels[c]}
            img_list = os.listdir(sample_folder)
            if len(img_list) > 0:
                train_list.append(tmp)
    
    for c in labels.keys():
        cls_folder = os.path.join(val_path, c)
        sample_list = os.listdir(cls_folder)

        for s in sample_list:
            if '.DS_Store' in s:
                continue
            sample_folder = os.path.join(cls_folder, s)
            tmp = {'path': sample_folder, 'label': labels[c]}
            if len(img_list) > 0:
                val_list.append(tmp)

    return train_list, val_list


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, data_pair,
                 num_segments=3, new_length=1,
                 image_tmpl='{:07d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, 
                 twice_sample=False, adjacent=False):

        self.root_path = root_path
        self.data_pair = data_pair
        self.num_segments = num_segments
        self.new_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        self.adjacent = adjacent
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

    def _load_image(self, directory, name):
        return [Image.open(os.path.join(directory, name)).convert('RGB')]

    def _sample_indices(self, num_frames):
        seg_size = float(num_frames - 1) / self.num_segments
        if self.num_segments <= num_frames:
            seq = np.array(
                [random.randint(ceil(seg_size*i), floor(seg_size*(i+1))) for i in range(self.num_segments)],
                dtype=int)
        else:
            seq = np.arange(num_frames)
            pad = np.zeros((self.num_segments-num_frames,))
            seq = np.concatenate([pad, seq], axis=0).astype(int)
        return seq + 1

    def _get_val_indices(self, num_frames):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if num_frames > self.num_segments + self.new_length - 1:
                tick = (num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, num_frames):
        if self.dense_sample:
            sample_pos = max(1, 1 + num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        record = self.data_pair[index]
        # check this is a legit video folder
        full_path = os.path.join(record['path'])
        
        num_frames = len(os.listdir(full_path))
        img_names = os.listdir(full_path)
        img_names.sort()

        if not self.test_mode:
            segment_indices = self._sample_indices(num_frames) if self.random_shift else self._get_val_indices(num_frames)
        else:
            segment_indices = self._get_test_indices(num_frames)

        if self.adjacent: # sample adjacent frame
            ad_offsets = segment_indices.copy() - 2
            ad_offsets = np.where(ad_offsets<1, 1, ad_offsets)

            new_offsets = np.zeros((self.num_segments*2,), dtype='int')
            new_offsets[::2] = ad_offsets
            new_offsets[1::2] = segment_indices
            segment_indices = new_offsets 
        
        selected_img_names = [img_names[int(i-1)] for i in segment_indices]

        return self.get(record, num_frames, selected_img_names)

    def get(self, record, num_frames, img_names):

        images = list()
        for seg_ind, name in enumerate(img_names):
            p = int(seg_ind)
            seg_imgs = self._load_image(record['path'], name)
            images.extend(seg_imgs)

        process_data = self.transform(images)
        return process_data, record['label']

    def __len__(self):
        return len(self.data_pair)


def get_afew_dataset(num_frames, input_size, scale_size):
    train_list, val_list = get_list('/home/tankunli2/hmdb_master/aligned_faces')

    train_dataset = TSNDataSet(None,
                          train_list, num_segments=num_frames,
                          image_tmpl='{:05d}_{:05d}.jpg',
                          transform=torchvision.transforms.Compose([
                            GroupScale(scale_size),
                            GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                            GroupRandomHorizontalFlip(is_flow=False),
                            Stack(roll=False),
                            ToTorchFormatTensor(div=True),
                            GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]),
                         random_shift=True,test_mode=False,
                         remove_missing=False, dense_sample=False, 
                         twice_sample=False, adjacent=False)
    
    test_dataset = TSNDataSet(None,
                          val_list, num_segments=num_frames,
                          image_tmpl='{:05d}_{:05d}.jpg',
                          transform=torchvision.transforms.Compose([
                            GroupScale(scale_size),
                            GroupCenterCrop(input_size),
                            Stack(roll=False),
                            ToTorchFormatTensor(div=True),
                            GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]),
                         random_shift=False,test_mode=True,
                         remove_missing=False, dense_sample=False, 
                         twice_sample=False, adjacent=False)

    return train_dataset, test_dataset


if __name__ == '__main__':
    # ds1, ds2 = get_afew_dataset(16, 112, 128)

    # dl = data.DataLoader(ds1, 8, True)
    # for  i, j in dl:
    #     print(i.size(), j)
    train_list, val_list = get_list('/home/tankunli2/hmdb_master/aligned_faces')
    all_list = train_list + val_list
    
    cnt = {}
    for i in all_list:
        if i['label'] not in cnt.keys():
            cnt[i['label']] = 0
        
        cnt[i['label']] += 1 