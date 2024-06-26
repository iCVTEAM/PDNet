import os
import cv2
from torch.utils.data import Dataset
from .dataset_type import DatasetType
from collections import defaultdict


class PPHumanSeg14KDataset(Dataset):
    """PP-HumanSeg14K dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self,
                 ds_type=DatasetType.VAL,
                 ds_rootdir='./PP-HumanSeg14K',
                 transform=None):
        self.ds_type = ds_type
        self.ds_rootdir = ds_rootdir
        self.transform = transform
        if self.ds_type == DatasetType.TRAIN:
            fname = 'train'
        else:
            fname = 'val'

        with open(os.path.join(ds_rootdir, fname + '_new.txt')) as f:
            tmp = f.readlines()
            imgs = [x.strip().split(' ')[0] for x in tmp]
            gts = [x.strip().split(' ')[1] for x in tmp]
            seqs = [x[7:-8] for x in imgs]
            seqs = sorted(list(set(seqs)))
            image_set = defaultdict(list)
            label_set = defaultdict(list)
            for idx in range(len(imgs)):
                image_set[imgs[idx][7:-8]].append(imgs[idx])
                label_set[imgs[idx][7:-8]].append(gts[idx])
            img_list = []
            label_list = []
            Index = {}
            for seq in seqs:
                image_set[seq].sort()
                label_set[seq].sort()
                start_num = len(img_list)
                img_list.extend(image_set[seq])
                end_num = len(img_list)
                Index[seq] = [start_num, end_num]
                label_list.extend(label_set[seq])

        self.img_list = img_list
        self.label_list = label_list
        self.Index = Index

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        seq = self.img_list[idx][7:-8]
        target = cv2.imread(os.path.join(self.ds_rootdir, self.img_list[idx]), cv2.IMREAD_COLOR)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        h, w, _ = target.shape

        prev = cv2.imread(os.path.join(self.ds_rootdir, self.img_list[self.Index[seq][0]]), cv2.IMREAD_COLOR)
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)

        if self.ds_type == DatasetType.TRAIN:
            mask = cv2.imread(os.path.join(self.ds_rootdir, self.label_list[idx]), cv2.IMREAD_GRAYSCALE)
            prev_mask = cv2.imread(os.path.join(self.ds_rootdir, self.label_list[self.Index[seq][0]]), cv2.IMREAD_GRAYSCALE)

            if self.transform is not None:
                [target, prev], [mask, prev_mask] = self.transform([target, prev], [mask, prev_mask])

            sample = {
                'target': target,
                'mask': mask,
                'prev': prev,
                'prev_mask': prev_mask,
                'seq': seq,
                'width': w,
                'height': h
            }
        else:
            if self.transform is not None:
                [target, prev], _ = self.transform([target, prev])
            frame = os.path.splitext(os.path.basename(self.img_list[idx]))[0]
            sample = {'target': target, 'prev': prev, 'seq': seq, 'frame': frame, 'width': w, 'height': h}

        return sample
