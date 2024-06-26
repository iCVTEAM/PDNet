import os
import cv2
from torch.utils.data import Dataset
from .dataset_type import DatasetType


class MVPSDataset(Dataset):
    """MVPS dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self,
                 ds_type=DatasetType.VAL,
                 ds_rootdir='./MVPS',
                 transform=None):
        self.ds_type = ds_type
        self.ds_rootdir = ds_rootdir
        self.transform = transform
        if self.ds_type == DatasetType.TRAIN:
            fname = 'train'
        else:
            fname = 'val'

        with open(os.path.join(ds_rootdir, 'ImageSets', fname + '.txt')) as f:
            tmp = f.readlines()
            seqs = [x.strip() for x in tmp]
            img_list = []
            label_list = []
            Index = {}
            for seq in seqs:
                images = os.listdir(os.path.join(ds_rootdir, 'Keyframes', '480p', seq))
                images.sort()
                images_path = [os.path.join('Keyframes', '480p', seq, x) for x in images]
                start_num = len(img_list)
                img_list.extend(images_path)
                end_num = len(img_list)
                Index[seq] = [start_num, end_num]
                labels = os.listdir(os.path.join(ds_rootdir, 'Annotations', '480p', seq))
                labels.sort()
                labels_path = [os.path.join('Annotations', '480p', seq, x) for x in labels]
                label_list.extend(labels_path)

        self.img_list = img_list
        self.label_list = label_list
        self.Index = Index

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        seq = self.img_list[idx].split('/')[2] + '/' + self.img_list[idx].split('/')[3]
        target = cv2.imread(os.path.join(self.ds_rootdir, self.img_list[idx]), cv2.IMREAD_COLOR)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        h, w, _ = target.shape

        ref = cv2.imread(os.path.join(self.ds_rootdir, self.img_list[self.Index[seq][0]]), cv2.IMREAD_COLOR)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

        if self.ds_type == DatasetType.TRAIN:
            mask = cv2.imread(os.path.join(self.ds_rootdir, self.label_list[idx]), cv2.IMREAD_GRAYSCALE)
            ref_mask = cv2.imread(os.path.join(self.ds_rootdir, self.label_list[self.Index[seq][0]]), cv2.IMREAD_GRAYSCALE)

            if self.transform is not None:
                [target, ref], [mask, ref_mask] = self.transform([target, ref], [mask, ref_mask])

            sample = {
                'target': target,
                'mask': mask,
                'ref': ref,
                'ref_mask': ref_mask,
                'seq': seq,
                'width': w,
                'height': h
            }
        else:
            if self.transform is not None:
                [target, ref], _ = self.transform([target, ref])
            frame = os.path.splitext(os.path.basename(self.img_list[idx]))[0]
            sample = {'target': target, 'ref': ref, 'seq': seq, 'frame': frame, 'width': w, 'height': h}

        return sample
