import os
from torch.utils.data import Dataset
import cv2
from easydict import EasyDict as edict


class ImgDataCfg:
    """config dataset path and value"""
    def __init__(self, datasets, ds_rootdir='./img-portrait-datasets'):
        self.datasets = datasets if isinstance(datasets, list) else [datasets]
        self.data_cfg = edict()
        self.data_cfg.data_info = []

        if 'EG1800' in self.datasets:
            dataset_info = edict()
            dataset_info.data_dir = os.path.join(ds_rootdir, 'EG1800')
            dataset_info.img_folder = 'Images'
            dataset_info.gt_folder = 'Labels'
            dataset_info.img_filetype = None
            dataset_info.gt_filetype = None
            self.data_cfg.data_info.append(dataset_info)
        if 'SuperviselyPerson' in self.datasets:
            dataset_info = edict()
            dataset_info.data_dir = os.path.join(ds_rootdir, 'SuperviselyPerson')
            dataset_info.img_folder = 'img'
            dataset_info.gt_folder = 'mask'
            dataset_info.img_filetype = None
            dataset_info.gt_filetype = None
            self.data_cfg.data_info.append(dataset_info)
        if 'COCO_person' in self.datasets:
            dataset_info = edict()
            dataset_info.data_dir = os.path.join(ds_rootdir, 'COCO_person')
            dataset_info.img_folder = 'img'
            dataset_info.gt_folder = 'mask'
            dataset_info.img_filetype = None
            dataset_info.gt_filetype = None
            self.data_cfg.data_info.append(dataset_info)

        self.data_cfg.img_pathlist, self.data_cfg.gt_pathlist = [], []
        for dataset in self.data_cfg.data_info:
            img_namelist = os.listdir(os.path.join(dataset.data_dir, dataset.img_folder))
            if dataset.img_filetype is None:
                img_pathlist = [os.path.join(dataset.data_dir, dataset.img_folder, filename) for filename in img_namelist]
            else:
                img_pathlist = [os.path.join(dataset.data_dir, dataset.img_folder, filename) for filename in img_namelist if filename.endswith(dataset.img_filetype)]
            img_pathlist.sort()
            self.data_cfg.img_pathlist.extend(img_pathlist)

            gt_namelist = os.listdir(os.path.join(dataset.data_dir, dataset.gt_folder))
            if dataset.gt_filetype is None:
                gt_pathlist = [os.path.join(dataset.data_dir, dataset.gt_folder, filename) for filename in gt_namelist]
            else:
                gt_pathlist = [os.path.join(dataset.data_dir, dataset.gt_folder, filename) for filename in gt_namelist if filename.endswith(dataset.gt_filetype)]
            gt_pathlist.sort()
            self.data_cfg.gt_pathlist.extend(gt_pathlist)


class SimpleImgDataset(Dataset):
    def __init__(self, img_pathlist, gt_pathlist, transform=None):
        self.img_pathlist = img_pathlist
        self.gt_pathlist = gt_pathlist
        self.transform = transform

    def __len__(self):
        return len(self.img_pathlist)

    def __getitem__(self, item):
        img = cv2.imread(self.img_pathlist[item], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.gt_pathlist[item], cv2.IMREAD_GRAYSCALE)
        h, w, _ = img.shape

        if self.transform is not None:
            [img], [mask] = self.transform([img], [mask])

        sample = {
            'image': img,
            'mask': mask,
            'name': os.path.splitext(os.path.basename(self.img_pathlist[item]))[0],
            'width': w,
            'height': h
        }
        return sample
