from utils import joint_transforms
from .dataset_type import DatasetType
from .mvps import MVPSDataset
from .pphumanseg14k import PPHumanSeg14KDataset
from .image import ImgDataCfg, SimpleImgDataset


mean = [0.485, 0.456, 0.406]  # imagenet mean and std value
std = [0.229, 0.224, 0.225]

def get_viddataset(dataset_name,
                   ds_rootdir,
                   ds_type=DatasetType.VAL,
                   input_size=480):
    maskmx = 38 if dataset_name == 'PP-HumanSeg14K' else 255
    if ds_type == DatasetType.TRAIN:
        transforms = joint_transforms.Compose([
            joint_transforms.Normalize(mean, std, maskmx=maskmx),
            joint_transforms.RandomCrop(),
            joint_transforms.RandomFlip(),
            joint_transforms.Resize((input_size, input_size)),
            joint_transforms.ToTensor()
        ])
        if dataset_name == 'MVPS':
            return MVPSDataset(ds_type=ds_type, ds_rootdir=ds_rootdir, transform=transforms)
        elif dataset_name == 'PP-HumanSeg14K':
            return PPHumanSeg14KDataset(ds_type=ds_type, ds_rootdir=ds_rootdir, transform=transforms)
    else:
        transforms = joint_transforms.Compose([
            joint_transforms.Normalize(mean, std, maskmx=maskmx),
            joint_transforms.Resize((input_size, input_size)),
            joint_transforms.ToTensor()
        ])
        if dataset_name == 'MVPS':
            return MVPSDataset(ds_type=DatasetType.VAL, ds_rootdir=ds_rootdir, transform=transforms)
        elif dataset_name == 'PP-HumanSeg14K':
            return PPHumanSeg14KDataset(ds_type=DatasetType.VAL, ds_rootdir=ds_rootdir, transform=transforms)


def get_imgdataset(datasets_name,
                   ds_rootdir,
                   augment=True,
                   input_size=480):
    obj_cfg = ImgDataCfg(datasets=datasets_name, ds_rootdir=ds_rootdir)
    data_cfg = obj_cfg.data_cfg

    if augment:
        transforms = joint_transforms.Compose([
            joint_transforms.Normalize(mean, std),
            joint_transforms.RandomCrop(),
            joint_transforms.RandomFlip(),
            joint_transforms.Resize((input_size, input_size)),
            joint_transforms.ToTensor()
        ])
        return SimpleImgDataset(data_cfg.img_pathlist, data_cfg.gt_pathlist, transform=transforms)
    else:
        transforms = joint_transforms.Compose([
            joint_transforms.Normalize(mean, std),
            joint_transforms.Resize((input_size, input_size)),
            joint_transforms.ToTensor()
        ])
        return SimpleImgDataset(data_cfg.img_pathlist, data_cfg.gt_pathlist, transform=transforms)
