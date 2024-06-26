# PDNet
**Towards imbalanced motion: part-decoupling network for video portrait segmentation** (SCIS 2024)

[[Paper]](https://www.sciengine.com/SCIS/doi/10.1007/s11432-023-4030-y) [[arXiv]](https://arxiv.org/abs/2307.16565) [[中文介绍]](https://mp.weixin.qq.com/s/gkVjp7PeSD8YsSrnOUK1BA)

## MVPS dataset
To acquire our MVPS dataset, please contact [yuts@buaa.edu.cn](mailto:yuts@buaa.edu.cn) via an **official email**. Please explain the purpose of usage in the email. The dataset is available for academic research only.

[evaluation code](https://github.com/iCVTEAM/MVPS-evaluation)

## Image portrait segmentation datasets for training
The datasets are from EG1800, Supervisely Person and MS COCO.

[download link](https://drive.google.com/drive/folders/1Mv6WyGQI8Vv04m8rneivyFJ5tigaXxtq?usp=sharing)

## Model weights
[ResNet-50 weights](https://drive.google.com/file/d/1x1oQpRheUZffqvaXUZxd54gB7YmD7pGN/view?usp=sharing)

[ResNet-101 weights](https://drive.google.com/file/d/1dMIVy2wFTBHAk1rFF-v6Gj5YICc7R5Ea/view?usp=sharing)

When using ResNet-101 as backbone, please set the ``MODEL.BACKBONE`` parameter in *config.yaml* to ``'resnet101'``.

## Command for training/test
PyTorch version: 1.11.0

Training: ``torchrun --nproc_per_node=2 runner.py``

Test: ``python runner.py --test``

## Citation
```
@article{PDNet,
  author={Tianshu Yu and Changqun Xia and Jia Li},
  title={Towards imbalanced motion: part-decoupling network for video portrait segmentation},
  journal={SCIENCE CHINA Information Sciences},
  year={2024},
  volume={67},
  number={7},
  pages={172104},
  doi={10.1007/s11432-023-4030-y}
}
```
