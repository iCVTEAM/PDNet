import numbers
import cv2
import numpy as np
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, masks=None):
        for t in self.transforms:
            images, masks = t(images, masks)
        return images, masks


class Normalize(object):
    def __init__(self, mean, std, maskmx=255):
        self.mean = mean
        self.std = std
        self.maskmx = maskmx

    def __call__(self, images, masks=None):
        for idx in range(len(images)):
            images[idx] = (images[idx] / 255 - self.mean) / self.std
        if masks is not None:
            for idx in range(len(masks)):
                masks[idx] = masks[idx] / self.maskmx
        return images, masks


class RandomCrop(object):
    def __call__(self, images, masks):
        H, W, _ = images[0].shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        for idx in range(len(images)):
            images[idx] = images[idx][p0:p1, p2:p3, :]
        if masks is not None:
            for idx in range(len(masks)):
                masks[idx] = masks[idx][p0:p1, p2:p3]
        return images, masks


class RandomFlip(object):
    def __call__(self, images, masks):
        if np.random.randint(2) == 0:
            for idx in range(len(images)):
                images[idx] = images[idx][:, ::-1, :]
            if masks is not None:
                for idx in range(len(masks)):
                    masks[idx] = masks[idx][:, ::-1]
        return images, masks


class Resize(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size  # w, h

    def __call__(self, images, masks):
        for idx in range(len(images)):
            images[idx] = cv2.resize(images[idx], self.size)
        if masks is not None:
            for idx in range(len(masks)):
                masks[idx] = cv2.resize(masks[idx], self.size)
        return images, masks


class ToTensor(object):
    def __call__(self, images, masks):
        for idx in range(len(images)):
            images[idx] = images[idx].astype(np.float32)
            images[idx] = images[idx].transpose((2, 0, 1))
            images[idx] = torch.from_numpy(images[idx])
        if masks is not None:
            for idx in range(len(masks)):
                masks[idx] = masks[idx].astype(np.float32)
                if masks[idx].ndim == 2:
                    masks[idx] = masks[idx][np.newaxis, :]
                masks[idx] = torch.from_numpy(masks[idx])
        return images, masks
