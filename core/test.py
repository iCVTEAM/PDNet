import os
import logging
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from dataset import DatasetType, get_viddataset
from model.pdnet import PDNet


def test_net(cfg):
    model = PDNet(cfg.MODEL.BACKBONE)
    model.eval()

    if cfg.DEVICE.USE_GPU:
        model.cuda()

    if cfg.TEST.STATE_DICT.MAP_LOCATION is None:
        ckpt = torch.load(cfg.TEST.STATE_DICT.PATH)
    elif cfg.TEST.STATE_DICT.MAP_LOCATION == '1_0':
        ckpt = torch.load(cfg.TEST.STATE_DICT.PATH, map_location={'cuda:1': 'cuda:0'})
    elif cfg.TEST.STATE_DICT.MAP_LOCATION == '0_1':
        ckpt = torch.load(cfg.TEST.STATE_DICT.PATH, map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(ckpt['model'])

    test_dataset = get_viddataset(cfg.DATASETS.VID_DATASET,
                                  cfg.DATASETS.VID_PATH,
                                  ds_type=DatasetType.VAL if cfg.TEST.DATA.TYPE == 'VAL' else DatasetType.TRAIN,
                                  input_size=cfg.TEST.DATA.INPUT_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    logging.getLogger().info('successfully set datasets and dataloaders')

    iter_num = len(test_dataloader)
    logging.getLogger().info("Begin to test")
    logging.getLogger().info("iteration numbers: %d", iter_num)

    if not os.path.exists(cfg.TEST.OUT_PATH):
        os.makedirs(cfg.TEST.OUT_PATH)

    for batch_idx, data in enumerate(test_dataloader):
        img, ref = data['target'], data['ref']
        if cfg.DEVICE.USE_GPU:
            img = img.cuda()
            ref = ref.cuda()
        w, h = data['width'][0], data['height'][0]
        seq, frame = data['seq'][0], data['frame'][0]

        with torch.no_grad():
            x = model(img, ref, shape=(h, w))
            prob = torch.sigmoid(x)
            prob = prob.squeeze().cpu().numpy()

        prob = np.round(prob * 255).astype(np.uint8)

        if not os.path.exists(os.path.join(cfg.TEST.OUT_PATH, seq)):
            os.makedirs(os.path.join(cfg.TEST.OUT_PATH, seq))
        cv2.imwrite(os.path.join(cfg.TEST.OUT_PATH, seq, frame + '.png'), prob)
        logging.getLogger().info('saved output for image %d', batch_idx)
