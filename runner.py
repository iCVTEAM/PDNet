import os
import logging, time, yaml
from easydict import EasyDict
from argparse import ArgumentParser
from core.train import train_net
from core.test import test_net


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    cur_time = time.strftime('%Y%m%d%H%M', time.localtime())
    log_file = os.path.join(log_path, cur_time + '.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def get_config_from_file():
    f = open('config.yaml', 'r')
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    f.close()
    cfg = EasyDict(cfg)
    return cfg


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', dest='gpu_id', type=str, default=None,
                        help='Specify \'<id>\' to use GPU with given id e.g., --gpu \'0\' uses first GPU.')
    parser.add_argument('--test', dest='test', action='store_true',
                        help='Test the network.')
    return parser.parse_args()


def import_config_from_args(cfg, args):
    if args.gpu_id is not None:
        cfg.DEVICE.USE_GPU = True
        cfg.DEVICE.GPU_ID = args.gpu_id


if __name__ == '__main__':
    cfg = get_config_from_file()
    args = parse_args()
    import_config_from_args(cfg, args)
    set_logger(cfg.LOG_PATH)
    logging.getLogger().info('start runner')

    if cfg.DEVICE.USE_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE.GPU_ID
    if cfg.RANDOM_SEED is not None:
        if cfg.DEVICE.USE_GPU:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import random
        import torch
        import numpy as np
        random.seed(cfg.RANDOM_SEED)
        torch.manual_seed(cfg.RANDOM_SEED)
        if cfg.DEVICE.USE_GPU:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if cfg.DEVICE.NGPUS == 1:
                torch.cuda.manual_seed(cfg.RANDOM_SEED)
            else:
                torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
        np.random.seed(cfg.RANDOM_SEED)

    if not args.test:
        train_net(cfg)
    else:
        test_net(cfg)
