import os
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
from collections import OrderedDict
from dataset import DatasetType, get_imgdataset, get_viddataset
from model.pdnet import PDNet, PartBasisGenerator
from utils.optimizer import get_backbone_params, get_head_params
from utils.loss import loss_calc1, loss_calc2, part_loss
from model.backbone import resnet_backbone
import torch.nn.functional as F
import gc
from utils import dist


def lr_poly(base_lr, iter, max_iter, power, epoch):
    if epoch <= 2:
        factor = 1
    elif 2 < epoch < 6:
        factor = 1
    else:
        factor = 0.5
    return base_lr * factor * ((1 - float(iter) / max_iter) ** power)


def adjust_learning_rate(cfg, optimizer, i_iter, iter, epoch, max_iter):
    lr = lr_poly(cfg.TRAIN.LEARNING_RATE, iter, max_iter, cfg.TRAIN.POWER, epoch)
    optimizer.param_groups[0]['lr'] = lr
    if i_iter % 3 == 0:
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = 0
    else:
        optimizer.param_groups[0]['lr'] = 0.01 * lr
        optimizer.param_groups[1]['lr'] = lr * 10

    return lr


def train_net(cfg):
    torch.distributed.init_process_group(backend="nccl")
    lrank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(lrank)
    dist.setup_printing_for_distributed(dist.is_main_process())

    train_imgdataset = get_imgdataset(cfg.DATASETS.IMG_DATASET, cfg.DATASETS.IMG_PATH, augment=True, input_size=cfg.MODEL.INPUT_SIZE)
    train_viddataset = get_viddataset(cfg.DATASETS.VID_DATASET, cfg.DATASETS.VID_PATH, ds_type=DatasetType.TRAIN, input_size=cfg.MODEL.INPUT_SIZE)

    # batch sampler
    logging.info('local rank %d / global rank %d successfully built train dataset.' % (lrank, dist.get_rank()))
    train_imgsampler = torch.utils.data.distributed.DistributedSampler(train_imgdataset, shuffle=True)
    train_vidsampler = torch.utils.data.distributed.DistributedSampler(train_viddataset, shuffle=True)

    train_imgdataloader = DataLoader(train_imgdataset, batch_size=cfg.TRAIN.BATCH_SIZE // dist.get_world_size(), sampler=train_imgsampler, num_workers=cfg.WORKERS, drop_last=True)
    train_viddataloader = DataLoader(train_viddataset, batch_size=cfg.TRAIN.BATCH_SIZE // dist.get_world_size(), sampler=train_vidsampler, num_workers=cfg.WORKERS, drop_last=True)
    if dist.is_main_process():
        logging.getLogger().info('successfully set datasets and dataloaders')

    model = PDNet(cfg.MODEL.BACKBONE, train_mode=True)
    part_basis_generator = PartBasisGenerator(512, model.ipda.part_num)
    feature_extractor = resnet_backbone('resnet18', pretrained=True)
    model.train()
    part_basis_generator.train()
    feature_extractor.eval()

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lrank], output_device=lrank)
    part_basis_generator.cuda()
    part_basis_generator = torch.nn.parallel.DistributedDataParallel(part_basis_generator, device_ids=[lrank], output_device=lrank)
    feature_extractor.cuda()
    feature_extractor = torch.nn.parallel.DistributedDataParallel(feature_extractor, device_ids=[lrank], output_device=lrank)
    single_model = model.module
    single_part_basis_generator = part_basis_generator.module

    epoch_resume = 0
    if cfg.TRAIN.RESUME.PATH is not None:
        if cfg.TRAIN.RESUME.MAP_LOCATION is None:
            ckpt = torch.load(cfg.TRAIN.RESUME.PATH)
        elif cfg.TRAIN.RESUME.MAP_LOCATION == '1_0':
            ckpt = torch.load(cfg.TRAIN.RESUME.PATH, map_location={'cuda:1': 'cuda:0'})
        elif cfg.TRAIN.RESUME.MAP_LOCATION == '0_1':
            ckpt = torch.load(cfg.TRAIN.RESUME.PATH, map_location={'cuda:0': 'cuda:1'})
        single_model.load_state_dict(ckpt['model'])
        single_part_basis_generator.load_state_dict(ckpt['part_basis_generator'])
        if cfg.TRAIN.RESUME.EPOCH_RESUME:
            epoch_resume = ckpt['epoch'] + 1
    elif cfg.TRAIN.RESUME.INIT_PATH is not None:
        if cfg.TRAIN.RESUME.MAP_LOCATION is None:
            init_weights = torch.load(cfg.TRAIN.RESUME.INIT_PATH)
        elif cfg.TRAIN.RESUME.MAP_LOCATION == '1_0':
            init_weights = torch.load(cfg.TRAIN.RESUME.INIT_PATH, map_location={'cuda:1': 'cuda:0'})
        elif cfg.TRAIN.RESUME.MAP_LOCATION == '0_1':
            init_weights = torch.load(cfg.TRAIN.RESUME.INIT_PATH, map_location={'cuda:0': 'cuda:1'})
        single_model.bkbone.load_state_dict(init_weights)

    optim = SGD([{'params': get_backbone_params(single_model)},
                 {'params': get_head_params(single_model, single_part_basis_generator)}],
                lr=cfg.TRAIN.LEARNING_RATE, momentum=0.9, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.RESUME.PATH is not None:
        optim.load_state_dict(ckpt['optim'])

    if cfg.TRAIN.MIX_PRECISION:
        scaler = GradScaler()

    batch_num = len(train_viddataloader)
    batch_num += (batch_num + 1) // 2
    if dist.is_main_process():
        logging.getLogger().info("Begin to train")
        logging.getLogger().info("iteration numbers of per epoch: %d", batch_num)
        logging.getLogger().info("max epoch num: %d", cfg.TRAIN.MAX_EPOCH)

    train_imgdataiter = iter(train_imgdataloader)
    for e in range(epoch_resume, cfg.TRAIN.MAX_EPOCH):
        if dist.is_main_process():
            logging.getLogger().info('Epoch %d', e)
            avg_loss = 0
        i_iter = 0
        train_imgdataloader.sampler.set_epoch(e)
        train_viddataloader.sampler.set_epoch(e)
        for batch_idx, data in enumerate(train_viddataloader):
            if i_iter % 3 == 0:
                lr = adjust_learning_rate(cfg, optim, i_iter, i_iter + e * batch_num, e, max_iter=cfg.TRAIN.MAX_EPOCH * batch_num)
                try:
                    img_data = next(train_imgdataiter)
                except StopIteration:
                    train_imgdataiter = iter(train_imgdataloader)
                    img_data = next(train_imgdataiter)
                img, gt = img_data['image'], img_data['mask']
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)

                with autocast() if cfg.TRAIN.MIX_PRECISION else nullcontext():
                    x, sup1, sup2, sup3, sup4, sup_x, _, _, _, _, _, _, _ = model(img, img, mix_precision=cfg.TRAIN.MIX_PRECISION)

                    loss = loss_calc1(x, gt) + 0.8 * loss_calc2(x, gt) + \
                        0.8 * (loss_calc1(sup1, gt) + 0.8 * loss_calc2(sup1, gt) + \
                               loss_calc1(sup2, gt) + 0.8 * loss_calc2(sup2, gt) + \
                               loss_calc1(sup3, gt) + 0.8 * loss_calc2(sup3, gt) + \
                               loss_calc1(sup4, gt) + 0.8 * loss_calc2(sup4, gt) + \
                               loss_calc1(sup_x, gt) + 0.8 * loss_calc2(sup_x, gt))
                    loss = dist.reduce_avg(loss)

                optim.zero_grad()
                if cfg.TRAIN.MIX_PRECISION:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()
                torch.cuda.synchronize()
                if dist.is_main_process():
                    logging.getLogger().info('Epoch %d Batch %d/%d loss=%.10f lr=%.5f', e, i_iter, batch_num, loss.item(), lr)
                    avg_loss += loss.item()
                i_iter += 1

                del x, sup1, sup2, sup3, sup4, sup_x, _, loss
                gc.collect()
                torch.cuda.empty_cache()

            lr = adjust_learning_rate(cfg, optim, i_iter, i_iter + e * batch_num, e, max_iter=cfg.TRAIN.MAX_EPOCH * batch_num)
            img, gt, ref, ref_gt, = data['target'], data['mask'], data['ref'], data['ref_mask']
            img = img.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            ref = ref.cuda(non_blocking=True)
            ref_gt = ref_gt.cuda(non_blocking=True)

            with autocast() if cfg.TRAIN.MIX_PRECISION else nullcontext():
                x, sup1, sup2, sup3, sup4, sup_x, part_x, sup1_y, sup2_y, sup3_y, sup4_y, sup_y, part_y = model(img, ref, mix_precision=cfg.TRAIN.MIX_PRECISION)
                basis = part_basis_generator(0)
                with torch.no_grad():
                    _, _, _, _, feat_x = feature_extractor(img)
                    feat_x = F.interpolate(feat_x, size=sup1.size()[2:], mode='bilinear', align_corners=True)
                    _, _, _, _, feat_y = feature_extractor(ref)
                    feat_y = F.interpolate(feat_y, size=sup1_y.size()[2:], mode='bilinear', align_corners=True)

                loss = loss_calc1(x, gt) + 0.8 * loss_calc2(x, gt) + \
                       0.8 * ((loss_calc1(sup1, gt) + 0.8 * loss_calc2(sup1, gt) + \
                              loss_calc1(sup2, gt) + 0.8 * loss_calc2(sup2, gt) + \
                              loss_calc1(sup3, gt) + 0.8 * loss_calc2(sup3, gt) + \
                              loss_calc1(sup4, gt) + 0.8 * loss_calc2(sup4, gt) + \
                              loss_calc1(sup_x, gt) + 0.8 * loss_calc2(sup_x, gt) + \
                              part_loss(part_x, gt, sup_x, feat_x, basis)) + \
                       0.8 * (loss_calc1(sup1_y, ref_gt) + 0.8 * loss_calc2(sup1_y, ref_gt) + \
                              loss_calc1(sup2_y, ref_gt) + 0.8 * loss_calc2(sup2_y, ref_gt) + \
                              loss_calc1(sup3_y, ref_gt) + 0.8 * loss_calc2(sup3_y, ref_gt) + \
                              loss_calc1(sup4_y, ref_gt) + 0.8 * loss_calc2(sup4_y, ref_gt) + \
                              loss_calc1(sup_y, ref_gt) + 0.8 * loss_calc2(sup_y, ref_gt) + \
                              part_loss(part_y, ref_gt, sup_y, feat_y, basis)))
                loss = dist.reduce_avg(loss)

            optim.zero_grad()
            if cfg.TRAIN.MIX_PRECISION:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            torch.cuda.synchronize()
            if dist.is_main_process():
                logging.getLogger().info('Epoch %d Batch %d/%d loss=%.10f lr=%.5f', e, i_iter, batch_num, loss.item(), lr)
                avg_loss += loss.item()
            i_iter += 1

            del x, sup1, sup2, sup3, sup4, sup_x, part_x, sup1_y, sup2_y, sup3_y, sup4_y, sup_y, part_y, loss
            gc.collect()
            torch.cuda.empty_cache()

        if dist.is_main_process():
            avg_loss /= batch_num
            logging.getLogger().info('Epoch %d average loss=%.4f', e, avg_loss)
        torch.distributed.barrier()

        if dist.is_main_process():
            ckpt = OrderedDict()
            ckpt['epoch'] = e
            ckpt['model'] = single_model.state_dict()
            ckpt['optim'] = optim.state_dict()
            ckpt['part_basis_generator'] = single_part_basis_generator.state_dict()
            os.makedirs(cfg.TRAIN.CKPT_PATH, exist_ok=True)
            torch.save(ckpt, os.path.join(cfg.TRAIN.CKPT_PATH, 'ckpt-epoch-%04d.pth' % e))
            logging.getLogger().info('Save checkpoint for epoch %d successfully', e)
