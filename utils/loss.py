import torch
import torch.nn.functional as F


def loss_calc1(pred, label):
    eps = 1
    labels = torch.ge(label, 0.5).float()
    batch_size = label.shape
    num_labels_pos = torch.sum(labels) + eps
    batch_1 = batch_size[0] * batch_size[2]
    batch_1 = batch_1 * batch_size[3]
    weight_1 = torch.div(num_labels_pos, batch_1)
    weight_1 = torch.reciprocal(weight_1)
    weight_22 = torch.mul(weight_1, torch.ones_like(label))
    return F.binary_cross_entropy_with_logits(pred, label, weight=weight_22)


def loss_calc2(pred, label):
    pred = torch.sigmoid(pred)
    return F.l1_loss(pred, label)


def gravity_center(pred):
    eps = 1e-7
    x = torch.arange(pred.shape[-2], device=pred.device) / pred.shape[-2] * 2 - 1.0
    y = torch.arange(pred.shape[-1], device=pred.device) / pred.shape[-1] * 2 - 1.0
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)

    ct_x = (grid_x * pred).sum(dim=(-2, -1)) / (pred.sum(dim=(-2, -1)) + eps)
    ct_y = (grid_y * pred).sum(dim=(-2, -1)) / (pred.sum(dim=(-2, -1)) + eps)
    ct = torch.cat([ct_x, ct_y], dim=-1)
    return ct


def get_variance(pred, x_c, y_c):
    eps = 1e-7
    x = torch.arange(pred.shape[-2], device=pred.device) / pred.shape[-2] * 2 - 1.0
    y = torch.arange(pred.shape[-1], device=pred.device) / pred.shape[-1] * 2 - 1.0
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)

    x_c = x_c.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    y_c = y_c.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    v_x_map = (grid_x - x_c) * (grid_x - x_c)
    v_y_map = (grid_y - y_c) * (grid_y - y_c)

    v_x = (pred * v_x_map).sum(dim=(-2, -1)) / (pred.sum(dim=(-2, -1)) + eps)
    v_y = (pred * v_y_map).sum(dim=(-2, -1)) / (pred.sum(dim=(-2, -1)) + eps)
    return v_x, v_y


def concentration_loss(part):
    part_num = part.shape[1]
    loss_sum = 0.
    for i in range(part_num):
        part_map = part[:, i, :, :].unsqueeze(1)
        ct_part = gravity_center(part_map)
        x_c, y_c = ct_part[:, 0], ct_part[:, 1]
        v_x, v_y = get_variance(part_map, x_c, y_c)
        loss_sum += v_x + v_y
    return loss_sum.mean()


def semantic_consistency_loss(features, pred, basis):
    weighted_basis = torch.matmul(pred.permute(0, 2, 3, 1), basis).permute(0, 3, 1, 2)
    return F.mse_loss(weighted_basis, features)


def featureL2Norm(feature):
    eps = 1e-7
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + eps, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


def orthonomal_loss(w):
    K, C = w.shape
    w_norm = featureL2Norm(w)
    WWT = torch.matmul(w_norm, w_norm.transpose(0, 1))
    return F.mse_loss(WWT - torch.eye(K, device=w.device), torch.zeros(K, K, device=w.device), reduction='sum')


def area_loss(part_pred):
    part_pred = part_pred.sum(dim=(-2, -1))
    part_pred = F.normalize(part_pred, dim=1)
    var = torch.var(part_pred, dim=1)
    return var.mean()


def part_loss(part_pred, label, mask_pred, feat, basis):
    mask_pred = torch.sigmoid(mask_pred)
    part_pred = torch.cat(part_pred, dim=1)
    part_pred_bin = torch.softmax(part_pred, dim=1)
    part_pred_bin_labeled = part_pred_bin * label
    part_pred_bin_masked = part_pred_bin * mask_pred
    feat_labeled = feat * label
    feat_masked = feat * mask_pred

    concen_loss = concentration_loss(part_pred_bin_labeled) + concentration_loss(part_pred_bin_masked) / 2
    sem_cons_loss = semantic_consistency_loss(feat_labeled, part_pred_bin_labeled, basis) + semantic_consistency_loss(feat_masked, part_pred_bin_masked, basis) / 2
    orth_loss = orthonomal_loss(basis)
    var_loss = (area_loss(part_pred_bin_labeled) + area_loss(part_pred_bin_masked)) / 2

    return concen_loss + sem_cons_loss + orth_loss + var_loss
