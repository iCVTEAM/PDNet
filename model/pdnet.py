import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import conv1x1, conv3x3
from torch.cuda.amp import autocast
from contextlib import nullcontext
from .backbone import resnet_backbone


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.Softmax) or isinstance(m, nn.Sigmoid):
            pass
        else:
            m.initialize()


class FeatureFusion(nn.Module):
    def __init__(self, channel):
        super(FeatureFusion, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        out = self.relu(self.bn_1(self.conv_1(out)))
        out = self.relu(self.bn_2(self.conv_2(out)))
        return out

    def initialize(self):
        weight_init(self)


class PartSep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PartSep, self).__init__()
        channels = in_channels
        self.conv_1 = conv1x1(in_channels, channels)
        self.bn_1 = nn.BatchNorm2d(channels)
        self.conv_2 = conv3x3(channels, channels)
        self.bn_2 = nn.BatchNorm2d(channels)
        self.conv_3 = conv1x1(channels, out_channels)
        self.bn_3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)))
        x = self.relu(self.bn_3(self.conv_3(x)))
        return x

    def initialize(self):
        weight_init(self)


class IPDA(nn.Module):
    def __init__(self, in_channels, out_channels, train_mode=False):
        super(IPDA, self).__init__()
        self.train_mode = train_mode

        self.sup_adapt1 = nn.Sequential(conv3x3(in_channels, in_channels), nn.BatchNorm2d(in_channels))
        self.sup_adapt2 = nn.Sequential(conv1x1(in_channels, out_channels), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

        self.part_num = 5

        self.part_sep_conv = nn.ModuleList([PartSep(out_channels, out_channels) for _ in range(self.part_num)])
        self.sup_head = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)

        self.part_q_conv = nn.ModuleList([conv1x1(out_channels, out_channels // 2) for _ in range(self.part_num)])
        self.part_k_conv = nn.ModuleList([conv1x1(out_channels, out_channels // 2) for _ in range(self.part_num)])
        self.part_v_conv = nn.ModuleList([conv1x1(out_channels, out_channels // 2) for _ in range(self.part_num)])
        self.part_att_softmax = nn.Softmax(dim=2)
        self.part_att_conv = nn.ModuleList(
            [nn.Sequential(conv1x1(out_channels // 2, out_channels), nn.BatchNorm2d(out_channels))
             for _ in range(self.part_num)])

        self.part_sep_pred = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        part_x = []
        part_y = []
        part_x_pred = []
        part_x_att = []
        x_low = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        y_low = F.interpolate(y, scale_factor=0.25, mode='bilinear', align_corners=True)

        x_low = self.relu(self.sup_adapt1(x_low))
        x_low = self.relu(self.sup_adapt2(x_low))
        y_low = self.relu(self.sup_adapt1(y_low))
        y_low = self.relu(self.sup_adapt2(y_low))

        for i in range(self.part_num):
            part_x.append(self.part_sep_conv[i](x_low))
            part_y.append(self.part_sep_conv[i](y_low))
            part_x_pred.append(self.part_sep_pred(part_x[i]))
        part_x_pred_bin = torch.cat(part_x_pred, dim=1)
        part_x_pred_bin = self.softmax(part_x_pred_bin)
        x_low_pred = self.sup_head(x_low)
        x_low_pred_bin = self.sigmoid(x_low_pred)
        part_x_pred_bin = part_x_pred_bin * x_low_pred_bin

        for i in range(self.part_num):
            h = part_x[i].shape[2]

            part_q_i = self.part_q_conv[i](part_x[i])
            part_q_i = part_q_i.reshape(part_q_i.shape[0], part_q_i.shape[1], -1).transpose(1, 2)
            part_k_i = self.part_k_conv[i](part_y[i])
            part_k_i = part_k_i.reshape(part_k_i.shape[0], part_k_i.shape[1], -1)
            part_v_i = self.part_v_conv[i](part_y[i])
            part_v_i = part_v_i.reshape(part_v_i.shape[0], part_v_i.shape[1], -1).transpose(1, 2)

            part_qk_i = torch.matmul(part_q_i, part_k_i)
            part_qk_i = self.part_att_softmax(part_qk_i)

            part_x_att_i = torch.matmul(part_qk_i, part_v_i).transpose(1, 2)
            part_x_att_i = part_x_att_i.contiguous().view(part_x_att_i.shape[0], part_x_att_i.shape[1], h, -1)
            part_x_att_i = self.relu(self.part_att_conv[i](part_x_att_i))
            part_x_att_i = (part_x[i] + part_x_att_i) * part_x_pred_bin[:, i, :, :].unsqueeze(1)
            part_x_att.append(part_x_att_i.unsqueeze(0))
        part_x_att = torch.cat(part_x_att, dim=0)
        part_x_fusion = part_x_att.sum(dim=0)

        if self.train_mode:
            sup_y = self.sup_head(y_low)
            part_y_pred = []
            for i in range(self.part_num):
                part_y_pred.append(self.part_sep_pred(part_y[i]))
            return part_x_fusion, x_low_pred, part_x_pred, sup_y, part_y_pred
        else:
            return part_x_fusion

    def initialize(self):
        weight_init(self)


class SpatiotemporalFusion(nn.Module):
    def __init__(self, channel, t_channel):
        super(SpatiotemporalFusion, self).__init__()
        self.conv_t = conv1x1(t_channel, channel)
        self.bn_t = nn.BatchNorm2d(channel)
        self.conv_t1 = conv1x1(channel, channel)
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, s, t):
        t = F.interpolate(t, size=s.size()[2:], mode='bilinear', align_corners=True)
        t = self.relu(self.bn_t(self.conv_t(t)))
        t1 = self.conv_t1(t)
        x = s + t + s * t1
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)))
        return x

    def initialize(self):
        weight_init(self)


class PartBasisGenerator(nn.Module):
    def __init__(self, feature_channel, part_num):
        super(PartBasisGenerator, self).__init__()
        self.w = nn.Parameter(torch.abs(torch.FloatTensor(part_num, feature_channel).normal_()))

    def forward(self, x=None):
        out = F.relu(self.w)
        return out


class PDNet(nn.Module):
    def __init__(self, bkbone, train_mode=False):
        super(PDNet, self).__init__()
        self.train_mode = train_mode

        self.bkbone = resnet_backbone(bkbone, pretrained=self.train_mode)

        self.l5_adapt = nn.Sequential(conv1x1(2048, 128), nn.BatchNorm2d(128))
        self.l4_adapt = nn.Sequential(conv1x1(1024, 128), nn.BatchNorm2d(128))
        self.l3_adapt = nn.Sequential(conv1x1(512, 128), nn.BatchNorm2d(128))
        self.l2_adapt = nn.Sequential(conv1x1(256, 128), nn.BatchNorm2d(128))

        self.relu = nn.ReLU(inplace=True)

        self.fuse4 = FeatureFusion(128)
        self.fuse3 = FeatureFusion(128)
        self.fuse2 = FeatureFusion(128)

        self.ipda = IPDA(128, 64, train_mode=self.train_mode)

        self.spatiotemporal_fusion = SpatiotemporalFusion(128, 64)
        self.head = nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=True)

        self.head_1 = nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=True)
        self.head_2 = nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=True)
        self.head_3 = nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=True)
        self.head_4 = nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=True)

        if self.train_mode:
            weight_init(self)

    def forward(self, x, y, shape=None, mix_precision=False):
        with autocast() if mix_precision else nullcontext():
            if shape is None:
                shape = x.shape[-2:]
            _, l2, l3, l4, l5 = self.bkbone(x)
            _, l2_y, l3_y, l4_y, l5_y = self.bkbone(y)

            x5 = self.relu(self.l5_adapt(l5))
            x5 = F.interpolate(x5, size=l4.size()[2:], mode='bilinear', align_corners=True)

            y5 = self.relu(self.l5_adapt(l5_y))
            y5 = F.interpolate(y5, size=l4_y.size()[2:], mode='bilinear', align_corners=True)

            x4 = self.relu(self.l4_adapt(l4))
            x4 = self.fuse4(x5, x4)
            x4 = F.interpolate(x4, size=l3.size()[2:], mode='bilinear', align_corners=True)

            y4 = self.relu(self.l4_adapt(l4_y))
            y4 = self.fuse4(y5, y4)
            y4 = F.interpolate(y4, size=l3_y.size()[2:], mode='bilinear', align_corners=True)

            x3 = self.relu(self.l3_adapt(l3))
            x3 = self.fuse3(x4, x3)
            x3 = F.interpolate(x3, size=l2.size()[2:], mode='bilinear', align_corners=True)

            y3 = self.relu(self.l3_adapt(l3_y))
            y3 = self.fuse3(y4, y3)
            y3 = F.interpolate(y3, size=l2_y.size()[2:], mode='bilinear', align_corners=True)

            x2 = self.relu(self.l2_adapt(l2))
            x2 = self.fuse2(x3, x2)

            y2 = self.relu(self.l2_adapt(l2_y))
            y2 = self.fuse2(y3, y2)

            if self.train_mode:
                part_x_att, x_low_pred, part_x_pred, sup_y, part_y_pred = self.ipda(x2, y2)
            else:
                part_x_att = self.ipda(x2, y2)
            x = self.spatiotemporal_fusion(x2, part_x_att)

            pred = F.interpolate(self.head(x), size=shape, mode='bilinear', align_corners=True)
            if self.train_mode:

                sup_l2 = F.interpolate(self.head_1(x2), size=shape, mode='bilinear', align_corners=True)
                sup_l3 = F.interpolate(self.head_2(x3), size=shape, mode='bilinear', align_corners=True)
                sup_l4 = F.interpolate(self.head_3(x4), size=shape, mode='bilinear', align_corners=True)
                sup_l5 = F.interpolate(self.head_4(x5), size=shape, mode='bilinear', align_corners=True)
                sup_x = F.interpolate(x_low_pred, size=shape, mode='bilinear', align_corners=True)

                sup_l2_y = F.interpolate(self.head_1(y2), size=shape, mode='bilinear', align_corners=True)
                sup_l3_y = F.interpolate(self.head_2(y3), size=shape, mode='bilinear', align_corners=True)
                sup_l4_y = F.interpolate(self.head_3(y4), size=shape, mode='bilinear', align_corners=True)
                sup_l5_y = F.interpolate(self.head_4(y5), size=shape, mode='bilinear', align_corners=True)
                sup_y = F.interpolate(sup_y, size=shape, mode='bilinear', align_corners=True)

                for i in range(len(part_x_pred)):
                    part_x_pred[i] = F.interpolate(part_x_pred[i], size=shape, mode='bilinear', align_corners=True)
                    part_y_pred[i] = F.interpolate(part_y_pred[i], size=shape, mode='bilinear', align_corners=True)

                return pred, sup_l2, sup_l3, sup_l4, sup_l5, sup_x, part_x_pred, sup_l2_y, sup_l3_y, sup_l4_y, sup_l5_y, sup_y, part_y_pred
            else:
                return pred
