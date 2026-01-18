import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime
import re
from torch.nn.modules.loss import _Loss
from torchvision import models


class SumLoss(nn.Module):
    def __init__(self, n_classes):
        super(SumLoss, self).__init__()
        self.n_classes = n_classes

    def data_fidelity(self, network_output):
        # basic material attenuation coefficients
        coefficient_dict = torch.tensor(
            [[-0.11171025260029717, -0.0870228187919463], [0.061890800299177255, 0.06287079910380881],
             [0.4346875, 0.22676576576576576], [-1., -1.]]).cuda()
        output_reshape = network_output.permute(0, 2, 3, 1)
        rebuild_CT = torch.matmul(output_reshape, coefficient_dict).permute(0, 3, 1, 2)
        return rebuild_CT

    def Gradient_Net(self, network_output, is_input=False):
        kernel_x = [[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).repeat_interleave(network_output.shape[1],
                                                                                           dim=0).cuda()

        kernel_y = [[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).repeat_interleave(network_output.shape[1],
                                                                                           dim=0).cuda()

        kernel_45 = [[-10., -3., 0.], [-3., 0., 3.], [0., 3., 10.]]
        kernel_45 = torch.FloatTensor(kernel_45).unsqueeze(0).unsqueeze(0).repeat_interleave(network_output.shape[1],
                                                                                             dim=0).cuda()

        kernel_135 = [[0., 3., 10.], [-3., 0., 3.], [-10., -3., 0.]]
        kernel_135 = torch.FloatTensor(kernel_135).unsqueeze(0).unsqueeze(0).repeat_interleave(network_output.shape[1],
                                                                                               dim=0).cuda()
        if is_input:
            kernel_sharp = [[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]]
            kernel_sharp = torch.FloatTensor(kernel_sharp).unsqueeze(0).unsqueeze(0).repeat_interleave(
                network_output.shape[1], dim=0).cuda()
            weight_sharp = nn.Parameter(data=kernel_sharp, requires_grad=False)
            network_output = F.conv2d(network_output, weight_sharp, groups=network_output.shape[1], padding=1)

        weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
        weight_45 = nn.Parameter(data=kernel_45, requires_grad=False)
        weight_135 = nn.Parameter(data=kernel_135, requires_grad=False)

        grad_x = F.conv2d(network_output, weight_x, groups=network_output.shape[1])
        grad_x = (grad_x - torch.min(grad_x)) / (torch.max(grad_x) - torch.min(grad_x))
        grad_y = F.conv2d(network_output, weight_y, groups=network_output.shape[1])
        grad_y = (grad_y - torch.min(grad_y)) / (torch.max(grad_y) - torch.min(grad_y))
        grad_45 = F.conv2d(network_output, weight_45, groups=network_output.shape[1])
        grad_45 = (grad_45 - torch.min(grad_45)) / (torch.max(grad_45) - torch.min(grad_45))
        grad_135 = F.conv2d(network_output, weight_135, groups=network_output.shape[1])
        grad_135 = (grad_135 - torch.min(grad_135)) / (torch.max(grad_135) - torch.min(grad_135))
        return torch.abs(grad_x), torch.abs(grad_y), torch.abs(grad_45), torch.abs(grad_135)

    def edge_preserve(self, network_output, network_input):
        Gradient_line, Gradient_column, Gradient_45, Gradient_135 = self.Gradient_Net(network_output, False)
        gradient_line_100, gradient_column_100, gradient_45_100, gradient_135_100 = self.Gradient_Net(
            network_input[:, 0, :, :].unsqueeze(dim=1), True)
        gradient_line_140, gradient_column_140, gradient_45_140, gradient_135_140 = self.Gradient_Net(
            network_input[:, 1, :, :].unsqueeze(dim=1), True)
        Gradient_line_100 = gradient_line_100.repeat_interleave(network_output.shape[1], dim=1)
        Gradient_line_140 = gradient_line_140.repeat_interleave(network_output.shape[1], dim=1)
        Gradient_column_100 = gradient_column_100.repeat_interleave(network_output.shape[1], dim=1)
        Gradient_column_140 = gradient_column_140.repeat_interleave(network_output.shape[1], dim=1)
        Gradient_45_100 = gradient_45_100.repeat_interleave(network_output.shape[1], dim=1)
        Gradient_45_140 = gradient_45_140.repeat_interleave(network_output.shape[1], dim=1)
        Gradient_135_100 = gradient_135_100.repeat_interleave(network_output.shape[1], dim=1)
        Gradient_135_140 = gradient_135_140.repeat_interleave(network_output.shape[1], dim=1)
        return Gradient_line, Gradient_column, Gradient_45, Gradient_135, Gradient_line_100, Gradient_line_140, Gradient_column_100, Gradient_column_140, Gradient_45_100, Gradient_45_140, Gradient_135_100, Gradient_135_140

    def edge_preserve_denoise(self, network_output, network_input):
        Gradient_line, Gradient_column, Gradient_45, Gradient_135 = self.Gradient_Net(network_output, False)
        Gradient_line_input, Gradient_column_input, Gradient_45_input, Gradient_135_input = self.Gradient_Net(
            network_input[:, 0, :, :].unsqueeze(dim=1), True)

        return Gradient_line, Gradient_column, Gradient_45, Gradient_135, Gradient_line_input, Gradient_column_input, Gradient_45_input, Gradient_135_input

    def sparsity_constraint(self, network_output):
        loss_sparsity = torch.mean(torch.norm(network_output, p=1, dim=1))
        return loss_sparsity


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
        raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]


class ResNet50FeatureExtractor(nn.Module):

    def __init__(self, blocks=[1, 2, 3, 4], pretrained=False, progress=True, **kwargs):
        super(ResNet50FeatureExtractor, self).__init__()
        self.model = models.resnet50(pretrained, progress, **kwargs)
        del self.model.avgpool
        del self.model.fc
        self.blocks = blocks

    def forward(self, x):
        feats = list()

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        if 1 in self.blocks:
            feats.append(x)

        x = self.model.layer2(x)
        if 2 in self.blocks:
            feats.append(x)

        x = self.model.layer3(x)
        if 3 in self.blocks:
            feats.append(x)

        x = self.model.layer4(x)
        if 4 in self.blocks:
            feats.append(x)

        return feats


class CompoundLoss(_Loss):

    def __init__(self, blocks=[1, 2, 3, 4]):
        super(CompoundLoss, self).__init__()

        self.blocks = blocks
        self.model = ResNet50FeatureExtractor(pretrained=True)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        loss_value = 0
        loss = 0
        for i in range(input.shape[1]):
            input_feats = self.model(torch.stack([input[:, i, :, :], input[:, i, :, :], input[:, i, :, :]], dim=1))
            target_feats = self.model(torch.stack([target[:, i, :, :], target[:, i, :, :], target[:, i, :, :]], dim=1))

            feats_num = len(self.blocks)
            for idx in range(feats_num):
                loss_value += self.criterion(input_feats[idx], target_feats[idx])
            loss_value /= feats_num
            loss += loss_value

        return loss / input.shape[1]


def compute_laplacian_gradient(output):
    laplacian_kernel = (torch.FloatTensor([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
                        .unsqueeze(0).unsqueeze(0).repeat_interleave(output.shape[1],dim=0).cuda())
    laplacian = F.conv2d(output, laplacian_kernel, groups=output.shape[1], padding=1)
    laplacian_abs = torch.abs(laplacian)
    min_val = torch.min(laplacian_abs)
    max_val = torch.max(laplacian_abs)
    laplacian_abs_norm = (laplacian_abs - min_val) / (max_val - min_val)
    return laplacian_abs_norm


def bound_loss(output, target):
    laplacian = compute_laplacian_gradient(target)
    diff = torch.abs(output - target)
    weighted_diff = laplacian * diff
    loss = torch.mean(weighted_diff)

    return loss