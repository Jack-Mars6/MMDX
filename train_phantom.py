import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from torch.autograd import Variable
# from config import get_config
import argparse
import sys
import time
import numpy as np
from math import exp
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms
from utils import SumLoss, CompoundLoss, bound_loss,most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from datetime import datetime
import random
from PIL import Image
import glob
from networks.unet_TransNeXt_mmdx import transnext_tiny


class DataLoader_Imagenet_val_train(Dataset):
    def __init__(self, data_dir):
        super(DataLoader_Imagenet_val_train, self).__init__()
        self.data_dir = data_dir
        self.label_dir = data_dir
        self.Adipose = np.load(os.path.join(self.data_dir, 'Adipose.npy'))
        self.Calcification = np.load(os.path.join(self.data_dir, 'Calcification.npy'))
        self.Fibroglandular = np.load(os.path.join(self.data_dir, 'Fibroglandular.npy'))
        self.Air = np.load(os.path.join(self.data_dir, 'Air.npy'))

        self.im_high = np.load(os.path.join(self.data_dir, 'train_data_high.npy'))
        self.im_low = np.load(os.path.join(self.data_dir, 'train_data_low.npy'))
        print('fetch {} samples for training'.format(self.im_high.shape[0]))
        print('fetch {} samples for label'.format(self.Air.shape[0]))

    def __getitem__(self, index):
        # fetch image
        if self.im_high.shape[0] != self.Air.shape[0]:
            raise RuntimeError("Unequal number between files and labels!")

        im_100 = self.im_low[index]
        im_140 = self.im_high[index]

        label_each = np.stack(
            (self.Adipose[index], self.Calcification[index], self.Fibroglandular[index], self.Air[index]), axis=0)

        im_100 = np.array(im_100, dtype=np.float32)
        im_140 = np.array(im_140, dtype=np.float32)
        im_100 = 2 * (((im_100 - np.min(im_100)) / (np.max(im_100) - np.min(im_100))) - 0.5)
        im_140 = 2 * (((im_140 - np.min(im_140)) / (np.max(im_140) - np.min(im_140))) - 0.5)
        label_each = np.array(label_each, dtype=np.float32)
        im = np.stack((im_100, im_140), axis=2)
        im = np.reshape(im, (im_100.shape[0], im_100.shape[1], 2))
        label_each = np.transpose(label_each, (1, 2, 0))

        transformer = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(args.img_size)])
        im = transformer(im)
        label_each = transformer(label_each)
        return im, label_each

    def __len__(self):
        return self.im_high.shape[0]


def checkpoint(net, epoch, name, type):
    save_model_path = os.path.join(args.save_model_path, args.log_name, systime)  # 拼接路径名
    os.makedirs(save_model_path, exist_ok=True)
    model_name = '{}-{:03d}-{}.pth'.format(name, epoch, type)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(),
               save_model_path)  # torch.save(state, dir) dir表示保存文件的路径+保存文件名（eg：/home/qinying/Desktop/modelpara.pth）
    print('Checkpoint saved to {}'.format(save_model_path))


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='./datasets/data/train', help='root dir for train data')
parser.add_argument('--test_dir', type=str,
                    default='./datasets/data/test', help='root dir for label data')
parser.add_argument('--log_name', type=str,
                    default='decompose', help='networks function')
parser.add_argument('--save_model_path', type=str,
                    default='./results_phantom/', help='dir for networks weight')
parser.add_argument('--kits_dataset', type=str,
                    default='DECT', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of networks')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation networks learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of networks input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training')  # 刚开始训练False，如果断了则需要变为True
parser.add_argument("--DATE_FORMAT", type=str, default='%A_%d_%B_%Y_%Hh_%Mm_%Ss', help='time format')
parser.add_argument("--parameter2", type=int, default=0.01, help='loss hyperparameter3')

parser.add_argument("--Ablation_mix", type=bool, default=False, help='Ablation study for incorporation')
parser.add_argument("--Ablation_att", type=bool, default=False, help='Ablation study for attention')

args = parser.parse_args()
if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

base_lr = args.base_lr
num_classes = args.num_classes
batch_size = args.batch_size * args.n_gpu

systime = datetime.now().strftime(args.DATE_FORMAT)

TrainingDataset = DataLoader_Imagenet_val_train(args.data_dir)
TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=0,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

model = transnext_tiny(in_chans=1, num_classes=4, Ablation_mix=args.Ablation_mix, Ablation_att=args.Ablation_att).cuda()

if args.n_gpu > 1:
    model = nn.DataParallel(model)
optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-5)

if args.resume:
    recent_folder = most_recent_folder(os.path.join(args.save_model_path, args.log_name), fmt=args.DATE_FORMAT)
    if not recent_folder:
        raise Exception('no recent folder were found')

    checkpoint_path = os.path.join(args.save_model_path, args.log_name, recent_folder)
    best_weights = best_acc_weights(os.path.join(args.save_model_path, args.log_name, recent_folder))
    if best_weights:
        weights_path = os.path.join(args.save_model_path, args.log_name, recent_folder, best_weights)
        print('found best acc weights file:{}'.format(weights_path))
        print('load best training file...')
        model.load_state_dict(torch.load(weights_path))

    recent_weights_file = most_recent_weights(os.path.join(args.save_model_path, args.log_name, recent_folder))
    if not recent_weights_file:
        raise Exception('no recent weights file were found')
    weights_path = os.path.join(args.save_model_path, args.log_name, recent_folder, recent_weights_file)
    print('loading weights file {} to resume training.....'.format(weights_path))
    model.load_state_dict(torch.load(weights_path))

    resume_epoch = last_epoch(os.path.join(args.save_model_path, args.log_name, recent_folder))
else:
    checkpoint(model, 0, "model", "regular")

print('init finish')

iter_num = 0
max_epoch = args.max_epochs
max_iterations = args.max_epochs * len(TrainingLoader)
best_performance = 0.0
Loss_mse = nn.MSELoss()
iterator = tqdm(range(max_epoch), ncols=70)

loss_list = []
for epoch_num in iterator:
    model.train()
    if args.resume:
        if epoch_num <= resume_epoch:
            continue
    loss_train_sum = 0

    for i_batch, ima in enumerate(TrainingLoader):
        st = time.time()
        image_batch, label_batch = ima
        image_batch = image_batch.cuda()
        label_batch = label_batch.cuda()
        outputs = model(image_batch)

        loss1 = Loss_mse(outputs, label_batch)
        loss2 = bound_loss(outputs, label_batch)

        loss = loss1 + args.parameter2 * loss2
        loss_train_sum += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_ = lr_scheduler.get_last_lr()[0]

        iter_num = iter_num + 1
        if (i_batch + 1) % 20 == 0:
            print(
                'epoch={:d}/{:d}, iteration={:d}/{:d}, learning_rate={:06f}, Loss={:.6f}, Time={:.4f}'
                .format(epoch_num, max_epoch - 1, i_batch + 1, len(TrainingLoader), lr_,
                        loss.item(), time.time() - st))
    lr_scheduler.step()
    checkpoint(model, epoch_num, "model", "regular")