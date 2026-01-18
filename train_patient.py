import torch.backends.cudnn as cudnn
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from utils import SumLoss, CompoundLoss, bound_loss, most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from datetime import datetime
import random
from PIL import Image
import glob
from networks.unet_model import *
from networks.unet_TransNeXt_mmdx import transnext_tiny


class DataLoader_Imagenet_val_train(Dataset):
    def __init__(self, data_dir, label_dir):
        super(DataLoader_Imagenet_val_train, self).__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.train_fns_100 = sorted(glob.glob(os.path.join(data_dir, '100kv/*.npy')))
        self.train_fns_140 = sorted(glob.glob(os.path.join(data_dir, '140kv/*.npy')))
        self.label_fns = sorted(glob.glob(os.path.join(label_dir, '*.npy')))
        print('fetch {} samples for training/testing'.format(len(self.train_fns_100)))
        print('fetch {} samples for label'.format(len(self.label_fns)))

    def __getitem__(self, index):
        # fetch image
        if len(self.label_fns) != len(self.train_fns_100):
            raise RuntimeError("Unequal number between 100kv files and 140kv files!")

        fn_100 = self.train_fns_100[index]
        fn_140 = self.train_fns_140[index]
        fn_label = self.label_fns[index]
        label_each = np.load(fn_label)
        im_100 = np.load(fn_100)
        im_140 = np.load(fn_140)
        im_100 = np.array(im_100, dtype=np.float32)
        im_140 = np.array(im_140, dtype=np.float32)
        label_each = np.array(label_each, dtype=np.float32)

        im = np.stack((im_100, im_140), axis=2)
        im = np.reshape(im, (im_100.shape[0], im_100.shape[1], 2))
        label_each = np.transpose(label_each, (1, 2, 0))
        transformer = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(args.img_size)])
        im = transformer(im)
        label_each = transformer(label_each)
        # print(label_each)
        return im, label_each

    def __len__(self):
        return len(self.train_fns_100)


def checkpoint(net, epoch, name, type):
    save_model_path = os.path.join(args.save_model_path, args.log_name, systime)  # 拼接路径名
    os.makedirs(save_model_path, exist_ok=True)
    model_name = '{}-{:03d}-{}.pth'.format(name, epoch, type)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='./datasets/DECT_data/train', help='root dir for train data')
parser.add_argument('--label_dir', type=str,
                    default='./datasets/pseudo', help='root dir for label data')
parser.add_argument('--log_name', type=str,
                    default='decompose', help='networks function')
parser.add_argument('--test_dir', type=str,
                    default='./datasets/DECT_data/test', help='root dir for test data')
parser.add_argument('--save_model_path', type=str,
                    default='./results_patient/', help='dir for networks weight')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of networks')
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
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training')
parser.add_argument("--DATE_FORMAT", type=str, default='%A_%d_%B_%Y_%Hh_%Mm_%Ss', help='time format')
parser.add_argument("--parameter3", type=int, default=0.01, help='loss hyperparameter3')

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

TrainingDataset = DataLoader_Imagenet_val_train(args.data_dir, args.label_dir)
TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=2,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

model = transnext_tiny(in_chans=1, num_classes=4, Ablation_mix=False, Ablation_att=False).cuda()

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
sumloss = SumLoss(n_classes=4)
iterator = tqdm(range(max_epoch), ncols=70)


for epoch_num in iterator:
    model.train()
    if args.resume:
        if epoch_num <= resume_epoch:
            iter_num = resume_epoch * len(TrainingLoader)
            continue
    loss_train_sum = 0

    for i_batch, ima in enumerate(TrainingLoader):
        st = time.time()
        image_batch, label_batch = ima
        image_batch = image_batch.cuda()
        label_batch = label_batch.cuda()
        outputs = model(image_batch)
        loss1 = Loss_mse(outputs, label_batch)

        Rebuild_CT = sumloss.data_fidelity(outputs)
        loss2 = Loss_mse(Rebuild_CT, image_batch)
        loss3 = bound_loss(Rebuild_CT, image_batch)

        alpha = np.exp(-5 * iter_num / max_iterations)
        loss = alpha * loss1 + (1 - alpha) * loss2 + args.parameter3 * loss3
        loss_train_sum += loss

        optimizer.zero_grad()
        loss.backward()  # nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()
        lr_ = lr_scheduler.get_last_lr()[0]

        iter_num = iter_num + 1
        if (i_batch + 1) % 20 == 0:
            print(
                'epoch={:d}/{:d}, iteration={:d}/{:d}, learning_rate={:06f}, alpha={:.6f}, '
                'Loss={:.6f}, Time={:.4f}'
                .format(epoch_num, max_epoch - 1, i_batch + 1, len(TrainingLoader), lr_, alpha,
                        loss.item(), time.time() - st))
    lr_scheduler.step()
    checkpoint(model, epoch_num, "model", "regular")