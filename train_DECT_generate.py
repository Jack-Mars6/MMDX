import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from torch.autograd import Variable
import argparse
import os
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
from utils import SumLoss, most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from datetime import datetime
import random
from PIL import Image
import glob
from networks.DECT_CNN import *
from networks.vmunet import VMUNet_conv
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader_val_train(Dataset):
    def __init__(self, data_dir):
        super(DataLoader_val_train, self).__init__()
        self.data_dir = data_dir
        self.train_fns_100 = sorted(glob.glob(
            os.path.join(data_dir, '100kv/*.npy')))
        self.train_fns_140 = sorted(glob.glob(os.path.join(data_dir, '140kv/*.npy')))
        print('fetch {} samples for training/testing'.format(len(self.train_fns_100)))

    def __getitem__(self, index):
        # fetch image
        if len(self.train_fns_140) != len(self.train_fns_100):
            raise RuntimeError("Unequal number between 100kv files and 140kv files!")

        fn_100 = self.train_fns_100[index]
        fn_140 = self.train_fns_140[index]
        im_100 = np.load(fn_100)
        im_140 = np.load(fn_140)
        im_100 = np.array(im_100, dtype=np.float32)
        im_140 = np.array(im_140, dtype=np.float32)
        im = np.stack((im_100, im_140), axis=2)
        im = np.reshape(im, (im_100.shape[0], im_100.shape[1], 2))
        transformer = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(args.img_size)])
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.train_fns_100)


def checkpoint(net, epoch, name, type):
    save_model_path = os.path.join(args.save_model_path, args.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = '{}-{:03d}-{}.pth'.format(name, epoch, type)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(),
               save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='./datasets/DECT_data/train',
                    help='root dir for train data')
parser.add_argument('--log_name', type=str,
                    default='train', help='networks function')
parser.add_argument('--save_model_path', type=str,
                    default='./SECT_DECT_results', help='dir for networks weight')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of networks')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
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
parser.add_argument('--pretrain', action='store_true', default=False, help='pre training')
parser.add_argument("--DATE_FORMAT", type=str, default='%A_%d_%B_%Y_%Hh_%Mm_%Ss', help='time format')
parser.add_argument("--parameter1", type=int, default=0.001, help='loss hyperparameter1')
parser.add_argument("--parameter2", type=int, default=0.005, help='loss hyperparameter2')

args = parser.parse_args()
# config = get_config()
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

TrainingDataset = DataLoader_val_train(args.data_dir)
TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=8,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

model = VMUNet_conv(input_channels=3, num_classes=2).to(device)

if args.n_gpu > 1:
    model = nn.DataParallel(model)
model.train()
optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

if args.pretrain:
    weights_path = './SECT_DECT_results/'
    model.load_state_dict(torch.load(weights_path))
    print('load pre train file')

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

print('init finish')

iter_num = 0
if args.resume:
    iter_num = resume_epoch * len(TrainingLoader)
max_epoch = args.max_epochs
max_iterations = args.max_epochs * len(TrainingLoader)
best_performance = 0.0
Loss1 = nn.MSELoss()
iterator = tqdm(range(max_epoch), ncols=70)
loss_list = []

for epoch_num in iterator:
    if args.resume:
        if epoch_num <= resume_epoch:
            continue
    loss_train_sum = 0
    loss_train_min = 1000

    for i_batch, ima in enumerate(TrainingLoader):
        st = time.time()
        image_batch = ima
        image_batch = image_batch.to(device)
        input_image = torch.sum(image_batch * 0.5, dim=1, keepdim=True)
        label = image_batch
        outputs = model(input_image)
        loss = Loss1(outputs, label)

        loss_train_sum += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        if (i_batch + 1) % 20 == 0:
            print(
                'epoch={:d}/{:d}, iteration={:d}/{:d}, learning_rate={:06f}, Loss={:.6f}, Time={:.4f}'
                .format(epoch_num, max_epoch - 1, i_batch + 1, len(TrainingLoader), lr_, loss.item(), time.time() - st))

    loss_arr = loss_train_sum.cpu().detach().numpy()
    loss_list.append(loss_arr)
    checkpoint(model, epoch_num, "model", "regular")

