import os
import numpy as np
import cv2
import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import torch
from networks.vmunet import VMUNet_conv
from networks.unet_model import *
from networks.unet_TransNeXt_mmdx import transnext_tiny


def calculate(pred, label):
    mse = np.mean((pred - label)**2)
    psnr = PSNR(image_true=label, image_test=pred, data_range=1.)
    ssim = SSIM(im1=label, im2=pred, data_range=1.)
    return psnr, ssim

def calculate_cal(pred, label):
    psnr = PSNR(image_true=label[(label!=0)], image_test=pred[(label!=0)], data_range=1.)
    ssim = SSIM(im1=label, im2=pred, data_range=1.)
    return psnr, ssim


def evaluate(model, path):
    # (200, 512, 512)
    arr_high = np.load(os.path.join(path, 'test_data_high.npy'))
    arr_low = np.load(os.path.join(path, 'test_data_low.npy'))
    fibroglandular = np.load(os.path.join(path, 'Fibroglandular.npy'))
    calcification = np.load(os.path.join(path, 'Calcification.npy'))
    adipose = np.load(os.path.join(path, 'Adipose.npy'))
    air = np.load(os.path.join(path, 'Air.npy'))
    # print(len(arr_high))
    psnr_fib_list = []
    psnr_cal_list = []
    psnr_adi_list = []
    psnr_air_list = []
    ssim_fib_list = []
    ssim_cal_list = []
    ssim_adi_list = []
    ssim_air_list = []

    for i in tqdm(range(200)):
        b_140 = arr_high[i]
        b_100 = arr_low[i]
        kv_low = np.array(b_100, dtype=np.float32)
        kv_low = 2 * (((kv_low - np.min(kv_low)) / (np.max(kv_low) - np.min(kv_low))) - 0.5)
        kv_high = np.array(b_140, dtype=np.float32)
        kv_high = 2 * (((kv_high - np.min(kv_high)) / (np.max(kv_high) - np.min(kv_high))) - 0.5)
        image_input = np.stack((kv_low, kv_high), axis=0)
        tensor_input = torch.from_numpy(image_input)
        tensor_input = torch.unsqueeze(tensor_input, dim=0).cuda()
        x_input = tensor_input.detach().cpu().numpy().reshape((2, 512, 512))

        model.eval()
        with torch.no_grad():
            output = model(tensor_input)
        output = torch.squeeze(output, dim=0)
        output = output.detach().cpu().numpy()

        fibroglandular_label = fibroglandular[i]
        calcification_label = calcification[i]
        adipose_label = adipose[i]
        air_label = air[i]

        adipose_output = output[0]
        calcification_output = output[1]
        fibroglandular_output = output[2]
        air_output = output[3]

        # 计算指标
        psnr_fib, ssim_fib = calculate(fibroglandular_output, fibroglandular_label)
        psnr_cal, ssim_cal = calculate_cal(calcification_output, calcification_label)
        psnr_adi, ssim_adi = calculate(adipose_output, adipose_label)
        psnr_air, ssim_air = calculate(air_output, air_label)
        psnr_fib_list.append(psnr_fib)
        psnr_cal_list.append(psnr_cal)
        psnr_adi_list.append(psnr_adi)
        psnr_air_list.append(psnr_air)
        ssim_fib_list.append(ssim_fib)
        ssim_cal_list.append(ssim_cal)
        ssim_adi_list.append(ssim_adi)
        ssim_air_list.append(ssim_air)

    return (np.mean(psnr_fib_list), np.mean(ssim_fib_list),
            np.mean(psnr_adi_list), np.mean(ssim_adi_list),
            np.mean(psnr_cal_list), np.mean(ssim_cal_list),
            np.mean(psnr_air_list), np.mean(ssim_air_list))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    decomposition_net = transnext_tiny(in_chans=1, num_classes=4, Ablation_mix=False, Ablation_att=False).cuda()
    decomposition_dir = './results_phantom/'
    decomposition_net.load_state_dict(torch.load(decomposition_dir))
    path = "./datasets/data/test"

    (psnr_fib, ssim_fib, psnr_adi, ssim_adi, psnr_cal, ssim_cal, psnr_air, ssim_air) = evaluate(decomposition_net, path)
    print('fibroglandular:', psnr_fib, ssim_fib)
    print('adipose:', psnr_adi, ssim_adi)
    print('calcification:', psnr_cal, ssim_cal)
    print('air:', psnr_air, ssim_air)