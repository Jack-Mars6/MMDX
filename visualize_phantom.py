import os
import numpy as np
import cv2
import torch
import glob
from networks.unet_model import *
from networks.unet_TransNeXt_mmdx import transnext_tiny
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def norm(img):
    # img = np.clip(img, 0, 255)
    img = (img - img.min()) / (img.max() - img.min())
    return img


def plot_material(image, my_cmap, file_dir, material_name):
    image = cv2.resize(image, (256, 256))

    fig1 = plt.figure(figsize=(2.56, 2.56), dpi=100)
    ax1 = fig1.add_axes([0, 0, 1, 1])
    ax1.axis('off')
    img = ax1.imshow(image, cmap=my_cmap, vmax=1, vmin=0)
    plt.savefig(f"{file_dir}{material_name}.png", dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig1)


arr_high = np.load('./datasets/data/test/test_data_high.npy')
arr_low = np.load('./datasets/data/test/test_data_low.npy')
i = 10
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

model = transnext_tiny(in_chans=1, num_classes=4, Ablation_mix=False, Ablation_att=False).cuda()
decomposition_dir = './results_phantom/'
model.load_state_dict(torch.load(decomposition_dir))

model.eval()
with torch.no_grad():
    output = model(tensor_input)
output = torch.squeeze(output, dim=0)
output = output.detach().cpu().numpy()

adipose = output[0]
calcification = output[1]
fibroglandular = output[2]
air = output[3]

save_dir = './visualize_color_phantom/'
os.makedirs(save_dir, exist_ok=True)
mycmap1 = colors.LinearSegmentedColormap.from_list('mycmap1', ['#000000', '#FFBB00'])
mycmap2 = colors.LinearSegmentedColormap.from_list('mycmap2', ['#000000', '#FFFFFF'])
mycmap3 = colors.LinearSegmentedColormap.from_list('mycmap3', ['#000000', '#00FFFF'])
mycmap4 = colors.LinearSegmentedColormap.from_list('mycmap4', ['#000000', '#33FF33'])

plot_material(calcification, mycmap2, save_dir, 'Calcification')
plot_material(adipose, mycmap1, save_dir, 'Adipose')
plot_material(fibroglandular, mycmap3, save_dir, 'Fibroglandular')
plot_material(air, mycmap4, save_dir, 'Air')
