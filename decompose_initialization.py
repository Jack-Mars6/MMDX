import os
import time
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import glob as glob
from tqdm import tqdm
from itertools import combinations
import torch.nn.functional as F
import torch

folder_path = "./datasets/DECT_data/train"
save_path = "./datasets/DECT_data/label"
file_path_140 = glob.glob(os.path.join(folder_path, './140kv/*.npy'))
file_path_140 = sorted(file_path_140)
file_path_100 = glob.glob(os.path.join(folder_path, './100kv/*.npy'))
file_path_100 = sorted(file_path_100)
print(len(file_path_140))
print(len(file_path_100))

os.makedirs(save_path, exist_ok=True)

# search coefficients from NIST
linear_cofficient = {'Adipose': (-0.11171025260029717, -0.0870228187919463),
                     'Air': (-1.0, -1.0),
                     'Muscle': (0.061890800299177255, 0.06287079910380881),
                     'Iodine': (0.4346875, 0.22676576576576576)}

temp_list = ['Adipose', 'Muscle', 'Air', 'Iodine']
combinations_list = []
for i in combinations(temp_list, 3):  # list all triplets
    combinations_list.append(i)

Adipose = np.array([[-0.11171025260029717], [-0.0870228187919463], [1]]).reshape(-1)
Air = np.array([[-1.0], [-1.0], [1]]).reshape(-1)
Muscle = np.array([[0.061890800299177255], [0.06287079910380881], [1]]).reshape(-1)
Iodine = np.array([[0.4346875], [0.22676576576576576], [1]]).reshape(-1)
prior_dict = {'Adipose': Adipose, 'Air': Air, 'Muscle': Muscle, 'Iodine': Iodine}
material_dict = {}


def gram_cal(vector1, vector2, vector3):
    gram_matrix = np.array([[np.dot(vector1, vector1), np.dot(vector1, vector2), np.dot(vector1, vector3)],
                            [np.dot(vector2, vector1), np.dot(vector2, vector2), np.dot(vector2, vector3)],
                            [np.dot(vector3, vector1), np.dot(vector3, vector2), np.dot(vector3, vector3)]])
    return np.linalg.det(gram_matrix)


for i in combinations_list:  # create a priority list
    relation = np.sqrt(gram_cal(prior_dict[i[0]], prior_dict[i[1]], prior_dict[i[2]])) / (
            np.linalg.norm(prior_dict[i[0]]) * np.linalg.norm(prior_dict[i[1]]) * np.linalg.norm(prior_dict[i[2]]))
    material_dict[relation] = i
prior_list = sorted(material_dict, reverse=True)
prior_combinations_list = []
for i in prior_list:
    prior_combinations_list.append(material_dict[i])

print(prior_combinations_list)


for q in range(0, 1000):
    st = time.time()
    ds_140 = np.load(file_path_140[q])
    ds_100 = np.load(file_path_100[q])
    linear_coefficient_140 = ds_140.reshape(-1)  # per-pixel process
    linear_coefficient_100 = ds_100.reshape(-1)
    Adipose_volume = np.zeros_like(linear_coefficient_140)
    Air_volume = np.zeros_like(linear_coefficient_140)
    Muscle_volume = np.zeros_like(linear_coefficient_140)
    Iodine_volume = np.zeros_like(linear_coefficient_140)
    for k in tqdm(range(linear_coefficient_140.size), ncols=70):
        Micro = np.array([[linear_coefficient_100[k]], [linear_coefficient_140[k]],
                          [1]])  # Micro：linear attenuation coefficients of mixtures
        count = 0
        violate_tri_dict = {}
        violate_tri_value = 10000
        for i in prior_combinations_list:  # Get the linear Attenuation coefficient of the triplets based on the priority list
            material_list = []
            for j in i:
                material_list.append(linear_cofficient[j])
            # M：linear attenuation coefficients of basic materials
            M = np.array([[material_list[0][0], material_list[1][0], material_list[2][0]],
                          [material_list[0][1], material_list[1][1], material_list[2][1]], [1, 1, 1]])
            alpha = np.dot(np.linalg.pinv(M), Micro)  # alpha：volume fraction
            if np.sum((alpha >= 0) & (alpha <= 1)) != 3:  # if satisfy the bounded constraint
                value = abs(np.sum(abs(alpha)) - 1)
                if value < violate_tri_value:
                    violate_tri_dict[value] = (i, alpha)
                    violate_tri_value = value
            else:
                count += 1
                eval(i[0] + '_volume')[k] = alpha[0]
                eval(i[1] + '_volume')[k] = alpha[1]
                eval(i[2] + '_volume')[k] = alpha[2]
                break
        if count == 0:
            (i, alpha) = violate_tri_dict[violate_tri_value]
            alpha = F.softmax(torch.from_numpy(alpha), dim=0).numpy()
            eval(i[0] + '_volume')[k] = alpha[0]
            eval(i[1] + '_volume')[k] = alpha[1]
            eval(i[2] + '_volume')[k] = alpha[2]

    Adipose_volume = Adipose_volume.reshape(ds_140.shape)
    Air_volume = Air_volume.reshape(ds_140.shape)
    Muscle_volume = Muscle_volume.reshape(ds_140.shape)
    Iodine_volume = Iodine_volume.reshape(ds_140.shape)

    label = np.stack((Adipose_volume, Muscle_volume, Iodine_volume, Air_volume), axis=0)
    save_path_each = os.path.join(
        save_path,
        "{:03d}.npy".format(q))

    np.save(save_path_each, label)
    print('{} / 1000, use{}s'.format(q, time.time() - st))
