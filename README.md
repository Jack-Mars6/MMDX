# MMDX
This project provides implementation for "Revealing hidden material composition via physics-induced deep learning for precision clinical diagnosis".

## Package Versions
- python 3.10
- pytorch 2.6.0
- torchvision 0.21.0
- mamba-ssm 2.2.3
- opencv-python 4.10.0.84
- numpy 2.2.6

## Training
 Perform the training process for the patient and phantom dataset. 
```
python train_patient.py
python train_phantom.py
```

Train the DECT generation network. 
```
python train_DECT_generate.py
```

## Evaluation
Calculate the metrics. You can download the model weights [here](https://pan.baidu.com/s/1YfqP280EaOFGIugm5oBtkA?pwd=bw9s).
```
python calculate_phantom.py
```

## License
This project is covered under the BSD-3-Clause License.
