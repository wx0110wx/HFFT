# High-Frequency Feature Transfer for Multispectral Image Super-Resolution


Implementation of "High-Frequency Feature Transfer for Multispectral Image Super-Resolution" in PyTorch
## Preparation

### Datasets
* We conduct experiments on two remote sensing datasets: GaoFen-1 and GaoFen-2
* GaoFen-1: Contains 4,761 image pairs for training and 680 samples for testing.
* GaoFen-2: Contains 5,322 image pairs for training and 680 samples for testing.
You can download the preprocessed datasets from ，then extract them to  ' datasets/ '
## Training

### Train teacher model
Train teacher model using
```
python main.py -- Resume False --mode teacher --epochs epochs_number
```
The trained model is saved in ' checpoints/teacher/'
### Train student model
After obtaining the teacher models, please place the address of the saved teacher model in
```
if opt.Resume:
            path_checkpoint = ''
```
train the student model using
```
python main.py --mode student --epochs epochs_number
```
