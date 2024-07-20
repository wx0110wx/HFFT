# High-Frequency Feature Transfer for Multispectral Image Super-Resolution

=
Implementation of "High-Frequency Feature Transfer for Multispectral Image Super-Resolution" in PyTorch
## Preparation

=
### Datasets
* We conduct experiments on two remote sensing datasets: GaoFen-1 and GaoFen-2
* GaoFen-1: Contains 4,761 image pairs for training and 680 samples for testing.
* GaoFen-2: Contains 5,322 image pairs for training and 680 samples for testing.
You can download the preprocessed datasets from ，then extract them to  ' datasets/ '
##Training

=
###Train teacher models
Train teacher model using
```
python main.py -- Resume False --mode teacher --epoch 40
```
The trained model is saved in ' checpoints/teacher/'
###Train teacher model
After obtaining the teacher models, train the student model using
```
python main.py --mode student --epoch 40
```
