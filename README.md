# High-Frequency Feature Transfer for Multispectral Image Super-Resolution


Implementation of "High-Frequency Feature Transfer for Multispectral Image Super-Resolution" in PyTorch
## Preparation

### Datasets
* We conduct experiments on two remote sensing datasets: GaoFen-1 and GaoFen-2
* GaoFen-1: Contains 4,761 image pairs for training and 680 samples for testing.
* GaoFen-2: Contains 5,322 image pairs for training and 680 samples for testing.
You can download the preprocessed datasets from [here](http://www.baidu.com/)，then extract them to  ' datasets/ '
## Training

### Train teacher model
Train teacher model using
```
python main.py -- Resume False --mode teacher --epochs epochs_number
```
The trained model is saved in ' checpoints/teacher/'
### Train student model
After obtaining the teacher models, please place the address of the saved teacher model on line 103 of main.py
```
if opt.Resume:
            path_checkpoint = './checkpoints/teacher/trained teacher model'
```
Then, train the student model using
```
python main.py --mode student --epochs epochs_number
```
The trained model is saved in ' checpoints/student/'
## Testing
Put the address of the test model on line 134 of main.py
```
if opt.test:
        print("test process")
        #加载模型参数
        model_= torch.load('./checkpoints/student or teacher model/')
```
Evaluate the performance of the training model using
```
python main.py --mode student or teacher --test
```
We also provide our pretrained models on all four datasets for reference. You can download them from [here](http://www.baidu.com/).


