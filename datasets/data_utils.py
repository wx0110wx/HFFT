import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageFilter
import numpy as np
import torchvision as vision


toPIL = vision.transforms.ToPILImage()


def noisy(img, std=3.0):
    mean = 0.0
    gauss = np.random.normal(mean, std, (img.height, img.width, 3))
    # noisy = np.clip(np.uint8(img + gauss), 0, 255)
    noisy = np.uint8(img + gauss)
    return noisy


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])


def load_img(filepath):
    # img = Image.open(filepath).convert('YCbCr')
    img = Image.open(filepath).convert('L')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, pan_dir,mul_dir,lr_u_dir, input_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.pan_image_filenames = [join(pan_dir, x)
                                for x in listdir(pan_dir) if is_image_file(x)]
        self.mul_image_filenames = [join(mul_dir, x)
                                    for x in listdir(mul_dir) if is_image_file(x)]
        self.lr_u_image_filenames = [join(lr_u_dir, x)
                                    for x in listdir(lr_u_dir) if is_image_file(x)]

        self.input_transform = input_transform
        # self.target_transform = target_transform
        # self.add_noise = add_noise
        # self.noise_std = noise_std

    def __getitem__(self, index):
        input_pan = load_img(self.pan_image_filenames[index])
        input_mul = load_img(self.mul_image_filenames[index])
        input_lr_u = load_img(self.lr_u_image_filenames[index])
        # target = input.copy()
        if self.input_transform:
            # if self.add_noise:
            #     input = noisy(input, self.noise_std)
            #     input = toPIL(input)
            input_pan = self.input_transform(input_pan)
            input_mul = self.input_transform(input_mul)
            input_lr_u = self.input_transform(input_lr_u)

        return input_pan,input_mul,input_lr_u

    def __len__(self):
        return len(self.mul_image_filenames)

