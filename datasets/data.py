from os.path import join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from datasets.data_utils import DatasetFromFolder


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def input_transform():
    return Compose([
        # CenterCrop(crop_size),
        # Resize((crop_size//upscale_factor, crop_size//upscale_factor)),
        Resize((256, 256)),
        # Resize((252, 252)),
        ToTensor(),
    ])

def target_transform():
    return Compose([
        # CenterCrop(crop_size),
        ToTensor(),
    ])

def get_training_set():
    # pan_dir = join("./datasets", "pan/brightness/train")
    # lr_u_dir = join("./datasets", "lr_u/brightness/train")
    # mul_dir = "/home/lxz/下载/20220514/datasets/mul/brightness/train"
    pan_dir = "/home/lxz/下载/20220514（另一个复件）/datasets/pan/brightness/test_1587*3"
    lr_u_dir = "/home/lxz/下载/20220514（另一个复件）/datasets/lr_u/brightness/train"
    mul_dir = "/home/lxz/下载/20220514（另一个复件）/datasets/mul/brightness/train"

    return DatasetFromFolder(pan_dir,mul_dir,lr_u_dir,
                             input_transform=input_transform())

# def get_validation_set():
#     pan_dir = join("./datasets", "pan/brightness/test")
#     mul_dir = join("./datasets", "mul/brightness/test")
#     lr_u_dir = join("./datasets", "lr_u/brightness/test/down_up_1")
#     return DatasetFromFolder(pan_dir,mul_dir,lr_u_dir,
#                              input_transform=input_transform())

# def get_test_set():
#     pan_dir = join("./datasets", "pan/brightness/test")
#     mul_dir = join("./datasets", "mul/brightness/test")
#     lr_u_dir = join("./datasets", "lr_u/brightness/test/down_up_1")
#     return DatasetFromFolder(pan_dir,mul_dir,lr_u_dir,
#                              input_transform=input_transform())
