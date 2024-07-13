from __future__ import print_function
import argparse
from math import log10, sqrt
import time
import os
from os.path import join
import cv2
from PIL import Image
from torchvision.transforms import Compose,ToTensor, Resize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets.data import get_training_set
from models.model import All
import datetime


parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--batch_size', type=int, default=1,
                    help="training batch size")
parser.add_argument('--test_batch_size', type=int,
                    default=1, help="testing batch size")
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning Rate. Default=0.001')
parser.add_argument("--clip", type=float, default=0.4,
                    help="Clipping Gradients. Default=0.4")
parser.add_argument("--weight-decay", "--wd", default=1e-4,
                    type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--cuda', action='store_true',default='True', help='use cuda?')
parser.add_argument('--threads', type=int, default=1,
                    help='number of threads for data loader to use')
parser.add_argument('--gpuids', default=[0], nargs='+',
                    help='GPU ID for using')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to test or resume model')

parser.add_argument("--step", type=int, default=10,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument('--test', action='store_true',default='1', help='test mode') #test
# parser.add_argument('--test', action='store_true',help='test mode') #train
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--mode', default='student', type=str, help='teacher or student')# 别忘了固定教师模型参数
parser.add_argument('--Resume', default=True, type=bool, help='loading pretrained model?')

def main():
    global opt
    opt = parser.parse_args()
    opt.gpuids = list(map(int, opt.gpuids))
    print(opt)
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    cudnn.benchmark = True
    # if opt.mode == 'teacher':
    #     if not opt.test:
    #         del_files('/home/lxz/下载/20220514/train_results/brightness/teacher/epoch1/input_pan')
    #         del_files('/home/lxz/下载/20220514/train_results/brightness/teacher/epoch1/input_lr_u')
    #         del_files('/home/lxz/下载/20220514/train_results/brightness/teacher/epoch1/pan_base')
    #         del_files('/home/lxz/下载/20220514/train_results/brightness/teacher/epoch1/distill_tea')
    #         del_files('/home/lxz/下载/20220514/train_results/brightness/teacher/epoch1/pan_output')
    #     else:
    #         del_files('./results/brightness/teacher/epoch10/pan')
    #         del_files('./results/brightness/teacher/epoch10/pan_base')
    #         del_files('./results/brightness/teacher/epoch10/detail')
    #         # del_files('./results/brightness/teacher/epoch10/detail_stu')
    #         del_files('./results/brightness/teacher/epoch10/pan_output')
    # if opt.mode == 'student':
    #     if not opt.test:
    #         del_files('/home/lxz/下载/20220514/train_results/brightness/student/epoch1/input_lr_u')
    #         del_files('/home/lxz/下载/20220514/train_results/brightness/student/epoch1/input_mul')
    #         del_files('/home/lxz/下载/20220514/train_results/brightness/student/epoch1/distill_stu')
    #         del_files('/home/lxz/下载/20220514/train_results/brightness/student/epoch1/l_output')
    #         del_files('/home/lxz/下载/20220514/train_results/brightness/student/epoch1/distill_tea')
    #         del_files('/home/lxz/下载/20220514/train_results/brightness/student/epoch1/input_pan')
    #         del_files('/home/lxz/下载/20220514/train_results/brightness/student/epoch1/pan_base')
    #     else:
    #         del_files('./results/brightness/student/epoch44/pan')
    #         del_files('./results/brightness/student/epoch44/pan_base')
    #         del_files('./results/brightness/student/epoch44/detail')
    #         del_files('./results/brightness/student/epoch44/detail_stu')
    #         del_files('./results/brightness/student/epoch44/l_output')

    model = All()
    criterion = nn.L1Loss()
    if opt.cuda:
        torch.cuda.set_device(opt.gpuids[0])
        with torch.cuda.device(opt.gpuids[0]):
            model=model.cuda()
            criterion = criterion.cuda()
        # model = nn.DataParallel(model, device_ids=opt.gpuids,output_device=opt.gpuids[0])

    '---------------------------------------------------------------------------------------------------------------------------------'
    #train
    if not opt.test:
        print("train process")
        train_set = get_training_set()
        training_data_loader = DataLoader(
            dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True, drop_last=True)

        if opt.Resume:
            path_checkpoint = '/home/lxz/下载/20220514（另一个复件）/checkpoints/student/2022-06-28 11:57:46.920120/param_epoch_30_loss_0.8037.pth'
            model.load_state_dict(torch.load(path_checkpoint), strict=False)
            # model = torch.load(path_checkpoint)

            print("resume from checkpoint"+':'+path_checkpoint)
            # path_checkpoint = '/home/lxz/下载/20220514/checkpoints/student/灰度图/no_compareLearning_and_zhengliu/epoch_10.pth'
            # model = torch.load(path_checkpoint)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr,
                               weight_decay=opt.weight_decay)

        train_time = 0.0
        now = datetime.datetime.now()
        for epoch in range(1, opt.epochs + 1):
            start_time = time.time()
            loss = train(model, criterion, epoch, optimizer, training_data_loader, now)
            elapsed_time = time.time() - start_time
            train_time += elapsed_time
            print("===> {:.2f} seconds to train this epoch".format(
                elapsed_time))
            if epoch % 2 == 0:
                checkpoint(model, epoch, loss, now)

        print("===> average training time per epoch: {:.2f} seconds".format(train_time / opt.epochs))
        print("===> training time: {:.2f} seconds".format(train_time))

    '---------------------------------------------------------------------------------------------------------------------------------'
    #test
    if opt.test:
        print("test process")
        #加载模型参数
        model_= torch.load('/home/lxz/下载/20220514（另一个复件）/checkpoints/student/2022-06-28 11:57:46.920120/epoch_30_loss_0.8037.pth')
        #加载测试数据集
        filename_pan = '/home/lxz/下载/20220514/datasets/pan/brightness/test/'
        filename_mul = '/home/lxz/下载/20220514/datasets/mul/brightness/test/'
        filename_lr_u = '/home/lxz/下载/20220514/datasets/lr_u/brightness/test/down_up_1/'
        filename_output = '/home/lxz/下载/20220514（另一个复件）/results/brightness/student/no_c_epoch30/l_output/'
        if not os.path.exists(filename_output):
            os.makedirs(filename_output)

        start_time = time.time()
        with torch.no_grad():
            model_.eval()
            test_img_nums=2267 #2267
            test(model_,filename_pan,filename_mul,filename_lr_u,filename_output,test_img_nums)
            elapsed_time = time.time() - start_time
            print("===> average {:.2f} image/sec for test".format(
                100.0/elapsed_time))
    '---------------------------------------------------------------------------------------------------------------------------------'

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr


def train(model, criterion, epoch, optimizer, training_data_loader, now):
    lr = adjust_learning_rate(epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input_pan, input_mul, input_lr_u =Variable(batch[0]), Variable(batch[1]),Variable(batch[2])
        if opt.cuda:
            input_pan = input_pan.cuda()
            input_mul = input_mul.cuda()
            input_lr_u = input_lr_u.cuda()

        optimizer.zero_grad()
        if opt.mode == 'teacher':
            pan_output, pan_base, distill_tea= model(input_pan,input_lr_u,input_mul, opt)
        elif opt.mode == 'student':
            pan_output, pan_base, distill_tea, distill_stu, l_output, lr_u_block2, lr_u_block3, mul_block2, mul_block3, l_output_block2, l_output_block3 = model(input_pan,input_lr_u,input_mul, opt)

        if opt.mode == 'teacher':
            filename7 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/teacher/{}/input_pan/'.format(now)
            # filename8 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/teacher/{}/input_mul/'.format(now)
            filename9 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/teacher/{}/input_lr_u/'.format(now)
            filename10 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/teacher/{}/pan_base/'.format(now)
            filename11 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/teacher/{}/distill_tea/'.format(now)
            # filename12 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/teacher/{}/distill_stu/'.format(now)
            filename13 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/teacher/{}/pan_output/'.format(now)
            # filename14 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/teacher/{}/l_output/'.format(now)
            if not os.path.exists(filename7):
                os.makedirs(filename7)
            if not os.path.exists(filename9):
                os.makedirs(filename9)
            if not os.path.exists(filename10):
                os.makedirs(filename10)
            if not os.path.exists(filename11):
                os.makedirs(filename11)
            if not os.path.exists(filename13):
                os.makedirs(filename13)
            save_image_tensor2pillow(input_pan, filename7 + str(iteration % 30) + '.tif')
            # save_image_tensor2pillow(input_mul, filename8 + str(iteration % 30) + '.tif')
            save_image_tensor2pillow(input_lr_u, filename9 + str(iteration % 30) + '.tif')
            save_image_tensor2pillow(pan_base, filename10 + str(iteration % 30) + '.tif')
            save_image_tensor2pillow(distill_tea, filename11 + str(iteration % 30) + '.tif')
            # save_image_tensor2pillow(distill_stu, filename12 + str(iteration % 30) + '.tif')
            save_image_tensor2pillow(pan_output, filename13 + str(iteration % 30) + '.tif')
            # save_image_tensor2pillow(l_output, filename14 + str(iteration % 30) + '.tif')
        if opt.mode == 'student':
            filename6='/home/lxz/下载/20220514（另一个复件）/train_results/brightness/student/{}/l_output/'.format(now)
            filename7 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/student/{}/lr_u_base/'.format(now)
            filename8 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/student/{}/input_mul/'.format(now)
            filename9 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/student/{}/input_lr_u/'.format(now)
            filename10 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/student/{}/pan_base/'.format(now)
            filename11 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/student/{}/distill_tea/'.format(now)
            filename12 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/student/{}/distill_stu/'.format(now)
            filename13 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/student/{}/pan_output/'.format(now)
            filename14 = '/home/lxz/下载/20220514（另一个复件）/train_results/brightness/student/{}/input_pan/'.format(now)
            if not os.path.exists(filename6):
                os.makedirs(filename6)
            if not os.path.exists(filename7):
                os.makedirs(filename7)
            if not os.path.exists(filename8):
                os.makedirs(filename8)
            if not os.path.exists(filename9):
                os.makedirs(filename9)
            if not os.path.exists(filename10):
                os.makedirs(filename10)
            if not os.path.exists(filename11):
                os.makedirs(filename11)
            if not os.path.exists(filename12):
                os.makedirs(filename12)
            if not os.path.exists(filename13):
                os.makedirs(filename13)
            if not os.path.exists(filename14):
                os.makedirs(filename14)
            save_image_tensor2pillow(l_output, filename6 + str(iteration%30) + '.tif')
            # save_image_tensor2pillow(lr_u_base, filename7 + str(iteration%30) + '.tif')
            save_image_tensor2pillow(input_mul, filename8 + str(iteration%30) + '.tif')
            save_image_tensor2pillow(input_lr_u, filename9 + str(iteration%30) + '.tif')
            save_image_tensor2pillow(pan_base, filename10 + str(iteration % 30) + '.tif')
            save_image_tensor2pillow(distill_tea, filename11 + str(iteration % 30) + '.tif')
            save_image_tensor2pillow(distill_stu, filename12 + str(iteration % 30) + '.tif')
            save_image_tensor2pillow(pan_output, filename13 + str(iteration % 30) + '.tif')
            save_image_tensor2pillow(input_pan, filename14 + str(iteration % 30) + '.tif')

        if opt.mode == 'teacher':
            # 用于教师模型训练的损失函数
            loss=criterion(pan_output, input_pan)+0.1*criterion(pan_base, input_lr_u)
            # loss=criterion(pan, input_pan)+criterion(pan_base,input_mul)
        else:
            # 用于学生模型训练的损失函数

            # loss = 0.5*criterion(l_output, input_mul) + criterion(distill_stu,distill_tea) + 0.05*criterion(lr_u_base, pan_base)
            loss = 0.5 * criterion(l_output, input_mul) + 5*criterion(distill_stu, distill_tea)\
                   +0.5*(criterion(l_output_block2,mul_block2)/(criterion(l_output_block2,lr_u_block2)+0.0000000001))\
                   +0.5*(criterion( l_output_block3,mul_block3) / (criterion( l_output_block3,lr_u_block3)+0.0000000001))
        epoch_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), opt.clip/lr)
        optimizer.step()
        if iteration % 100 ==0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(
        epoch, epoch_loss / len(training_data_loader)))
    ave_loss = epoch_loss / len(training_data_loader)
    return ave_loss

def test(model, filename_pan, filename_mul, filename_lr_u,filename_output,test_img_nums):
    img_transform = Compose([
        Resize((252, 252)),
        ToTensor(),
    ])
    now = datetime.datetime.now()
    for i in range(test_img_nums):#[:1]
        # input_pan = Image.open(filename_pan + str(i+1) + '.tif')
        # input_mul = Image.open(filename_mul + str(i+1) + '.tif')
        # input_lr_u = Image.open(filename_lr_u + str(i+1) + '.tif')
        input_pan = Image.open(filename_pan + str(i + 1) + '.tif')
        input_mul = Image.open(filename_mul + str(i + 1) + '.tif')
        input_lr_u = Image.open(filename_lr_u + str(i + 1) + '.tif')
        #img to tensor
        input_pan=img_transform(input_pan)
        input_mul=img_transform(input_mul)
        input_lr_u=img_transform(input_lr_u)
        #add 1 channel
        input_pan = Variable(input_pan[None, :, :, :])
        input_mul = Variable(input_mul[None, :, :, :])
        input_lr_u = Variable(input_lr_u[None, :, :, :])

        if opt.cuda:
            input_pan = input_pan.cuda()
            input_mul = input_mul.cuda()
            input_lr_u = input_lr_u.cuda()
            if opt.mode =='teacher':
                pan_output, pan_base, distill_tea = model(input_pan, input_lr_u, input_mul, opt)
            elif opt.mode == 'student':
                distill_stu, l_output, lr_u_block2, lr_u_block3, mul_block2, mul_block3, l_output_block2, l_output_block3 = model(
                    input_pan, input_lr_u, input_mul, opt)
            else:
                print('Test error. Please respect mode spell.')

            #一些中间输出结果

            if opt.mode == 'teacher':
                filename0 = './results/brightness/teacher/{}/input_lr_u/'.format(now)
                filename1='./results/brightness/teacher/{}/pan/'.format(now)
                filename2='./results/brightness/teacher/{}/pan_base/'.format(now)
                filename3='./results/brightness/teacher/{}/detail_tea/'.format(now)
                filename4='./results/brightness/teacher/{}/pan_output/'.format(now)
                if not os.path.exists(filename0):
                    os.makedirs(filename0)
                if not os.path.exists(filename1):
                    os.makedirs(filename1)
                if not os.path.exists(filename2):
                    os.makedirs(filename2)
                if not os.path.exists(filename3):
                    os.makedirs(filename3)
                if not os.path.exists(filename4):
                    os.makedirs(filename4)
                save_image_tensor2pillow(input_lr_u, filename0 + str(i+1) + '.tif')
                save_image_tensor2pillow(input_pan, filename1 + str(i+1) + '.tif')
                save_image_tensor2pillow(pan_base, filename2 + str(i+1) + '.tif')
                save_image_tensor2pillow(distill_tea, filename3 + str(i+1) + '.tif')
                save_image_tensor2pillow(pan_output, filename4 + str(i + 1) + '.tif')
                print("第"+str(i+1)+"张图像")
            if opt.mode == 'student':
                filename1 = './results/brightness/student/{}/lr_u_base/'.format(now)
                # filename2 = './results/brightness/student/{}/pan_base/'.format(now)
                filename3 = './results/brightness/student/{}/detail_tea/'.format(now)
                filename4='./results/brightness/student/{}/detail_stu/'.format(now)
                filename5='./results/brightness/student/{}/l_output/'.format(now)
                filename0 = './results/brightness/student/{}/input_lr_u/'.format(now)
                filename6='./results/brightness/student/{}/input_mul/'.format(now)
                if not os.path.exists(filename0):
                    os.makedirs(filename0)
                if not os.path.exists(filename1):
                    os.makedirs(filename1)
                # if not os.path.exists(filename2):
                #     os.makedirs(filename2)
                if not os.path.exists(filename3):
                    os.makedirs(filename3)
                if not os.path.exists(filename4):
                    os.makedirs(filename4)
                if not os.path.exists(filename5):
                    os.makedirs(filename5)
                if not os.path.exists(filename6):
                    os.makedirs(filename6)
                save_image_tensor2pillow(l_output, filename5 + str(i + 1) + '.tif')
                # save_image_tensor2pillow(lr_u_base, filename1 + str(i + 1) + '.tif')
                # save_image_tensor2pillow(pan_base, filename2 + str(i + 1) + '.tif')
                # save_image_tensor2pillow(distill_tea, filename3 + str(i + 1) + '.tif')
                save_image_tensor2pillow(distill_stu, filename4 + str(i+1) + '.tif')
                # save_image_tensor2pillow(pan_output, filename5 + str(i + 1) + '.tif')
                save_image_tensor2pillow(input_lr_u, filename0 + str(i + 1) + '.tif')
                save_image_tensor2pillow(input_mul, filename6 + str(i + 1) + '.tif')
                print("第" + str(i + 1) + "张图像")



def checkpoint(model, epoch, loss, now):
    if opt.mode == 'teacher':
        if not(os.path.isdir('checkpoints/teacher/{}'.format(now))):
            os.makedirs(os.path.join('checkpoints/teacher/{}'.format(now)))
        # if not(os.path.isdir('checkpoints/student/灰度图/no_compareLearning_and_zhengliu')):
        #     os.makedirs(os.path.join('checkpoints/student/灰度图/no_compareLearning_and_zhengliu'))

        model_out_path = "checkpoints/teacher/{}/epoch_{}_loss_{:.4f}.pth".format(now, epoch, loss)
        model_out_path_param = "checkpoints/teacher/灰度图/param_epoch_{}_loss_{:.4f}.pth".format(now, epoch, loss)
        torch.save(model, model_out_path)
        torch.save(model.state_dict(), model_out_path_param)
        print("Checkpoint saved to {}".format(model_out_path))
    if opt.mode == 'student':
        if not(os.path.isdir('checkpoints/student/{}'.format(now))):
            os.makedirs(os.path.join('checkpoints/student/{}'.format(now)))
        model_out_path = "checkpoints/student/{}/epoch_{}_loss_{:.4f}.pth".format(now, epoch,loss)
        model_out_path_param = "checkpoints/student/{}/param_epoch_{}_loss_{:.4f}.pth".format(now, epoch, loss)
        torch.save(model, model_out_path)
        torch.save(model.state_dict(), model_out_path_param)
        print("Checkpoint saved to {}".format(model_out_path))


def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    # input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor)
    im.save(filename)

if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print("===> total time: {:.2f} seconds".format(elapsed_time))
