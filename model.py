# from models.vdsr import VDSR
from models.base_model import *
from models.vgg16 import vgg16_pretrain
import torch.nn as nn


class teacher(nn.Module):
    def __init__(self, input_channel=3):
        super().__init__()
        self.detail=VGG_detail_tea()
        self.base=VGG_base()
        # for p in self.parameters():
        #     p.requires_grad=False
    def forward(self, x):
        # base层
        pan_base=self.base(x)
        # detail层
        distill=self.detail(x)
        output = self.detail(x)+pan_base

        return output, pan_base, distill

class student(nn.Module):
    def __init__(self, input_channel=3):
        super().__init__()
        # self.base = VGG_base_stu()
        self.detail=VGG_detail_stu()
        self.last = Conv5()
        self.vgg=vgg16_pretrain()
    def forward(self, lr_u, mul):
        # #detail层
        # distill=self.detail(lr_u)
        # l_output = self.detail(lr_u)+lr_u
        distill = self.detail(lr_u)
        # lr_u_base = self.base(lr_u)
        l_output = distill + lr_u
        l_output = self.last(l_output)
        #对比学习
        lr_u_block2,lr_u_block3=self.vgg(lr_u)
        mul_block2,mul_block3=self.vgg(mul)
        l_output_block2,l_output_block3=self.vgg(l_output)

        return distill,l_output,lr_u_block2,lr_u_block3,mul_block2,mul_block3,l_output_block2,l_output_block3

class All(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher=teacher()
        self.student=student()
    def forward(self, pan,lr_u, mul, opt):
        if not opt.test:
            if opt.mode == 'teacher':
                pan_output, pan_base, distill_tea = self.teacher(pan)
                return pan_output, pan_base, distill_tea
            elif opt.mode == 'student':
                for p in self.teacher.parameters():
                    p.requires_grad = False
                # with torch.no_grad():
                #     self.teacher.eval()
                pan_output, pan_base, distill_tea = self.teacher(pan)
                distill_stu, l_output, lr_u_block2, lr_u_block3, mul_block2, mul_block3, l_output_block2, l_output_block3=self.student(lr_u,mul)
                return pan_output, pan_base, distill_tea, distill_stu, l_output, lr_u_block2, lr_u_block3, mul_block2, mul_block3, l_output_block2, l_output_block3
        elif opt.test:
            if opt.mode == 'teacher':
                pan_output, pan_base, distill_tea = self.teacher(pan)
                return pan_output, pan_base, distill_tea
            elif opt.mode == 'student':
                distill_stu, l_output, lr_u_block2, lr_u_block3, mul_block2, mul_block3, l_output_block2, l_output_block3 = self.student(lr_u, mul)
                return distill_stu, l_output, lr_u_block2, lr_u_block3, mul_block2, mul_block3, l_output_block2, l_output_block3

if __name__ == '__main__':
    net_test = teacher()
    # 打印网络中所有类内变量的信息（按照先后顺序）
    print(net_test)
    print("--------------------")
    # 打印网络中所有类内变量参数值
    print(net_test.state_dict())
    # # 打印网络构成的参数字典中所有的网络键值,之后根据这个键值就可以去查看固定哪一层的参数，然后通过索引甚至可以看到具体这一层的第几个参数
    # print(net_test.state_dict().keys())
