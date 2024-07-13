import torch.nn as nn
import torch

# class VGG_base(nn.Module):
#
#     # initialize model
#     def __init__(self, input_channel=3):
#         super().__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),  # default parameter：nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
#         )
#
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )
#
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             # nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         )
#
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#
#         self.conv7 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#
#         self.conv8 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             # nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         )
#
#         self.conv9 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(1),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingBilinear2d(scale_factor=2)
#         )
#
#         self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
#                           self.conv8, self.conv9]
#
#         print("VGG Model Initialize Successfully!")
#
#     # forward
#     def forward(self, x):
#         global output
#         for conv in self.conv_list:
#             output= conv(x)
#         return output
#
# class VGG_detail(nn.Module):
#
#     # initialize model
#     def __init__(self, input_channel=3):
#         super().__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),  # default parameter：nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
#         )
#
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )
#
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             # nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         )
#
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#
#         self.conv7 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#
#         self.conv8 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             # nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         )
#
#         self.conv9 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(1),
#             nn.ReLU(inplace=True),
#             # nn.UpsamplingBilinear2d(scale_factor=2)
#         )
#
#         self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
#                           self.conv8, self.conv9]
#
#         print("VGG Model Initialize Successfully!")
#
#     # forward
#     def forward(self, x):
#         global output
#         for conv in self.conv_list:
#             output = conv(x)
#         return output

class VGG_base(nn.Module):
    def __init__(self):
        super(VGG_base, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_1 = BaseConv(1, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv4_2 = BaseConv(256, 3, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(256, 1, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(128, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(64, 1, 3, 1, activation=nn.ReLU(), use_bn=True)

    # def forward(self, input):
    #     input = self.conv1_1(input)
    #     input = self.conv1_2(input)
    #     input = self.pool(input)
    #     input = self.conv2_1(input)
    #     input = self.conv2_2(input)
    #     input = self.conv3_1(input)
    #     input = self.conv3_2(input)
    #     input= self.conv3_3(input)
    #     input = self.conv4_1(input)
    #     input = self.conv4_2(input)
    #     input = self.upsample(input)
    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        input = self.conv2_2(input)
        input = self.conv2_2(input)
        input = self.upsample(input)
        input = self.conv5_1(input)
        input = self.conv1_2(input)
        input = self.conv5_2(input)


        return input

class VGG_base_stu(nn.Module):
    def __init__(self):
        super(VGG_base_stu, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_1 = BaseConv(1, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv4_2 = BaseConv(256, 3, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(256, 1, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(128, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(64, 1, 3, 1, activation=nn.ReLU(), use_bn=True)

    # def forward(self, input):
    #     input = self.conv1_1(input)
    #     input = self.conv1_2(input)
    #     input = self.pool(input)
    #     input = self.conv2_1(input)
    #     input = self.conv2_2(input)
    #     input = self.conv3_1(input)
    #     input = self.conv3_2(input)
    #     input= self.conv3_3(input)
    #     input = self.conv4_1(input)
    #     input = self.conv4_2(input)
    #     input = self.upsample(input)
    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        input = self.conv2_2(input)
        input = self.conv2_2(input)
        input = self.upsample(input)
        input = self.conv5_1(input)
        input = self.conv1_2(input)
        input = self.conv5_2(input)


        return input

class Conv5(nn.Module):
    def __init__(self):
        super(Conv5, self).__init__()
        self.conv1 = BaseConv(1, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3 = BaseConv(64, 1, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv2(input)
        input = self.conv2(input)
        input = self.conv3(input)
        return input

class VGG_detail_tea(nn.Module):
    def __init__(self):
        super(VGG_detail_tea, self).__init__()
        # self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_1 = BaseConv(1, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv4_2 = BaseConv(256, 3, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(256, 1, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(128, 1, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.conv2_1(input)
        input = self.conv2_2(input)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        input = self.conv3_3(input)
        input = self.conv4_1(input)
        input = self.conv4_2(input)

        return input

class VGG_detail_stu(nn.Module):
    def __init__(self):
        super(VGG_detail_stu, self).__init__()
        # self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_1 = BaseConv(1, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv4_2 = BaseConv(256, 3, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(256, 1, 3, 1, activation=None, use_bn=False)
        self.conv5_1 = BaseConv(128, 1, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.conv2_1(input)
        input = self.conv2_2(input)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        input= self.conv3_3(input)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        return input

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        # if self.use_in:
        #     input = self.IN(input)

        if self.activation:
            input = self.activation(input)

        return input

if __name__ == "__main__":
    inputs = torch.rand((8, 3, 224, 224)).cuda()
    model = VGG_base().cuda().train()
    outputs = model(inputs)
    print(outputs.shape)