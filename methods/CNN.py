import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, img_channels: int = 3, num_classes: int = 1000, img_size=32):
        super(CNN, self).__init__()
        self.img_size = img_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # 输入的图片 （1，28，28）
                in_channels=img_channels,
                out_channels=16,  # 经过一个卷积层之后 （16,28,28）
                kernel_size=5,
                stride=1,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化层处理，维度为（16,14,14）
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(  # 输入（16,14,14）
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # 输出（32,14,14）
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 输出（32,7,7）
        )

        self.out = nn.Linear(32 * int(self.img_size / 4) * int(self.img_size / 4), num_classes)

    def forward(self, x):
        x = self.conv1(x)  # （batch_size,16,14,14）
        x = self.conv2(x)  # 输出（batch_size,32,7,7）
        x = x.view(x.size(0), -1)  # (batch_size,32*7*7)
        out = self.out(x)  # (batch_size,10)
        return out
