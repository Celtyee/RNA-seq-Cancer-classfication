import torch
from methods.ResNet import ResNet50, ResNet101, ResNet152
from methods.VGG import vgg16, vgg19
from methods.googleNet import GoogLeNet
from methods.AlexNet import AlexNet
from methods.CNN import CNN
from torchvision import transforms

gpu_id = 2
device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
lr = 1e-4
optimize = 'Adam'
max_epoch = 200
val_interval = 2

img_channels = 3
num_classes = 5

val_ratio = 0.2
test_ratio = 0.1

model_dict = {
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "VGG16": vgg16,
    "VGG19": vgg19,
    "GoogleNet": GoogLeNet,
    "AlexNet": AlexNet,
    "CNN": CNN
}

resize = 224
transform = transforms.Compose([transforms.Resize([resize, resize], antialias=False)])
