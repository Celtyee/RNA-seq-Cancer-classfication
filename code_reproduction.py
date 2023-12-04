#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io as scio
import numpy as np
import random
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# In[2]:


data_dict = scio.loadmat("./dataset/cancer types.mat")


# In[3]:


df = np.array(data_dict['data'])
labels = df[:, -1]
features = df[:, :-1]
df.shape


# In[4]:


split = []
for _ in range(len(labels)):
    split.append(random.uniform(0,1))
split = np.array(split)

test_ratio = 0.2

train_labels = labels[split>=test_ratio]
train_features = features[split>=test_ratio]

test_labels = labels[split<test_ratio]
test_features = labels[split<test_ratio]


# In[5]:



num_classes = 5
normalize_up = 255
gene_expression_len = 24248
class RNAseqDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, trans=None):
        self.features = features
        self.labels = labels
        self.trans = trans
        
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = np.zeros(32*32)
        for i, cell_val in enumerate(self.features[idx]):
            pixel = round(cell_val*normalize_up/gene_expression_len)
            image[i] = pixel
        image = image.reshape(32,32)
        image = torch.tensor(image)
        if not self.trans is None:
            image = self.trans(image)
        label = torch.zeros(num_classes)
        label[int(self.labels[idx] -1)] = 1 
        return image, label


# In[31]:


# import torchvision
# from torchvision import models
# resnet50 = models.resnet50(pretrained=True)
# print(resnet50)


# In[12]:


class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)


def ResNet18(img_channels=3, num_classes=1000):
    return ResNet(18, Block, img_channels, num_classes)


def ResNet34(img_channels=3, num_classes=1000):
    return ResNet(34, Block, img_channels, num_classes)


def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(50, Block, img_channels, num_classes)


def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(101, Block, img_channels, num_classes)


def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(152, Block, img_channels, num_classes)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0, model_name='model.pth'):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_name = model_name
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("early stop is triggered")
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, self.model_name)
        torch.save(model.state_dict(), path)  # save the best model on validation dataset
        self.val_loss_min = val_loss

def train_one_epoch(train_loader, model, device, criterion):
    model.train()
    epoch_loss = 0
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss /= len(train_loader)
    return epoch_loss

def val_epoch(val_loader, model, device, criterion):
    model.eval()
    num_correct = 0
    epoch_loss = 0
    for inputs, labels in tqdm(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            num_eq = torch.eq(outputs.argmax(dim=1), labels.argmax(dim=1)).sum().item()
            num_correct += num_eq
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
    metric = num_correct/len(val_loader)
    epoch_loss /= len(val_loader)
    return metric, epoch_loss


# In[ ]:


gpu_id = 0
devices = device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
lr = 1e-3
optimize = 'Adam'
max_epoch = 2
val_interval = 2

model = ResNet50(img_channels=1, num_classes=num_classes)
train_ds = RNAseqDataset(train_features, train_labels)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

test_ds = RNAseqDataset(test_features, test_labels)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=True, num_workers=4)

criterion = torch.nn.CrossEntropyLoss()
optimizer = getattr(torch.optim, optimize)(model.parameters(), lr)

early_stopping = EarlyStopping("./models", model_name="ResNet50")
print(device)
for epoch in range(max_epoch):
    print(epoch)
    if early_stopping.early_stop:
        break
    epoch_loss = train_one_epoch(train_loader, model, device, criterion)
    early_stopping(epoch_loss, model)
    if (epoch + 1)%val_interval == 0:
        metric, epoch_loss = val_epoch(test_loader, model, device, criterion)
        print(f"The metric is {metric}. The validation loss is {epoch_loss}")



