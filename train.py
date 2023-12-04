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

from config import *
from utils import *
from evaluate import model_eval


def train_epoch(train_loader, model, criterion, epoch_id, optimizer):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
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


def val_epoch(val_loader, model, criterion):
    model.eval()
    num_correct = 0
    num_all = 0
    epoch_loss = 0
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            num_eq = torch.eq(outputs.argmax(dim=1), labels.argmax(dim=1))
            num_correct += num_eq.sum().item()
            num_all += len(num_eq)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
    metric = num_correct / num_all
    epoch_loss /= len(val_loader)
    return metric, epoch_loss


def main():
    model_name = "GoogleNet"
    model = None
    loader_dict = None
    if "ResNet" in model_name:
        loader_dict = data_generation()
        model = model_dict[model_name](img_channels=img_channels, num_classes=num_classes).to(device)
    elif "VGG" in model_name:
        loader_dict = data_generation()
        model = model_dict[model_name](img_channels=img_channels, num_classes=num_classes).to(device)
    elif "Google" in model_name:
        loader_dict = data_generation()
        model = model_dict[model_name](img_channels=img_channels, num_classes=num_classes).to(device)
    elif "Alex" in model_name:
        loader_dict = data_generation(img_size=224)
        model = model_dict[model_name](img_channels=img_channels, num_classes=num_classes).to(device)
    elif "CNN" in model_name:
        loader_dict = data_generation()
        model = model_dict[model_name](img_channels=img_channels, num_classes=num_classes, img_size=resize).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, optimize)(model.parameters(), lr)

    early_stopping = EarlyStopping("./models", model_name=f"{model_name}.pth", patience=50)
    for epoch in tqdm(range(max_epoch)):
        if early_stopping.early_stop:
            break
        epoch_loss = train_epoch(loader_dict["train"], model, criterion, epoch_id=epoch,
                                 optimizer=optimizer)

        if (epoch + 1) % val_interval == 0:
            _, epoch_loss = val_epoch(loader_dict["val"], model, criterion)
            early_stopping(epoch_loss, model)

    eval_dict = model_eval(loader_dict["test"], model=model, model_path=early_stopping.path)
    for key in eval_dict.keys():
        print(f"The {key} is {eval_dict[key]:.4f}.")


if __name__ == "__main__":
    main()
