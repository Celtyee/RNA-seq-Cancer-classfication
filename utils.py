import torch
from config import *
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import scipy.io as scio
from torchvision import transforms


class RNAseqDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, img_size: int = 32, trans=None):
        self.features = features
        self.labels = labels
        self.trans = trans
        self.normalize_up = 255
        self.gene_expression_len = 24248
        self.img_size = img_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = np.zeros(self.img_size * self.img_size)
        for i, cell_val in enumerate(self.features[idx]):
            pixel = round(cell_val * self.normalize_up / self.gene_expression_len)
            image[i] = pixel
        image = image.reshape(self.img_size, self.img_size)
        image = torch.Tensor(image)
        image = image.unsqueeze(0)
        # turn into 3 channels
        image = torch.cat([image, image, image], dim=0)
        if self.trans:
            image = self.trans(image)

        label = torch.zeros(num_classes)
        label[int(self.labels[idx] - 1)] = 1
        return image, label


def data_generation(img_size: int = 32):
    data_dict = scio.loadmat("./dataset/cancer types.mat")
    loader_dict = {}
    # In[3]:
    df = np.array(data_dict['data'])
    # labels = df[:, -1]
    # features = df[:, :-1]

    # In[4]:
    split = []
    for _ in range(len(df)):
        split.append(random.uniform(0, 1))
    split = np.array(split)
    if os.path.exists("./dataset/train.npy") and os.path.exists("./dataset/val.npy") and os.path.exists("./dataset/test.npy"):
        train = np.load("./dataset/train.npy")
        val = np.load("./dataset/val.npy")
        test = np.load("./dataset/test.npy")
    else:
        print("create new ")
        val_select = np.logical_and(split >= test_ratio, split < (test_ratio + val_ratio))

        train = df[split >= (test_ratio + val_ratio)]
        np.save("dataset/train.npy", train)
        val = df[val_select]
        np.save("dataset/val.npy", val)
        test = df[split < test_ratio]
        np.save("dataset/test.npy", test)

    train_labels = train[:, -1]
    train_features = train[:, :-1]

    val_labels = val[:, -1]
    val_features = val[:, :-1]

    test_labels = test[:, -1]
    test_features = test[:, :-1]

    train_ds = RNAseqDataset(train_features, train_labels, img_size=img_size, trans=transform)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, drop_last=True)

    val_ds = RNAseqDataset(val_features, val_labels, img_size=img_size, trans=transform)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=4, drop_last=True)

    test_ds = RNAseqDataset(test_features, test_labels, img_size=img_size, trans=transform)
    test_loader = DataLoader(test_ds, batch_size=4, num_workers=4, drop_last=True)

    loader_dict['train'] = train_loader
    loader_dict['val'] = val_loader
    loader_dict['test'] = test_loader

    loader_dict['test_ds'] = test_ds

    return loader_dict


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
        self.path = os.path.join(self.save_path, self.model_name)

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
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        torch.save(model.state_dict(), self.path)  # save the best model on validation dataset
        self.val_loss_min = val_loss
