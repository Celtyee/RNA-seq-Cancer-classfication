from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import numpy as np

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

from utils import RNAseqDataset
from config import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random


def model_eval(test_loader, model, model_path) -> dict:
    y_pred = torch.Tensor([]).to(device)
    y_true = torch.Tensor([]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    eval_dict = {}
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            predicts = outputs.argmax(1)
            trues = labels.argmax(1)
            y_pred = torch.cat((y_pred, predicts), dim=0)
            y_true = torch.cat((y_true, trues), dim=0)

    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    eval_dict["Accuracy"] = torch.eq(y_pred, y_true).sum().item() / len(y_true)
    eval_dict["F1"] = f1_score(y_true.numpy(), y_pred.numpy(), average="macro")
    eval_dict["Recall"] = recall_score(y_true.numpy(), y_pred.numpy(), average="macro")
    eval_dict["Precision"] = precision_score(y_true.numpy(), y_pred.numpy(), average="macro")

    return eval_dict


def grad_cam(model, inputs, model_name, save_imgs=False):
    input_tensor = inputs.to(device)
    input_tensor = input_tensor.unsqueeze(0)

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = None
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    fp_rgb_fig = inputs.numpy() / 255
    fp_rgb_fig = fp_rgb_fig.transpose((1, 2, 0))
    visualization_origin = show_cam_on_image(fp_rgb_fig, grayscale_cam, use_rgb=True, image_weight=1)
    img_origin = Image.fromarray(visualization_origin)

    visualization_cam = show_cam_on_image(fp_rgb_fig, grayscale_cam, use_rgb=True, image_weight=0)
    img_cam = Image.fromarray(visualization_cam)
    if save_imgs:
        img_cam.save(f'grad_cam_cam_{model_name}.png')
        img_origin.save(f'grad_cam_origin_{model_name}.png')
    return visualization_origin, visualization_cam


def main():
    model_name = "ResNet50"
    model = model_dict[model_name](img_channels=img_channels, num_classes=num_classes).to(device)
    model_path = f"models/{model_name}.pth"
    test = np.load("./dataset/test.npy")
    labels = test[:, -1]
    features = test[:, :-1]
    test_ds = RNAseqDataset(features, labels, trans=transform)

    select = []
    rows = 10
    fig = None
    for r in range(rows):
        img_id = random.randint(0, len(labels) - 1)
        select.append(img_id)
        input = test_ds[img_id][0]
        print(test_ds[img_id][1])
        origin, cam = grad_cam(model, input, model_name)
        temp_img = np.concatenate((origin, cam), axis=1)
        if r == 0:
            fig = temp_img
        else:
            fig = np.concatenate((fig, temp_img), axis=0)

    img = Image.fromarray(fig)
    img.save("visualize.png")
    selected_features = test[select, :-1]
    selected_labels = test[select, -1]
    test_ds = RNAseqDataset(selected_features, selected_labels, trans=transform)
    test_loader = DataLoader(test_ds, batch_size=2, num_workers=2, drop_last=True)
    eval_dict = model_eval(test_loader, model, model_path)
    for key in eval_dict.keys():
        print(f"{key} is {eval_dict[key]:.4f}")


if __name__ == "__main__":
    main()
