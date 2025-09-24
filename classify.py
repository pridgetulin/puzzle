from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

from pretrain import SimCLRStage1
from datasets import *
from torch.utils.data import DataLoader
from collections import namedtuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import time

device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")

# stage two ,supervised learning
class SimCLRStage2(nn.Module):
    def __init__(self, num_class):
        super(SimCLRStage2, self).__init__()
        # encoder
        self.f = SimCLRStage1().f
        # classifier
        self.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(inplace=True),nn.Linear(512, num_class))
        for param in self.f.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

def set_optimizers(net,net_params,counter,lr_l):
    if net_params:
        load_params = torch.load(net_params, weights_only=True)
        net.load_state_dict(load_params,strict=False)
    opt = torch.optim.Adam(params=net.fc.parameters(), lr=lr_l[counter], weight_decay=0.005)
    return opt

def main():
    # 配置参数
    print('\nSTART---training----')
    #csv_path = 'D:/note/h690/jd_sherds_info.csv'
    image_dir = 'D:/note/h690/sherd_images'
    #out_path = 'D:/note/h690/jd_info.csv'
    #label_path = 'D:/note/h690/label_info.csv'
    ## 内部
    ex_class = 'D:/note/h690/ex_jd_info.csv'
    ex_label = 'D:/note/h690/ex_label_info.csv'

    net_params = 'pretrain_resnet50.pth'
    lr_l = [1e-4, 3e-5, 1e-5]
    batch_size = 1
    epochs_l = [100, 100, 100]
    loss_epochs = 1e5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(64),
        transforms.ToTensor()])

    datasets = Datasets(ex_class, image_dir, ex_label, transform)
    dl_loder = DataLoader(datasets, batch_size = batch_size, shuffle = True, num_workers = 1)

    model = SimCLRStage2(num_class = 13).to(device)
    #model.load_state_dict(torch.load(net_params, map_location='cpu'), strict=False)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for counter, epochs in enumerate(epochs_l):
        optimizer = set_optimizers(model, net_params, counter, lr_l)
        for e in range(1, epochs + 1):
            train_epoch = time.time()
            #train_correct = 0
            #train_total = 0

            model.train()
            total_loss = 0
            step = 0
            for i, (img, targets) in enumerate(dl_loder, 1):
                img = img.to(device)
                targets = targets.float().to(device)

                optimizer.zero_grad()
                outputs = model(img)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()
                total_loss += loss.detach().item()
                step += 1
            epo_l = total_loss / step
            train_time = time.time() - train_epoch
            print('train_time:', train_time, 's ')
            print("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            print('epochs:', e, "epoch loss:", epo_l)

            if epo_l <= loss_epochs:
                loss_epochs = epo_l
                net_params = 'class_resnet50.pth'
                torch.save(model.state_dict(), net_params)
if __name__ == '__main__':
    main()