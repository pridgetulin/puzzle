import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
import time

from datasets import *
from torch.utils.data import DataLoader
from collections import namedtuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


# stage one ,unsupervised learning
class SimCLRStage1(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLRStage1, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),nn.GroupNorm(num_groups=64, num_channels=512),
                               #nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, out_1, out_2, batch_size, temperature=0.5):
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        N, _ = out.shape
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        #print(sim_matrix.shape,batch_size)
        mask = (torch.ones_like(sim_matrix) - torch.eye(N, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(N, -1)

        # 分子： *为对应位置相乘，也是点积
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

def set_optimizers(net,net_params,counter,lr_l):
    if net_params:
        load_params = torch.load(net_params, weights_only=True)
        net.load_state_dict(load_params)
    opt = torch.optim.Adam(params=net.parameters(), lr=lr_l[counter], weight_decay=0.005)
    return opt

# train stage one
def train():
    print('\nSTART---training----')
    csv_path = 'D:/note/h690/interior.csv'
    image_dir = 'D:/note/h690/sherd_images'
    out_path = 'D:/note/h690/jd_info.csv'
    label_path = 'D:/note/h690/label_info.csv'
    #net_params = 'pretrain_resnet50.pth'
    net_params = None
    lr_l = [1e-4, 1e-4, 1e-5]
    batch_size = 24
    epochs_l = [300, 300, 300]
    loss_epochs  = 1e5
    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(64),transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),#随机应用颜色抖动（概率为0.8）
                        transforms.RandomGrayscale(p=0.2), transforms.ToTensor()])#图像转为灰度图像（概率为0.2）
    datasets = Datasetpre(csv_path, image_dir, transform)
    dl_loder = DataLoader(datasets, batch_size=batch_size, shuffle=False, num_workers=1)

    model = SimCLRStage1(feature_dim=128).to(device)
    #model = nn.DataParallel(model)
    lossLR = Loss()#.to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    for counter, epochs in enumerate(epochs_l):
        optimizer = set_optimizers(model,net_params,counter,lr_l)
        for e in range(1, epochs + 1):
            train_epoch = time.time()
            model.train()
            total_loss = 0
            for batch, (imgL, imgR) in enumerate(dl_loder):
                #if
                torch.cuda.empty_cache()
                imgL, imgR = imgL.to(device), imgR.to(device)

                _, pre_L = model(imgL)
                _, pre_R = model(imgR)

                loss = lossLR(pre_L, pre_R, batch_size)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("epoch", e, "batch", batch, "loss:", loss.detach().item())
                total_loss += loss.detach().item()
                torch.cuda.empty_cache()
            epo_l = total_loss / len(datasets) * batch_size
            train_time = time.time() - train_epoch
            print('train_time:', train_time, 's ')
            print("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            print('epochs:',e,"epoch loss:", epo_l)

            if epo_l <= loss_epochs:
                loss_epochs = epo_l
                net_params = 'pretrain_resnet50.pth'
                torch.save(model.state_dict(), net_params)
if __name__ == "__main__":
    train()