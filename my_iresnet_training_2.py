import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, dataloader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import os
import numpy as np
import argparse
import sys
from PIL import Image,ImageFile
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

import mymodel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ImageFile.LOAD_TRUNCATED_IMAGES = True
Batchsize=4
epochs=100
minibatches=2000
drop_ratio=0.3
lr=1e-6
weight_decay=0.
resize=120

def load_model(path):
    dataclass = len(os.listdir(path))
    model = mymodel.Backbone(50,drop_ratio,'ir_se')
    model.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                       nn.Dropout(drop_ratio),
                                       nn.Flatten(),
                                       nn.Linear(512 * 7 * 7, dataclass),
                                       nn.BatchNorm1d(dataclass))
    print('IR-SE-50 generated')

    # # load part of the weights
    # checkpoint = torch.load("model_ir_se50.pth")
    # model_dict = model.state_dict()
    # checkpoint = {k: v for k, v in checkpoint.items() if
    #               (k in model_dict) and (model_dict[k].shape == checkpoint[k].shape)}
    # model_dict.update(checkpoint)
    # model.load_state_dict(model_dict)
    # for parameter in model.parameters():
    #     parameter.requires_grad = False
    # print('weights loaded')

    model=model.to(device)

    return model, dataclass

def set_opt(model):
    optimizer = Adam(model.parameters(),lr,weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    return optimizer,criterion

def dataLoader(path):
    dataset = ImageFolder(root=path,transform = Compose([
        Resize(resize),
        CenterCrop(112),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    )
    len_train=int(len(dataset)*0.9)
    len_val=len(dataset)-len_train
    train_set, val_set = torch.utils.data.random_split(dataset, [len_train,len_val])
    train_loader=DataLoader(train_set,batch_size=Batchsize,shuffle=True)
    val_loader=DataLoader(val_set,batch_size=Batchsize,shuffle=True)
    print('[Length of Loader] Train, Val :',len(train_loader.dataset),',',len(val_loader.dataset))

    return train_loader,val_loader

def train_model(model, train_loader,criterion,optimizer,epoch,dataclass):
    model.train()
    train_loss=[]

    for idx,(image,label) in enumerate(train_loader):
        image = image.to(device)
        label = nn.functional.one_hot(label,num_classes=dataclass)
        label = label.type(torch.float32).to(device)

        optimizer.zero_grad()
        output=model(image)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        if idx%minibatches == 0:
            train_loss.append(loss.item())
            print("\nTrain Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(epoch,
            idx*len(image), len(train_loader.dataset), 100.*idx / len(train_loader),
            loss.item()))
            
    return train_loss

def evaluate_model(model,test_loader,criterion,model_path,dataclass):
    torch.save(model.state_dict(), model_path)
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for image,olabel in test_loader:
            image = image.to(device)
            label = nn.functional.one_hot(olabel,num_classes=dataclass)
            label = label.type(torch.float32).to(device)
            output = model(image)
            test_loss += criterion(output,label).item()
            etc, prediction = output.max(1,keepdim=True)
            correct+= torch.eq(prediction,olabel.to(device).view_as(prediction)).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / len(test_loader)
    return test_loss, test_accuracy


def save_loss_plot(loss1,loss2,result_path):
    d=datetime.datetime.now()
    x=np.arange(1,epochs+1)
    fig,axes = plt.subplots(2,1)

    # Draw plot
    axes[0].plot(x,loss1)
    axes[1].plot(x,loss2)

    # X,Y labeling
    fig.text(0.5, 0.015, 'Epoch', ha='center', va='center')
    fig.text(0.02, 0.5, 'Loss', ha='center', va='center', rotation='vertical')

    # Set title
    axes[0].set_title('Train Loss')
    axes[1].set_title('Val Loss')

    plt.tight_layout()
    fig.savefig(result_path+'Train1_loss_{}_{}.png'.format(epochs,d.strftime('%y%m%d_%H%M')),dpi=300)

def earlystopping():
    return 0

def train():
    d=datetime.datetime.now()
    path='../Dataset/211022_Data/Images/Training_4_all_H/Trainset/sketch/'
    result_path='../Dataset/211022_Data/Images/Training_4_all_H/Results/'
    model_path='../Dataset/211022_Data/Images/Training_4_all_H/Results/model_{}_{}.pt'.format(resize,d.strftime('%y%m%d'))
    model,dataclass = load_model(path)
    train_loader,val_loader = dataLoader(path)
    optimizer,criterion=set_opt(model)
    train_loss_lst=[]
    test_loss_lst=[]
    test_acc_lst=[]
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in tqdm(range(epochs)):
        train_loss=train_model(model,train_loader,criterion,optimizer,epoch,dataclass)
        test_loss,test_acc=evaluate_model(model,val_loader,criterion,model_path,dataclass)
        print('\n[EPOCCH:{}], \tVal Loss: {:.4f},\tVal Acc: {:.3f} % \n'.format(epoch,test_loss,test_acc))
        train_loss_lst.append(sum(train_loss)/len(train_loss))
        test_loss_lst.append(test_loss)
        test_acc_lst.append(test_acc)
        scheduler.step()

    save_loss_plot(train_loss_lst,test_loss_lst,result_path)

if __name__ =='__main__':
    train()