import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize

import os
import numpy as np
import argparse
import sys
from PIL import Image,ImageFile

import iresnet as irsenet
from pathlib import Path

from numpy import *
import matplotlib.pyplot as plt
import operator
from tqdm import tqdm
import datetime
import mymodel

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Batchsize=4
epochs=100
minibatches=1000
drop_ratio=0.3
lr=1e-3
weight_decay=1e-6

def load_state(model_path):
    model = mymodel.Backbone(50,drop_ratio,'ir_se').to(device)
    model= nn.Sequential(model,
                        nn.Dropout(drop_ratio),
                        nn.Linear(512,512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(drop_ratio),
                        nn.Linear(512,1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(drop_ratio),
                        nn.Linear(1024,1000),
                        nn.BatchNorm1d(1000))
    print('Transfered IR-SE-50 generated')

    # load part of the weights
    checkpoint = torch.load(model_path)
    # model_dict = model.state_dict()
    # checkpoint = {k: v for k, v in checkpoint.items() if
    #               (k in model_dict) and (model_dict[k].shape == checkpoint[k].shape)}
    # model_dict.update(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    print('weights loaded')
    model=model.to(device)
    model.eval()

    return model

def calnsaveResults(indices, rate_c, save_file_name):
    count_c = []
    writer = open(save_file_name + '.txt', 'w')

    for rank in range(len(indices)):
        cnt_c = 0
        for item_c in rate_c:
            if rank == item_c:
                cnt_c += 1
            else:
                continue

        count_c.append(cnt_c)

    print(count_c)
    sum_rate_c = []
    temp1 = 0

    for add1 in count_c:
        temp1 += add1
        sum_rate_c.append(temp1)

    print("cosine_rate_c : ", sum_rate_c)
    print(len(sum_rate_c))

    X = range(len(count_c))
    C = [value / len(rate_c) * 100 for value in sum_rate_c]

    print("cosine rank: ", C)
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.title("Result")
    plt.xlabel("rank")
    plt.ylabel("recognition rate")

    plt.plot(X, C)
    plt.legend(['cosine_dist'], loc='lower center')
    plt.savefig(save_file_name + '.png')
    plt.close()

    for item in C:
        writer.write("%s " % item)
    writer.write("\n")

    return C

def saveResults(acc_list, save_file_name):

    writer = open(save_file_name + '.txt', 'w')

    rank = []
    for i in range(1, len(list(acc_list)) + 1):
        rank.append(i)

    plt.xlabel("rank")
    plt.ylabel("recognition rate")
    plt.plot(rank, acc_list)
    plt.legend(['cosine_dist'], loc='lower center')
    plt.savefig(save_file_name + '.png')
    plt.close()

    for item in acc_list:
        writer.write("%s " % item)
    writer.write("\n")

    return acc_list

def getLable(class_list):
    label = {}
    for cls, i in enumerate(class_list):
        label[i] = cls
    label = dict(sorted(label.items(), key=operator.itemgetter(1)))

    return label

def extractFV(img_path, model):

    transforms = Compose([
        Resize(200),
        CenterCrop(112),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    img_file = Image.open(img_path)#.convert('RGB')
    testX = transforms(img_file)
    testX = testX.unsqueeze(0).to(device)
    vector = model(testX)
    # norm_v = F.normalize(vector, p=2, dim=1)
    feature = np.squeeze(vector.data.tolist()[0])

    return feature

def getGalnProbeSet(image_path,sketch_path):
    img_list = os.listdir(image_path)
    sketch_list = os.listdir(sketch_path)

    p_file = []
    cls_list = []
    gall_list = []

    for img_ele in img_list:
        img_title = img_ele[:-4]
        cls_list.append(img_title)
        for sketch_ele in sketch_list:
            label = sketch_ele.split('_')[0] # For org_sketch: '.', sketch/H/: '_'
            if img_title == label:
                p_file.append(sketch_path + sketch_ele)
        total_path = image_path + img_ele
        gall_list.append(total_path)

    print("Number of GT_list :")
    print(len(gall_list))
    print("Number of files to predict :")
    print(len(p_file))

    return cls_list, gall_list, p_file

def calSim(feature, g_mat):

    cls_num = len(g_mat)
    g_mat = torch.from_numpy(asarray(g_mat)).float()
    g_mat_tensor = Variable(g_mat)
    # g_mat_tensor = F.normalize(g_mat_tensor, p=2, dim=1)

    probe_fv = torch.from_numpy(asarray(feature)).float()
    probe_fv_tensor = Variable(probe_fv)
    probe_fv_tensor = probe_fv_tensor.view(-1, 1)
    # probe_fv_tensor = F.normalize(probe_fv_tensor, p=1, dim=1)
    dot_prod = torch.mm(g_mat_tensor, probe_fv_tensor).view(-1,)
    values, indices = torch.topk(dot_prod, cls_num)

    return values, indices

def evaluate():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    image_path = "../../Dataset/211022_Data/Images/Training_1_1000_H/Testset/gt/"
    sketch_path = "../../Dataset/211022_Data/Images/Training_1_1000_H/Testset/sketch/"
    save_path = "../../Dataset/201022_Results/prediction/"
    model_path='../../Dataset/211022_Data/Images/Training_1_1000_H/Results/model_120_211122.pt'
    model = load_state(model_path)
    os.makedirs(save_path, exist_ok=True)

    class_list, g_file, p_file = getGalnProbeSet(image_path,sketch_path)
    gall_mat = []

    for ele in g_file:
        fv = extractFV(ele, model)
        gall_mat.append(fv)

    rate_c = []
    print("********* probe_file length: ", len(p_file))
    label = getLable(class_list)

    for iter, f in tqdm(enumerate(p_file)):
        # print("[", iter, "] curr file: ", f)
        f_split = f.split('/')[-1].split('_')[0] # For org_sketch: '.', sketch/H/: '_'
        for label_ele in class_list:
            if label_ele == f_split:
                curr_label = label[label_ele]

        pivot_fv = extractFV(f, model)
        values, indices = calSim(pivot_fv, gall_mat)

        for idx_c, pred_lable in enumerate(indices):
            if pred_lable == curr_label:
                rate_c.append(idx_c)
                break
    d=datetime.datetime.now()
    save_file_name = save_path + 'result' + d.strftime('%Y_%m_%d_%H_%M')
    calnsaveResults(indices, rate_c, save_file_name)

if __name__ == '__main__':
    evaluate()