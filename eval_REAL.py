import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import numpy as np
import argparse

import sys
import PIL.Image

import iresnet as irsenet
from pathlib import Path

from numpy import *
import matplotlib.pyplot as plt
import operator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_state():
    model = irsenet.iresnet50(False, fp16=False).to(device)
    print('IR-SE-50 generated')

    checkpoint = torch.load("save/glint_360K_ir50_webface_finetuned.pth")
    model_dict = model.state_dict()

    checkpoint = {k: v for k, v in checkpoint.items() if
                  (k in model_dict) and (model_dict[k].shape == checkpoint[k].shape)}
    model_dict.update(checkpoint)

    model.load_state_dict(model_dict)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    print("model load!")
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
    from torchvision import transforms

    transforms = transforms.Compose([
        transforms.Resize(120),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    img_file = PIL.Image.open(img_path)

    testX = transforms(img_file)
    testX = torch.unsqueeze(testX, 0).to(device)
    vector = model(testX)
    # norm_v = F.normalize(vector, p=2, dim=1)
    feature = np.squeeze(vector.data.tolist()[0])

    return feature

def calculate_sim_matrice(feature, g_mat):
    cls_num = len(g_mat)
    g_mat = torch.from_numpy(np.asarray(g_mat)).float()
    g_mat_tensor = Variable(g_mat)
    # g_mat_tensor = F.normalize(g_mat_tensor, p=2, dim=1)

    probe_fv = torch.from_numpy(np.asarray(feature)).float()
    probe_fv_tensor = Variable(probe_fv)
    probe_fv_tensor = probe_fv_tensor.view(-1, 1)
    # probe_fv_tensor = F.normalize(probe_fv_tensor, p=1, dim=1)

    dot_prod = torch.mm(g_mat_tensor, probe_fv_tensor).view(-1,)
    values, indices = torch.topk(dot_prod, cls_num)

    return values, indices

def readTXTfile(fileName):
    fr = open(fileName)
    data_list = [line.rstrip() for line in fr.readlines()]
    return data_list

def getGalnProbeSet():

    image_path = "/data/hyeonjung/workspace/FPS/DB/REAL_TEST_PROJECT/1000/image/"
    img_list = os.listdir(image_path)

    sketch_path = "/data/hyeonjung/workspace/FPS/DB/REAL_TEST_PROJECT/sketch/"
    sketch_list = os.listdir(sketch_path)

    p_file = []
    cls_list = []

    gall_list = []

    for img_ele in img_list:
        img_title = img_ele[:-4]
        cls_list.append(img_title)
        for sketch_ele in sketch_list:
            label = sketch_ele.split('_')[0]
            if img_title == label:
                p_file.append(sketch_path + sketch_ele)
        total_path = image_path + img_ele
        gall_list.append(total_path)

    print(len(gall_list))
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
    model = load_state()

    save_path = "/data/hyeonjung/workspace/FPS/211026_result/Real/"
    os.makedirs(save_path, exist_ok=True)

    class_list, g_file, p_file = getGalnProbeSet()
    gall_mat = []

    for ele in g_file:
        fv = extractFV(ele, model)
        gall_mat.append(fv)

    rate_c = []
    print("********* probe_file length: ", len(p_file))
    label = getLable(class_list)

    for iter, f in enumerate(p_file):
        print("[", iter, "] curr file: ", f)
        f_split = f.split('/')[-1]
        for label_ele in class_list:
            if label_ele in f_split:
                curr_label = label[label_ele]

        pivot_fv = extractFV(f, model)
        values, indices = calSim(pivot_fv, gall_mat)

        for idx_c, pred_lable in enumerate(indices):
            if pred_lable == curr_label:
                rate_c.append(idx_c)
                break
    save_file_name = save_path + 'result'
    calnsaveResults(indices, rate_c, save_file_name)


if __name__ == '__main__':
    evaluate()