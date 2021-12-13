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

from pathlib import Path

from numpy import *
import matplotlib.pyplot as plt
import operator
from tqdm import tqdm
import datetime
import mymodel

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_state():
    model = mymodel.Backbone(50, 1.,'ir_se').to(device)
    print('IR-SE-50 generated')
    checkpoint = torch.load("model_ir_se50.pth")
    model_dict = model.state_dict()

    checkpoint = {k: v for k, v in checkpoint.items() if
                  (k in model_dict) and (model_dict[k].shape == checkpoint[k].shape)}
    model_dict.update(checkpoint)

    model.load_state_dict(model_dict)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    print("model loaded!")
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

    img_file = Image.open(img_path).convert('RGB')
    testX = transforms(img_file)
    testX = testX.unsqueeze(0).to(device)
    vector = model(testX)
    # norm_v = F.normalize(vector, p=2, dim=1)
    feature = np.squeeze(vector.data.tolist())

    return feature

def getGalnProbeSet(image_path,sketch_path):
    img_list = os.listdir(image_path)
    sketch_list = os.listdir(sketch_path)

    p_file = []
    cls_dict = {}
    gall_list = []

    for i,img_ele in enumerate(img_list):
        img_title = img_ele[:-4]
        cls_dict[i] = img_title
        for sketch_ele in sketch_list:
            label = sketch_ele.split('.')[0] # For org_sketch: '.', sketch/H/: '_'
            if img_title in label:
                p_file.append(sketch_path + sketch_ele)
        total_path = image_path + img_ele
        gall_list.append(total_path)

    print("Number of fiels in Gallery :",len(gall_list))
    print("Number of files to probe :",len(p_file),end='\n')

    return cls_dict, gall_list, p_file

def calSim(feature, g_mat, ranknum):

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

    return indices[:ranknum]

def evalrank(model,classdict,pfile,gall_mat,ranknum):
    count=0
    pred_files = []
    for p in pfile:
        p_fv = extractFV(p,model)
        indice = calSim(p_fv,gall_mat,ranknum)
        plus=0
        pname = p.split('/')[-1].split('.')[0]
        for i in indice:
            if classdict[int(i)] in pname: # probe name must include gt / adding 'fake'
                plus+=1
            if plus > 1:
                print('Multiple class error')
        count+=plus
    try:
        accuracy = count/len(pfile)*100
    except:
        print("pfile 0")
        accuracy=0
    
    return accuracy

def match_sim(model,image_path,sketch_path,ranknum):
    classdict, gfile, pfile = getGalnProbeSet(image_path,sketch_path)
    gall_mat = []
    for ele in gfile:
        fv = extractFV(ele,model)
        gall_mat.append(fv)
    
    # print one ranknum accuracy
    accuracy = evalrank(model,classdict,pfile,gall_mat,ranknum)
    print('[ Accuracy of RANK{} ] : {:.4f}'.format(ranknum,accuracy))

    return accuracy
    # for k in ranknum:
    #     accuracy = evalrank(model,classdict,pfile,gall_mat,ranknum)

def eval():
    ground_path = '../Dataset/Face_detection_image/'
    image_path = ground_path+'Virtual_human_image/selected_image/sorted_all/'
    save_path = "../Dataset/201022_Results/prediction/"
    model_path = ground_path+'Results/model_200_211123.pt'
    model = load_state()

    gender = ['M','W']
    age = [20,30,40,50,60]
    k = [1,5]
    acc_all =[]
    for i in k:
        for g in gender:
            for a in age:
                sketch_path = ground_path+'Montage_image/{}_{}_Montage/'.format(g,a)
                ranknum = i
                acc = match_sim(model,image_path,sketch_path,ranknum)
                acc_all.append(acc)
    print(acc_all)

if __name__ == '__main__':
    eval()