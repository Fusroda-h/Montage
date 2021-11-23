import torch
import torch.nn as nn
import PIL.Image
import torch.nn.functional as F

import os
import numpy as np
import argparse
import sys
# sys.path.append("..")
# from arcface_model import Backbone, Arcface

import iresnet as irsenet

from model_resnet import resnet50
from config import get_config

from pathlib import Path

from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_state(save_path, epoch):
    conf = get_config()
    # model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(device)
    model = irsenet.iresnet50(False, fp16=False).to(device)
    print('IR-SE-50 generated')
    # checkpoint = torch.load(save_path / 'model_{}.pth'.format(epoch))
    # checkpoint = torch.load("/data/hyeonjung/workspace/FPS/mProject/save/ir_se50_msceleb.pth.tar")
    # checkpoint = torch.load("/data/hyeonjung/workspace/FPS/mProject/save/model_ir_se50.pth")
    checkpoint = torch.load("/data/hyeonjung/workspace/FPS/mProject/save/glint_360K_ir50_webface_finetuned.pth")
    # checkpoint = torch.load("/data/hyeonjung/workspace/FPS/Simple_Face_recognition/traced_model.pt")
    model_dict = model.state_dict()

    checkpoint = {k: v for k, v in checkpoint.items() if
                  (k in model_dict) and (model_dict[k].shape == checkpoint[k].shape)}
    model_dict.update(checkpoint)

    model.load_state_dict(model_dict)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    print("pretrained epoch {} model load complete".format(epoch))
    return model

# def load_state(save_path, epoch):
#     # model creation
#     conf = get_config()
#     # model = resnet50(num_classes=conf.class_num)
#     model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
#
#     # print('ResNet50 generated')
#     print('IR-SE-50 generated')
#     pretrained_model = torch.load(save_path / 'model_{}.pth'.format(epoch))
#     model = DataParallel(model).cuda()
#     model.module.load_state_dict(pretrained_model)
#
#     print("pretrained epoch {} model load complete".format(epoch))
#
#     return model




def extractFV(img_path, save_path, model):
    from torchvision import transforms

    transforms = transforms.Compose([
        transforms.Resize(130),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    img_file = PIL.Image.open(img_path)

    testX = transforms(img_file)
    testX = torch.unsqueeze(testX, 0).to(device)
    vector = model(testX)
    # norm_v = F.normalize(vector, p=2, dim=1)
    feature = np.squeeze(vector.data.tolist()[0])

    np.savetxt(save_path, np.c_[feature], delimiter=' ')

    return feature



if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # summary writer
    log_path = 'analysis'
    writer = SummaryWriter(log_path)
    pretrained_model_root = Path('./save/')

    model = load_state(pretrained_model_root, 0)

    # s_path = "/data/hyeonjung/workspace/FPS/DB/TEST/crop_s/"
    # i_path = "/data/hyeonjung/workspace/FPS/DB/TEST/crop_m/"
    s_path = "/data/hyeonjung/workspace/FPS/DB/TEST/0050_original_images/crop_female/"
    i_path = "/data/hyeonjung/workspace/FPS/DB/TEST/0050_original_images/crop_male/"

    # i_path = "/data/hyeonjung/workspace/FPS/DB/face_sample/probe/"
    # s_path = "/data/hyeonjung/workspace/FPS/DB/face_sample/gallery/"

    # s_path = "/data/hyeonjung/workspace/FPS/DB/TEST/IITD/photo/"
    # i_path = "/data/hyeonjung/workspace/FPS/DB/TEST/IITD/sketch/"

    save_path = "/data/hyeonjung/workspace/FPS/ms_model/age50/"
    os.makedirs(save_path, exist_ok=True)

    s_s_path = save_path + "sketch/"
    i_s_path = save_path + "image/"

    os.makedirs(s_s_path, exist_ok=True)
    os.makedirs(i_s_path, exist_ok=True)

    s_list = os.listdir(s_path)
    s_list.sort()
    i_list = os.listdir(i_path)
    i_list.sort()

    for s_file in s_list:
        img_path = s_path + s_file
        # file_name = s_file.split("_")[0]
        # save_path = s_s_path + file_name + ".txt"

        save_path = s_s_path + s_file[:-4] + ".txt"

        extractFV(img_path, save_path, model)

    print("sketch done!")

    for i_file in i_list:
        img_path = i_path + i_file
        save_path = i_s_path + i_file[:-4] + ".txt"
        extractFV(img_path, save_path, model)

    print("image done!")





