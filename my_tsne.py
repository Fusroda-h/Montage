import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import save
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import my_evalREAL_rect
import torch
import os
import mymodel
import my_evalREAL_rect as mev
import my_evalREAL_Forensic_transfered as mev2
import my_evalREAL_Forensic_transfered_152 as mev3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_fvmat(model, image_path, sketch_path):
    class_list, g_file, p_file = mev.getGalnProbeSet(image_path,sketch_path)
    gall_mat = []
    pall_mat =[]

    for ele in g_file:
        fv1 = mev.extractFV(ele, model)
        gall_mat.append(fv1)
    for ele in p_file:
        fv2 = mev.extractFV(ele,model)
        pall_mat.append(fv2)

    gmat = np.array(gall_mat) # (100,512)
    pmat = np.array(pall_mat) # (100,512)
    matall=np.concatenate((gmat,pmat),axis=0)
    tsne = TSNE(random_state=0)
    fv_tsne = tsne.fit_transform(matall)

    return fv_tsne, len(matall)

def makefigure(fv_tsne,length, savepath):
    # draw TSNE plot

    colors = ['#476A2A', '#7851B8']
    #'#BD3430', '#4A2D4E', '#875525','#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']

    # 시각화
    for i in range(length): # 0부터  digits.data까지 정수
        plt.scatter(fv_tsne[i, 0], fv_tsne[i, 1], # x, y 
                color=colors[i//100]) # 색상

    plt.xlim(fv_tsne[:, 0].min(), fv_tsne[:, 0].max()) # 최소, 최대
    plt.ylim(fv_tsne[:, 1].min(), fv_tsne[:, 1].max()) # 최소, 최대
    plt.xlabel('t-SNE feature 0') # x축 이름
    plt.ylabel('t-SNE feature 1') # y축 이름
    plt.savefig(savepath)
    plt.show() # 그래프 출력


if __name__ == '__main__':
    
    path = '../Dataset/211022_Data/Images/Training_5_100_H/'
    image_path = path+"Testset/gt/"
    sketch_path = path+"Testset/sketch/"
    save_path = path+"Result/tsne/"

    ######################################################################
    ####################### WRITE BEFORE EXECUTION #######################
    fig_name = 'random_100_train_1000_ir152_resize120_2.png'
    ######################################################################

    savepath = save_path+fig_name

    # model = mev.load_state()
    # model_path = '../Dataset/211022_Data/Images/Training_1_1000_H/Results/model_120_211122.pt'
    model_path = 'weights/model_120_211129_ir152.pt'
    model2 = mev3.load_state(model_path)

    os.makedirs(save_path,exist_ok=True)
    fv_tsne, l = make_fvmat(model2,image_path,sketch_path)
    makefigure(fv_tsne,l,savepath)
