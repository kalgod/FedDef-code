import os
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from math import ceil
import pickle
import warnings
import copy
from parse import parse_arg
import KitNET
import sys
import datetime
sys.path.insert(0, '../attack')
sys.path.insert(0, '../auto')
sys.path.insert(0, '../adver')
# https://github.com/Harry24k/adversarial-attacks-pytorch
from robustbench.utils import load_model, clean_accuracy
import torchattacks
print("torchattacks %s"%(torchattacks.__version__))
warnings.filterwarnings("ignore")

opt = parse_arg()
print(opt)

cuda = True if torch.cuda.is_available() else False

print(cuda)

args=opt
alpha=int(args.alpha*100)

if opt.use_ori==1:
    x=np.load('./data/{}.npy'.format(opt.dataset))
    result=x[:,-1]
    x=x[:,:-1]
    x = torch.from_numpy(x).type(torch.FloatTensor)
    with open("./kitsune_model/__model__{}.pkl".format(opt.dataset), 'rb') as f: net1 = pickle.load(f)# 读取模型信息
    #x,ben_min,ben_max=norm(x)
    x=net1.norm(x)
    #x=(x-minn)/(maxx-minn+1e-9)

else:
    if opt.defense!=1:
        x=np.load('./gan_dataset/attack/x_{}_{}_{}.npy'.format(args.dataset, args.model, args.defense))
        label=np.load('./gan_dataset/attack/y_{}_{}_{}.npy'.format(args.dataset, args.model, args.defense))
    else:
        x=np.load('./gan_dataset/attack/x_{}_{}_{}_alpha_{}.npy'.format(args.dataset, args.model, args.defense,alpha))
        label=np.load('./gan_dataset/attack/y_{}_{}_{}_alpha_{}.npy'.format(args.dataset, args.model, args.defense,alpha))

    #x=x[:100]
    #label=label[:100]

    result=np.argmax(label,axis=1)
    #result[result!=0]=0
    x = torch.from_numpy(x).type(torch.FloatTensor)
    discrete=[]
    #print(x,result)

ben_x=x[result==0]
mal_x=x[result!=0]
ben_x=ben_x[:1000]
mal_x=mal_x[:1000]
mal_x=copy.deepcopy(x)

with open("./kitsune_model/__model__{}.pkl".format(opt.dataset), 'rb') as f: 
    net1 = pickle.load(f)# 读取模型信息
    
if opt.defense!=1:
    with open('./save_model/{}_{}_{}_model.pkl'.format(args.dataset,args.model,opt.defense), 'rb') as f: 
        net2 = pickle.load(f)
else:
    with open('./save_model/{}_{}_{}_alpha_{}_model.pkl'.format(args.dataset,args.model,opt.defense,alpha), 'rb') as f: 
        net2 = pickle.load(f)

alpha=int(opt.alpha*100)

images=x
labels=torch.from_numpy(result-result)
net2=net2.to('cpu')
model=net2
#总体上，FGSM>AutoPGD>PGD的acc，但是autopgd不稳定，因为dlr loss。
#cw攻击中有特殊逻辑，当adv_images攻击失败，就不会更新最后的对抗样本，所以如果100次生成后还没有攻击成功，那么就会维持原来的样本，因此距离可能=0（我们的防御），而对于其他的防御，在cic2017和unsw数据集中，早就acc=0，所以best L2=0，所以也不会更新了
#CW攻击有时候要好于PGD方法，且扰动距离还更小。
#但是PGD的优势在于，可以增大扰动限制，达到完美Acc=0，CW的w优化x时，可能会局部最优
#deepfool攻击是无目标的。普通防御的重构样本，相比我们的，更接近正常样本，所以更容易对抗攻击成正常的，而我们的样本攻击后大概率还是异常的样本，所以acc还是很高
#CIC2017数据集中，普通防御下，重构的样本本身Acc=0，比较接近正常样本的特征，我们的数据就更远离正常样本。这个数据集本身就容易攻击，所以很多距离=0
#普通防御下，重构的异常数据后，本身就接近正常样本了，所以dnn的初始Acc就很低，所以距离即使=0，acc也很低

# print("- Torchattacks")
atk = torchattacks.PGD(model, eps=40/255, alpha=6/255, steps=100, random_start=False)#cic是4/255+2/255，其他的是40/255+6/255
#atk = torchattacks.APGDT(model, norm='Linf', eps=40/255, steps=100,verbose=False)#cic2017-4/255,其他40/255
#atk = torchattacks.FAB(model, norm='Linf', eps=40/255, steps=100, n_restarts=1,alpha_max=0.1, eta=1.05, beta=0.9, verbose=True, seed=0,multi_targeted=False, n_classes=23)
#atk = torchattacks.DeepFool(model, steps=50, overshoot=0.02,eps=400/255) #cic2017-4/255,其他40/255
#atk = torchattacks.CW(model, c=0.01, kappa=0, steps=100, lr=0.1)
# atk = torchattacks.FGSM(model, eps=40/255) #只有cic-2017-4/255，其他40/255

#from advertorch.attacks.fast_adaptive_boundary import *
#adversary = FABAttack(model,norm='Linf',n_restarts=1,n_iter=100,eps=None,alpha_max=0.1,eta=1.05,beta=0.9,loss_fn=None,verbose=True)

atk.set_mode_targeted_by_label(quiet=True) # do not show the message
start = datetime.datetime.now()
adv_images = atk(images, labels)
#adv_images = adversary.perturb(images, labels)
end = datetime.datetime.now()

gen_z=adv_images

# print(x[0],gen_z[0])
print("dis: ",torch.sqrt(torch.mean((adv_images-images)**2)))

# results=net1.execute(x)
# print("pre kitsune is ",np.mean(results))
# log_probs=net2(x)
# result=np.argmax(log_probs.detach().numpy(),axis=1)
# result[result!=0]=1
# print("pre dnn acc is",100*np.mean(result),"%\n")

results=net1.execute(gen_z)
print("after kitsune is ",np.mean(results))
log_probs=net2(gen_z)
result=np.argmax(log_probs.detach().numpy(),axis=1)
result[result!=0]=1
print("after dnn acc is",100*np.mean(result),"%")
    
alpha=int(opt.alpha*100)

if opt.save_model==1: 
    if opt.defense!=1:
        np.save('./gan_dataset/attack/save/acc_{}_{}.npy'.format(args.dataset, args.defense),arr=acc_plot)
        np.save('./gan_dataset/attack/save/rmse_{}_{}.npy'.format(args.dataset,args.defense),arr=rmse_plot)
    else:
        np.save('./gan_dataset/attack/save/acc_{}_{}_alpha_{}.npy'.format(args.dataset, args.defense,alpha),arr=acc_plot)
        np.save('./gan_dataset/attack/save/rmse_{}_{}_alpha_{}.npy'.format(args.dataset,args.defense,alpha),arr=rmse_plot)
