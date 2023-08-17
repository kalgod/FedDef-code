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
from attack.pgd import *
from attack.apgdt import *
from attack.cw import *
from attack.deepfool import *
from attack.fgsm import *
warnings.filterwarnings("ignore")
import time
import psutil

start_time=time.time()
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
model=net1

print("- Torchattacks")
#atk = FGSM(model, eps=40/255)
#atk = CW(model, c=1e-4, kappa=0, steps=100, lr=0.1,dataset=opt.dataset)#CW效果很差，一个是w的转换，一个是loss函数self.c，一个是逻辑上更新best_adv
atk = PGD(model, eps=80/255, alpha=6/255, steps=100, random_start=False)
#atk.set_mode_targeted_by_label(quiet=True) # do not show the message
start = datetime.datetime.now()
adv_images = atk.forward(images, labels)
end = datetime.datetime.now()

gen_z=adv_images

print(x[0],"\n",gen_z[0])
print("dis: ",torch.sqrt(torch.mean((adv_images-images)**2)))

results=net1.execute(x)
if opt.dataset=="kdd": acc=np.mean(results>0.3)
elif opt.dataset=="mirai": acc=np.mean(results>0.35)
elif opt.dataset=="cic2017": acc=np.mean(results>0.08)
else: acc=np.mean(results>0.3)
print("after kitsune is ",np.mean(results)," and acc is ",100*acc,"%")
log_probs=net2(x)
result=np.argmax(log_probs.detach().numpy(),axis=1)
result[result!=0]=1
print("pre dnn acc is",np.mean(result),"\n")

results=net1.execute(gen_z)
if opt.dataset=="kdd": acc=np.mean(results>0.3)
elif opt.dataset=="mirai": acc=np.mean(results>0.35)
elif opt.dataset=="cic2017": acc=np.mean(results>0.08)
else: acc=np.mean(results>0.3)
print("after kitsune is ",np.mean(results)," and acc is ",100*acc,"%")
log_probs=net2(gen_z)
result=np.argmax(log_probs.detach().numpy(),axis=1)
result[result!=0]=1
print("after dnn acc is",np.mean(result))
    
alpha=int(opt.alpha*100)

if opt.save_model==1: 
    if opt.defense!=1:
        np.save('./gan_dataset/attack/save/acc_{}_{}.npy'.format(args.dataset, args.defense),arr=acc_plot)
        np.save('./gan_dataset/attack/save/rmse_{}_{}.npy'.format(args.dataset,args.defense),arr=rmse_plot)
    else:
        np.save('./gan_dataset/attack/save/acc_{}_{}_alpha_{}.npy'.format(args.dataset, args.defense,alpha),arr=acc_plot)
        np.save('./gan_dataset/attack/save/rmse_{}_{}_alpha_{}.npy'.format(args.dataset,args.defense,alpha),arr=rmse_plot)
        
end_time=time.time()
process = psutil.Process()
memory_used = process.memory_info().rss /(1024*1024)
print("Memory used: {:.2f} MB".format(memory_used))
print("Time used: {:.2f} s".format(end_time-start_time))