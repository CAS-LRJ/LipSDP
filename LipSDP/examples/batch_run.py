import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torchsummary import summary
from MNIST_Net import Network
from scipy.io import savemat
from math import sqrt,floor,ceil
from mnist_example import create_data_loaders,BATCH_SIZE
import numpy as np
import os
import time

MODEL_FILE='mnist_fnn_tanh_30_20'
Lip_Const=63.695
epsilon=0.004
INPUT_SIZE = 784
OUTPUT_SIZE = 10
MAX_R=1000

def main():
    start=time.clock()
    train_loader, test_loader = create_data_loaders()
    model = torch.load('saved_model/'+MODEL_FILE+'.pth')
    model.eval()
    batch(model,test_loader)
    end=time.clock()
    print(end-start)

def batch(model,test_loader):
    lip=Lip_Const*sqrt(INPUT_SIZE)/sqrt(OUTPUT_SIZE)
    # print(lip) 
    dist={}   
    total=0    
    for data, labels in test_loader:        
        data = data.view(BATCH_SIZE, -1)
        for i in range(BATCH_SIZE):          
            total=total+1
            result=model(data[i]).detach().numpy().astype(np.float64)
            labeltemp=model(data[i]).detach().numpy().astype(np.float64).argsort()[-2:][::-1]
            #result[labeltemp[0]]-result[labeltemp[1]]
            # print(labels[i],labeltemp[0],labeltemp[1])
            # print(result[model(data[i]).detach().numpy().astype(np.float64).argsort()])
            robust_r=(result[labeltemp[0]]-result[labeltemp[1]])/2/lip
            robust_r=int(ceil(robust_r*1000))
            if dist.__contains__(robust_r):
                dist[robust_r]=dist[robust_r]+1
            else:
                dist[robust_r]=1
            # print(robust_r)
    print(dist)
    cur=0
    for i in range(MAX_R):
        if dist.__contains__(i):
            cur=cur+dist[i]
        print('The percentage of r over ',i/1000,'is',100-cur/total*100,'%')
        if cur==total:
            break

if __name__ == "__main__":
    main()