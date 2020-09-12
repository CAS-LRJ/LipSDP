import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torchsummary import summary
from MNIST_Net import Network
from scipy.io import savemat
from mnist_example import create_data_loaders,BATCH_SIZE
import numpy as np
import os

ACTIVATION='sigmoid'
MODEL_FILE='mnist_fnn_sigmoid_20'
INPUT_SIZE = 784
OUTPUT_SIZE = 10
#NUM_INPUTS no more than 100(BATCH_SIZE)
NUM_INPUTS = 3
EPSILON = 1e-3

def main():
    train_loader, test_loader = create_data_loaders()
    model = torch.load('saved_model/'+MODEL_FILE+'.pth')
    model.eval()
    summary(model, (1, INPUT_SIZE))
    weights,bias=extract_params(model)
    # print(weights)
    # return
    test_n_inputs(model,test_loader,weights,bias)
    # print(weights)
    # print(bias)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dtanh(x):
    return 1-np.power(np.tanh(x),2)

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def drelu(x):
    if x<0:
        return 0
    return 1

def test_n_inputs(model,test_loader,weights,bias):

    num_batch=0
    all_info=[]
    for data, labels in test_loader:        
        data = data.view(BATCH_SIZE, -1)
        for i in range(BATCH_SIZE):
            if ACTIVATION=='relu':
                alpha=1.0
                beta=0.0
            if ACTIVATION=='tanh':
                alpha=1.0
                beta=0.0
            if ACTIVATION=='sigmoid':
                alpha=0.25
                beta=0.0
            raw_data=data[i].detach().numpy().astype(np.float64)
            upper_data=raw_data+EPSILON
            upper_data[upper_data > 1.0] = 1.0            
            lower_data=raw_data-EPSILON
            lower_data[lower_data < 0] = 0
            assert(len(weights)==len(bias))            
            for layer in range(len(weights)):
                weight=weights[layer]
                lower_bound=[]
                upper_bound=[]
                for neuron in range(len(weight)):
                    neuron_weight=weight[neuron]
                    neuron_lower=0.0
                    neuron_upper=0.0
                    for num in range(len(neuron_weight)):
                        if(neuron_weight[num]>0):
                            neuron_upper+=neuron_weight[num]*upper_data[num]
                            neuron_lower+=neuron_weight[num]*lower_data[num]
                        else:
                            neuron_upper+=neuron_weight[num]*lower_data[num]
                            neuron_lower+=neuron_weight[num]*upper_data[num]
                    lower_bound.append(neuron_lower)
                    upper_bound.append(neuron_upper)
                    if layer<len(weights)-1:
                        if ACTIVATION=='relu':
                            alpha=min(drelu(neuron_lower),alpha)
                            beta=max(drelu(neuron_upper),beta)
                        if ACTIVATION=='tanh':
                            alpha=min(dtanh(neuron_lower),dtanh(neuron_upper),alpha)
                            beta=max(dtanh(neuron_lower),dtanh(neuron_upper),beta)
                            if neuron_lower<=0.0 and neuron_upper>=0.0:
                                beta=1.0
                        if ACTIVATION=='sigmoid':
                            alpha=min(dsigmoid(neuron_lower),dsigmoid(neuron_upper),alpha)
                            beta=max(dsigmoid(neuron_lower),dsigmoid(neuron_upper),beta)
                            if neuron_lower<=0.0 and neuron_upper>=0.0:
                                beta=0.25                                
                lower_bound=np.add(lower_bound,bias[layer])
                upper_bound=np.add(upper_bound,bias[layer])                
                if layer<len(weights)-1:
                    if ACTIVATION=='relu':
                        lower_bound[lower_bound<0]=0
                        upper_bound[upper_bound<0]=0
                    if ACTIVATION=='tanh':
                        lower_bound=np.tanh(lower_bound)
                        upper_bound=np.tanh(upper_bound)
                    if ACTIVATION=='sigmoid':
                        lower_bound=sigmoid(lower_bound)
                        upper_bound=sigmoid(upper_bound)
                lower_data=lower_bound
                upper_data=upper_bound            
            result=model(data[i]).detach().numpy().astype(np.float64)
            labeltemp=model(data[i]).detach().numpy().astype(np.float64).argsort()[-2:][::-1]
            info=[alpha,beta,(result[labeltemp[0]]-result[labeltemp[1]])/np.sqrt(2),num_batch,i,labeltemp[0],labeltemp[1]]
            all_info.append(info)
        num_batch+=1

    all_info=np.array(all_info)
    all_info=all_info[all_info[:,0].argsort()[::-1]][:NUM_INPUTS]
    # print(all_info)
    with open('saved_info/'+MODEL_FILE+'.txt','w') as f:
        for info in all_info:
            f.write('Alpha:{:.8f} Beta:{:8f} (Largest_Output-Second_Large_Output)/sqrt(2):{:8f} Batch:{:d} No.:{:d} label:{:d} attack_label:{:d}\n'.format(info[0],info[1],info[2],int(info[3]),int(info[4]),int(info[5]),int(info[6])))


def extract_params(net):
    """Extract weights of trained neural network model

    params:
        * net: torch.nn instance - trained neural network model

    returns:
        * weights: list of arrays - weights of neural network
        * bias: list of arrays - bias of neural network
    """

    weights = []
    bias = []
    for param_tensor in net.state_dict():
        tensor = net.state_dict()[param_tensor].detach().numpy().astype(np.float64)

        if 'weight' in param_tensor:
            weights.append(tensor)
            #print(tensor)

        if 'bias' in param_tensor:
            bias.append(tensor)
            #print(tensor)
    return weights,bias


if __name__ == "__main__":
    main()