##train
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from scipy.linalg import norm, pinv, toeplitz
from tqdm import tqdm, trange
from algorithm_toe import *
from paras12 import *
import torch

import torch.nn.functional as F
# import dataset_general_mq
from scipy import io
import os
from optimize_matrices import get_matrices
from torch.nn import Module, Parameter, ReLU
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.cuda.empty_cache()
# try to use gpu
if torch.cuda.is_available():
    device = torch.device('cuda:1')  # 指定使用第1张GPU
else:
    device = "cpu"
#仿真数据
data_path =   os.path.join(os.getcwd(),'res/data_8td_randA_randpphi_snr_train_k1_2/')
dataset_training = torch.load(os.path.join(data_path,"ula_training_data_randA_randphi_snr_8td.pt"))
dataset_testing = torch.load(os.path.join(data_path,"ula_testing_data_randA_randphi_snr_8td.pt"))
dataloader_training = DataLoader(dataset_training, 
                                 batch_size = batchSize, shuffle=True,drop_last=True)
dataloader_testing = DataLoader(dataset_testing, 
                                 batch_size = dataset_testing.X.shape[0], shuffle=True)

# K=3
batches = int(dataset_training.X.shape[0]/batchSize)
#print(trainingPoints/batchSize)
training_losslist = np.zeros(epochs * batches)
testing_losslist = np.zeros(epochs * batches)
rp = np.zeros(epochs * batches)
mean_pos_err = np.zeros(epochs * batches)
s = 2
layover=2
p = ((np.linspace((sparsityLevel*2 * 1 * 1.2) // maxit, sparsityLevel*2 * 1 * 1.2, maxit))+1).astype(int)
p_toe = ((np.linspace((s * 1 * 1.2) // maxit, s * 1 * 1.2, maxit))+1).astype(int)
phi, W_frob = get_matrices(M, N, data_path,matrix_dir=data_path)


# for model_name in models:
# model = LISTA(M, N, maxit, phi, lmbda)
# model = Toe_LISTA(M, N, maxit)
model = Toe_LISTA_Ada(M, N, maxit, p_toe)

model.to(device)
X, Y = dataset_training[:100]
print(X.shape, Y.shape)
X = X.numpy().T
Y = Y.numpy().T
XHX = X.T.conj() @ X 
XHXinv = pinv(XHX)
Phi = Y @ XHXinv @ X.T.conj() # conjugates omitted since X will be real
PhiH = torch.from_numpy(Phi.conj().T)
L = np.max(np.abs(np.linalg.eigvals(Phi.conj().T @ Phi)))

#print(model.state_dict()['Wre'])
with torch.no_grad():
    for name, param in model.named_parameters():
        if 'Wre' in name:
            param.copy_(1/L * PhiH.real)
        if 'Wie' in name:
            param.copy_(1/L * PhiH.imag)
        if 'win' in name: # for weighted lista
            param.copy_(1/L * torch.ones(model.maxit+1, M1))
        if 'wout' in name: # for weighted lista
            param.copy_(1/L * torch.ones(model.maxit+1, M2))
        if 'theta' in name:
            param.copy_(0.01/L * torch.ones(model.maxit+1))


# In[49]:
import pandas as pd
model_str = str(model)
# 找到第一个括号的位置
index = model_str.find('(')
# 如果找到了括号，截取字符串
if index != -1:
    cleaned_str = model_str[:index]
else:
    cleaned_str = model_str  # 如果没有找到括号，返回整个字符串

print(cleaned_str)


optim = torch.optim.Adam(model.parameters(), lr=5e-4)#ALISTA,AT,5e-2#LISTA-TOE-SOFT,5e-4
obj = torch.nn.MSELoss()
sigmoid = torch.nn.Sigmoid()
# 使用ReduceLROnPlateau调度器
scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=1000, verbose=True)
pos_loss = torch.nn.MSELoss()


t = trange(epochs)
#t = range(epochs)
for e in t:

    for i, data in enumerate(dataloader_training):

        idx = e * batches + i
        x, y = data
        
        # Split data to real and complex
        xr = x.to(torch.float32).to(device)
        xi = torch.zeros_like(xr).to(torch.float32).to(device)
        
        yr = y.real.to(torch.float32).to(device)
        yi = y.imag.to(torch.float32).to(device)
        
        # Send through model
        xpredr, xpredi = model(yr, yi)
        loss = obj(xpredr, xr) + obj(xpredi, xi)
        with torch.no_grad():
            training_losslist[idx] = loss

        loss.backward()    
        optim.step()
        optim.zero_grad()
        
        if testFreq and idx % testFreq == 0:
            for k, test_data in enumerate(dataloader_testing):
                
                x, y = test_data
                
                xr = x.to(torch.float32).to(device)
                xi = torch.zeros_like(xr).to(torch.float32).to(device)
                
                yr = y.real.to(torch.float32).to(device)
                yi = y.imag.to(torch.float32).to(device)
                
                xpredr, xpredi = model(yr, yi)

                testing_losslist[idx] += torch.mean((xpredr.cpu() - xr.cpu())**2) + torch.mean((xpredi.cpu() - xi.cpu())**2)
                scheduler.step(testing_losslist[idx])
                # testing_losslist[idx] += torch.mean((xpredr.cpu() - xr.cpu())**2) 
        t.set_description("Batch: {}/{}\t Training Loss: {}\t Validation Loss: {}".format(i, len(dataloader_training), training_losslist[idx], testing_losslist[idx]), refresh=True)

    # 打印当前学习率
        for param_group in optim.param_groups:
            print(f"Learning rate: {param_group['lr']}")

    if (e + 1) % 10 == 0:
        pd.DataFrame(
            {
                "train_loss": training_losslist,
                "test_loss": testing_losslist
            }
        ).to_csv(os.path.join(data_path + "/"+ cleaned_str +"_epoch_"+str(e)+"_log_250319"))

        print(training_losslist[-1])
        plt.semilogy(training_losslist[10:])
        plt.semilogy(testing_losslist[10:])
        torch.save(model.state_dict(), os.path.join(data_path + "/"+ cleaned_str +"_epoch_"+str(e)+"_8td_250319.pt"))