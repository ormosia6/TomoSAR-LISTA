
import numpy as np
from scipy import fft
from numpy import pi, fft

from scipy import io
import os
data_path =   os.path.join(os.getcwd(),'res/data_8td_randA_randpphi_snr_train_k1_2/')
if not os.path.exists(data_path):
    # 如果不存在，创建文件夹
    os.makedirs(data_path)
maxit = 10#迭代数
M = 12#通道数
N = 120#信号长度
zmin = 0
zmax = N
k = 1
sl = np.arange(zmin,zmax,k).reshape(1,int((zmax-zmin)/k))
lmbda = 0.4

sig = 0.4
epochs = 10
batchSize = 128
testFreq = 1

trainingPoints = 10*128
testingPoints = 5*128
sparsityLevel = 2#稀疏度

frequency = 10e9
c = 3e8
lambda_radar = c/frequency
#bn_k = 2*pi*bn/lambda_radar
bn_k = np.array([0,39.7791542655718,79.9721296892241,120.193632791642,155.478696841370,201.201788441832,201.201788441832,240.980942707404,281.173918131057,321.395421233475,356.680485189511,402.403576883665]).reshape(M,1)
bn = bn_k*lambda_radar/(2*np.pi)

Rmin  = 1.917723999023438e+03
rou_s = lambda_radar*Rmin/2/bn[-1]
detla_samb = lambda_radar*Rmin/2/(bn[1]-bn[0])


D = np.zeros((M,N),dtype=complex)

# 生成观测矩阵
D = np.exp(1j*2*bn_k*sl/Rmin)
# io.savemat(os.path.join(data_path,'D_complex.mat'),{'D':D}) 

# # Create ULA and Nested Array Matricies
M1 = M // 2
M2 = M - M1

D_real = np.real(D)
D_imag = np.imag(D)
original_D = np.vstack((np.hstack((D_real, -D_imag)), np.hstack((D_imag, D_real))))
np.save(os.path.join(data_path,'D.npy'),original_D) 
