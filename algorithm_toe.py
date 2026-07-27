##algorithm
import time
import importlib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from scipy.linalg import norm, pinv, toeplitz
from scipy.signal import find_peaks
from scipy import fft
from tqdm import tqdm, trange
import cvxpy as cp
from numpy import pi, fft
import torch
import math
from torch.nn.functional import relu, mse_loss, conv1d
from torch.nn import Module, Parameter, ReLU
from optimize_matrices import generalized_coherence
import torch.nn as nn
torch.cuda.empty_cache()
# try to use gpu
# if torch.cuda.is_available():
#     device = torch.device('cuda:1')  # 指定使用第1张GPU
# else:
device = "cpu"

class sparse_dataset(Dataset):
    def __init__(self, N, K, Nexamples, snr=0.0, A=None):
        self.X = np.zeros((Nexamples, N, 1))
        for ii in trange(Nexamples):
            self.X[ii,...] = self.generate_sparse_vector(N, K)
        #self.X *= np.random.randn(*self.X.shape)
        self.X = torch.from_numpy(self.X).reshape(Nexamples, N)
        self.Y = self.X @ A.T
        t_snr = 10 ** (snr / 10.0)
        power_signal = torch.mean(abs(self.Y**2))
        npower = power_signal / t_snr
        self.N = np.sqrt(npower) * torch.randn_like(self.Y)
        power_noise = torch.mean(abs(self.N**2)) 
        SNR = 10 * torch.log10(power_signal / power_noise)
        print("SNR: ", SNR.item())  # 打印 SNR
        self.Y = self.Y + self.N
    def __getitem__(self, i):
        return (self.X[i, ...], self.Y[i, ...])
    
    def __len__(self, ):
        return self.X.shape[0]
    
    def generate_sparse_vector(self, N, K):
        x = np.zeros((N,1))
        RCS = np.random.uniform(1,4,1)
        RCS_k = np.array([RCS, RCS*0.9, RCS*0.8, RCS*0.7]) #随机生成4个RCS
        # x[:K,...] = 1
        # phi = np.random.uniform(0, 2 * np.pi, 1)
        # RCS_k = RCS_k * np.exp(1j * phi)
        x[:K,...] = RCS_k[:K,...]
        np.random.shuffle(x)
        return x

    def regenerate_measurements(self, sig=0.0, A=None):
        # self.Y = A @ self.X# @ A.T
        self.Y = self.X @ A.T
        power_signal = torch.mean(self.Y**2)
        self.N = np.sqrt(sig) * torch.randn_like(self.Y)
        self.Y = self.Y + self.N        
        power_noise = torch.mean(self.N**2)
        # snr = 10*log10(power_signal/power_noise)
        # print("SNR: ",snr)
    def getmeasurements(self, i):
        return self
    def getnoise(self, i):
        return self.N[i, ...] 

class sparse_dataset_double(Dataset):
    def __init__(self, N, K, Nexamples, interval, snr=0.0, A=None,):
        self.X = np.zeros((Nexamples, N, 1))
        for ii in trange(Nexamples):
            self.X[ii,...] = self.generate_sparse_vector(N, K,interval)
        self.X = torch.from_numpy(self.X).reshape(Nexamples, N)
        self.Y = self.X @ A.T
        t_snr = 10 ** (snr / 10.0)
        power_signal = torch.mean(abs(self.Y**2))
        npower = power_signal / t_snr
        self.N = np.sqrt(npower) * torch.randn_like(self.Y)
        power_noise = torch.mean(abs(self.N**2)) 
        SNR = 10 * torch.log10(power_signal / power_noise)
        print("SNR: ", SNR.item())  # 打印 SNR
        self.Y = self.Y + self.N
    def __getitem__(self, i):
        return (self.X[i, ...], self.Y[i, ...])
    
    def __len__(self, ):
        return self.X.shape[0]
    

    def generate_sparse_vector(self, N, K,interval):
        x = np.zeros((N,1))
        RCS = np.random.uniform(1,2,1)
        RCS_k = np.array([RCS, RCS*0.9, RCS*0.8, RCS*0.7]) #随机生成4个RCS
        # x[:K,...] = 1
        # phi = np.random.uniform(0, 2 * np.pi, 1)
        # RCS_k = RCS_k * np.exp(1j * phi)

        start_pos = np.random.randint(0, N-interval-1)

        # 在固定间隔的位置上放置 RCS_k 的值
        for i in range(K):
            x[start_pos + i * interval] = RCS_k[i]
        return x

    def regenerate_measurements(self, sig=0.0, A=None):
        # self.Y = A @ self.X# @ A.T
        self.Y = self.X @ A.T
        power_signal = torch.mean(self.Y**2)
        self.N = np.sqrt(sig) * torch.randn_like(self.Y)
        self.Y = self.Y + self.N        
        power_noise = torch.mean(self.N**2)
        # snr = 10*log10(power_signal/power_noise)
        # print("SNR: ",snr)
    def getmeasurements(self, i):
        return self
    def getnoise(self, i):
        return self.N[i, ...] 


class LISTA(nn.Module):
    def __init__(self, M, N, maxit, phi,lmbda):
        super(LISTA, self).__init__()
        # self.m = m
        # self.n = n
        # self.k = k # 迭代次数
        # self.phi  = phi # dictionary
        self.lmbda = lmbda

        self.M = M*2
        self.N = N*2
        self.maxit = maxit
        self.phi = torch.Tensor(phi).to(device)
        # self.W = torch.Tensor(W).to(device)

        # generate the network parameters
        self._W = nn.Linear(in_features=self.N, out_features=self.M, bias=False)
        self._S = nn.Linear(in_features=self.M, out_features=self.M, bias=False)
        # self.phi = torch.Tensor(phi).to(device) # 方法调用 用于将张量移动到指定的设备上
        self.L = np.max(np.linalg.eigvals(np.dot(phi, phi.T)).astype(np.float32)) # 矩阵特征值最大值

        # weight initial
        L = self.L
        S = torch.from_numpy(np.eye(phi.shape[1]) - (1/L)*np.matmul(phi.T, phi))
        S = S.float().to(device)
        W = torch.from_numpy((1/L)*phi.T)
        W = W.float().to(device)
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(W)
        # self.shrinkage = nn.Softshrink(self.lmbda / self.L) # a is learning rate theta=a/L equals the threshold
        self.shrinkage = nn.Softshrink(0.01 / self.L) 

    def forward(self, yr, yi):
        y = torch.cat([yr, yi], dim=1)
        x = self.shrinkage(self._W(y))
        if self.maxit == 1 :
            return x
        for i in range(self.maxit):
            x = self.shrinkage(self._W(y) + self._S(x))
        # x = x.T
        [BB,len_x] = x.shape
        xr = x[:, :int(len_x/2)]  # 1-200 作为实部
        xi = x[:, int(len_x/2):]  # 201-400 作为虚部
        return xr, xi




class Toe_LISTA(Module):
    
    def __init__(self, M, N, maxit):
        super(Toe_LISTA, self).__init__()
    
        # Real and imaginary Wg and We matricies 
        self.Wre = Parameter(torch.zeros([maxit+1, N, M]), requires_grad=True)
        self.Wie = Parameter(torch.zeros([maxit+1, N, M]), requires_grad=True)
        self.hrg = Parameter(torch.zeros([maxit+1, 1, 1, N]), requires_grad=True) # dimension such that it works with conv1d
        self.hig = Parameter(torch.zeros([maxit+1, 1, 1, N]), requires_grad=True)
        # self.hrg = Parameter(torch.zeros([maxit+1, 1, 1, 2*N-1]), requires_grad=True) # dimension such that it works with conv1d
        # self.hig = Parameter(torch.zeros([maxit+1, 1, 1, 2*N-1]), requires_grad=True)
        # alpha and lambda hyper-parameters to LASSO/ISTA
        self.theta = Parameter(torch.ones(maxit+1), requires_grad=True)
        
        # Save the passed values
        self.M = M
        self.N = N
        self.maxit = maxit
        self.relu = ReLU()

        return
    
    def forward(self, yr, yi, epsilon=1e-10):
        
        Wret = torch.transpose(self.Wre[0], 0, 1)
        Wiet = torch.transpose(self.Wie[0], 0, 1)
                
        # Apply We branch to y to 0-th iteration
        zr = torch.matmul(yr, Wret) - torch.matmul(yi, Wiet)
        zi = torch.matmul(yi, Wret) + torch.matmul(yr, Wiet)
        
        # Apply soft-thresholding according to Eldar's paper.
        xabs = torch.sqrt(torch.square(zr) + torch.square(zi) + epsilon)
        xr = torch.divide(zr, xabs + epsilon) * self.relu(xabs - self.theta[0])
        xi = torch.divide(zi, xabs + epsilon) * self.relu(xabs - self.theta[0])
        
        for t in range(1, self.maxit+1):

            Wret = torch.transpose(self.Wre[t], 0, 1)
            Wiet = torch.transpose(self.Wie[t], 0, 1)
            hrgt = self.hrg[t]
            higt = self.hig[t]
        
            # Apply We branch to y to t-th iteration
            ar = torch.matmul(yr, Wret) - torch.matmul(yi, Wiet)
            ai = torch.matmul(yi, Wret) + torch.matmul(yr, Wiet)
            
            # Apply hg conv1d branch to x^(t) for t-th iteration
            br = conv1d(xr.unsqueeze(1), hrgt, padding='same') - conv1d(xi.unsqueeze(1), higt, padding='same')
            bi = conv1d(xi.unsqueeze(1), hrgt, padding='same') + conv1d(xr.unsqueeze(1), higt, padding='same')
            
            # Add the two branches                                                                           
            zr = ar + br.squeeze(1)
            zi = ai + bi.squeeze(1)
            
            # Apply soft-thresholding
            xabs = torch.sqrt(torch.square(zr) + torch.square(zi) + epsilon)
            # xr = torch.divide(zr, xabs + epsilon) * self.relu(xabs - self.theta[t])
            # xi = torch.divide(zi, xabs + epsilon) * self.relu(xabs - self.theta[t])
            xr = torch.divide(zr, xabs + epsilon) * self.relu(xabs - self.theta[t])
            xi = torch.divide(zi, xabs + epsilon) * self.relu(xabs - self.theta[t])
      
        return xr, xi

# In[5]:

def soft_threshold_toe(x, xabs,theta, p):
    x = x.T
    xabs = xabs.T
    if p == 0:
        return (torch.sign(x) * torch.relu(torch.abs(x) - theta)).T

    abs_ = xabs
    topk, _ = torch.topk(abs_, int(p), dim=0)
    topk, _ = topk.min(dim=0)
    index = (abs_ > topk).float()
    return (index * x + (1 - index) * torch.sign(x) * torch.relu(torch.abs(x) - theta)).T


class Toe_LISTA_Ada(Module):
    
    def __init__(self, M, N, maxit,p):
        super(Toe_LISTA_Ada, self).__init__()
    
        # Real and imaginary Wg and We matricies 
        self.Wre = Parameter(torch.zeros([maxit+1, N, M]), requires_grad=True)
        self.Wie = Parameter(torch.zeros([maxit+1, N, M]), requires_grad=True)
        self.hrg = Parameter(torch.zeros([maxit+1, 1, 1, N]), requires_grad=True) # dimension such that it works with conv1d
        self.hig = Parameter(torch.zeros([maxit+1, 1, 1, N]), requires_grad=True)
        # self.hrg = Parameter(torch.zeros([maxit+1, 1, 1, 2*N-1]), requires_grad=True) # dimension such that it works with conv1d
        # self.hig = Parameter(torch.zeros([maxit+1, 1, 1, 2*N-1]), requires_grad=True)
        # alpha and lambda hyper-parameters to LASSO/ISTA
        self.theta = Parameter(torch.ones(maxit+1), requires_grad=True)
        # self.thet_a = nn.ParameterList([nn.Parameter(torch.ones(1) * 0.5) for i in range(maxit)])
        self.p = p
        # Save the passed values
        self.M = M
        self.N = N
        self.maxit = maxit
        self.relu = ReLU()

        return
    
    
    def forward(self, yr, yi, epsilon=1e-10):
        
        Wret = torch.transpose(self.Wre[0], 0, 1)
        Wiet = torch.transpose(self.Wie[0], 0, 1)
                
        # Apply We branch to y to 0-th iteration
        zr = torch.matmul(yr, Wret) - torch.matmul(yi, Wiet)
        zi = torch.matmul(yi, Wret) + torch.matmul(yr, Wiet)
        
        # Apply soft-thresholding according to Eldar's paper.
        xabs = torch.sqrt(torch.square(zr) + torch.square(zi) + epsilon)
        xr = soft_threshold_toe(zr , xabs, self.theta[0], self.p[0])
        xi = soft_threshold_toe(zi , xabs, self.theta[0], self.p[0])
        
        for t in range(1, self.maxit):

            Wret = torch.transpose(self.Wre[t], 0, 1)
            Wiet = torch.transpose(self.Wie[t], 0, 1)
            hrgt = self.hrg[t]
            higt = self.hig[t]
        
            # Apply We branch to y to t-th iteration
            ar = torch.matmul(yr, Wret) - torch.matmul(yi, Wiet)
            ai = torch.matmul(yi, Wret) + torch.matmul(yr, Wiet)
            
            # Apply hg conv1d branch to x^(t) for t-th iteration
            br = conv1d(xr.unsqueeze(1), hrgt, padding='same') - conv1d(xi.unsqueeze(1), higt, padding='same')
            bi = conv1d(xi.unsqueeze(1), hrgt, padding='same') + conv1d(xr.unsqueeze(1), higt, padding='same')
            
            # Add the two branches                                                                           
            zr = ar + br.squeeze(1)
            zi = ai + bi.squeeze(1)
            
            # Apply soft-thresholding
            xabs = 2*torch.sqrt(torch.square(zr) + torch.square(zi) + epsilon)
            xr = soft_threshold_toe(zr , xabs, self.theta[t], self.p[t])
            xi = soft_threshold_toe(zi , xabs, self.theta[t], self.p[t])
      
        return xr, xi

class ComplexLISTA_Weights_Nested(Module):
    
    def __init__(self, M1, M2, N, maxit):
        super(ComplexLISTA_Weights_Nested, self).__init__()
    
        # Real and imaginary Wg and We matricies 
        self.M = M1 + M2
        self.Wre = Parameter(torch.zeros([maxit+1, N, self.M]), requires_grad=True)
        self.Wie = Parameter(torch.zeros([maxit+1, N, self.M]), requires_grad=True)
        self.win = Parameter(torch.zeros([maxit+1, M1]), requires_grad=True) # dimension such that it works with conv1d
        self.wout = Parameter(torch.zeros([maxit+1, M2]), requires_grad=True) # dimension such that it works with conv1d

        
        # alpha and lambda hyper-parameters to LASSO/ISTA
        self.theta = Parameter(torch.ones(maxit+1), requires_grad=True)
        
        # Save the passed values
        self.M1 = M1
        self.M2 = M2
        self.N = N
        self.maxit = maxit

        # Create useful relu layer
        self.relu = ReLU()

        # Assuming the measurement model
        self.complex_exp = lambda x : np.exp(2j*pi*x)
        self.fgrid = fft.fftfreq(N)
        
        self.inner = np.arange(M1)
        self.outer = np.arange(1, M2+1)*(M1)

        self.argin = np.outer(self.inner, 2*pi*self.fgrid)
        self.argout = np.outer(self.outer, 2*pi*self.fgrid)

        # Predefine useful matricies
        self.Cin = torch.from_numpy(np.cos(self.argin)).to(torch.float32).unsqueeze(0).unsqueeze(0)
        self.Sin = torch.from_numpy(np.sin(self.argin)).to(torch.float32).unsqueeze(0).unsqueeze(0)
        self.Cin = Parameter(self.Cin, requires_grad=False)
        self.Sin = Parameter(self.Sin, requires_grad=False)
        
        self.Cout = torch.from_numpy(np.cos(self.argout)).to(torch.float32).unsqueeze(0).unsqueeze(0)
        self.Sout = torch.from_numpy(np.sin(self.argout)).to(torch.float32).unsqueeze(0).unsqueeze(0)
        self.Cout = Parameter(self.Cout, requires_grad=False)
        self.Sout = Parameter(self.Sout, requires_grad=False)

        return
    
    def forward(self, yr, yi, epsilon=1e-10):
        
        Wret = torch.transpose(self.Wre[0], 0, 1)
        Wiet = torch.transpose(self.Wie[0], 0, 1)
                
        # Apply We branch to y to 0-th iteration
        zr = torch.matmul(yr, Wret) - torch.matmul(yi, Wiet)
        zi = torch.matmul(yi, Wret) + torch.matmul(yr, Wiet)
        
        # Apply soft-thresholding according to Eldar's paper.
        xabs = torch.sqrt(torch.square(zr) + torch.square(zi) + epsilon)
        xr = torch.divide(zr, xabs + epsilon) * self.relu(xabs - self.theta[0])
        xi = torch.divide(zi, xabs + epsilon) * self.relu(xabs - self.theta[0])
        
        for t in range(1, self.maxit+1):

            Wret = torch.transpose(self.Wre[t], 0, 1)
            Wiet = torch.transpose(self.Wie[t], 0, 1)
        
            # Apply We branch to y to t-th iteration
            ar = torch.matmul(yr, Wret) - torch.matmul(yi, Wiet)
            ai = torch.matmul(yi, Wret) + torch.matmul(yr, Wiet)
            
            # Apply hg conv1d branch to x^(t) for t-th iteration
            hrgt = torch.matmul(self.win[t], self.Cin) + torch.matmul(self.wout[t], self.Cout)
            higt = torch.matmul(self.win[t], self.Sin) + torch.matmul(self.wout[t], self.Sout)

            br = conv1d(xr.unsqueeze(1), hrgt, padding='same') - conv1d(xi.unsqueeze(1), higt, padding='same')
            bi = conv1d(xi.unsqueeze(1), hrgt, padding='same') + conv1d(xr.unsqueeze(1), higt, padding='same')

            # Add the two branches                                                                           
            zr = ar + br.squeeze(1)
            zi = ai + bi.squeeze(1)
            
            # Apply soft-thresholding
            xabs = torch.sqrt(torch.square(zr) + torch.square(zi) + epsilon)
            xr = torch.divide(zr, xabs + epsilon) * self.relu(xabs - self.theta[t])
            xi = torch.divide(zi, xabs + epsilon) * self.relu(xabs - self.theta[t])
      
        return xr, xi


# In[16]:


class ComplexLISTA_Weights_Random(Module):
    
    def __init__(self, M, N, maxit):
        super(ComplexLISTA_Weights_Random, self).__init__()
    
        # Real and imaginary Wg and We matricies 
        self.Wre = Parameter(torch.zeros([maxit+1, N, M]), requires_grad=True)
        self.Wie = Parameter(torch.zeros([maxit+1, N, M]), requires_grad=True)
        self.Wg = Parameter(torch.zeros([maxit+1, M]), requires_grad=True) # dimension such that it works with conv1d
        
        # alpha and lambda hyper-parameters to LASSO/ISTA
        self.theta = Parameter(torch.ones(maxit+1), requires_grad=True)
        
        # Save the passed values
        self.M = M
        self.N = N
        self.maxit = maxit

        # Create useful relu layer
        self.relu = ReLU()

        # Assuming the measurement model
        self.complex_exp = lambda x : np.exp(2j*pi*x)
        self.fgrid = fft.fftfreq(N)
        self.ula = np.arange(M)
        self.arg = np.outer(self.ula, 2*pi*self.fgrid)

        # Predefine useful matricies
        self.C = torch.from_numpy(np.cos(self.arg)).to(torch.float32).unsqueeze(0).unsqueeze(0)
        self.S = torch.from_numpy(np.sin(self.arg)).to(torch.float32).unsqueeze(0).unsqueeze(0)

        # Do the matricies really matter?
        self.C = torch.randn_like(self.C)
        self.S = torch.randn_like(self.S)

        self.C = Parameter(self.C, requires_grad=False)
        self.S = Parameter(self.S, requires_grad=False)

        return
    
    def forward(self, yr, yi, epsilon=1e-10):
        
        Wret = torch.transpose(self.Wre[0], 0, 1)
        Wiet = torch.transpose(self.Wie[0], 0, 1)
                
        # Apply We branch to y to 0-th iteration
        zr = torch.matmul(yr, Wret) - torch.matmul(yi, Wiet)
        zi = torch.matmul(yi, Wret) + torch.matmul(yr, Wiet)
        
        # Apply soft-thresholding according to Eldar's paper.
        xabs = torch.sqrt(torch.square(zr) + torch.square(zi) + epsilon)
        xr = torch.divide(zr, xabs + epsilon) * self.relu(xabs - self.theta[0])
        xi = torch.divide(zi, xabs + epsilon) * self.relu(xabs - self.theta[0])
        
        for t in range(1, self.maxit+1):

            Wret = torch.transpose(self.Wre[t], 0, 1)
            Wiet = torch.transpose(self.Wie[t], 0, 1)
        
            # Apply We branch to y to t-th iteration
            ar = torch.matmul(yr, Wret) - torch.matmul(yi, Wiet)
            ai = torch.matmul(yi, Wret) + torch.matmul(yr, Wiet)
            
            # Apply hg conv1d branch to x^(t) for t-th iteration
            hrgt = torch.matmul(self.Wg[t], self.C)
            higt = torch.matmul(self.Wg[t], self.S)

            br = conv1d(xr.unsqueeze(1), hrgt, padding='same') - conv1d(xi.unsqueeze(1), higt, padding='same')
            bi = conv1d(xi.unsqueeze(1), hrgt, padding='same') + conv1d(xr.unsqueeze(1), higt, padding='same')

            # Add the two branches                                                                           
            zr = ar + br.squeeze(1)
            zi = ai + bi.squeeze(1)
            
            # Apply soft-thresholding
            xabs = torch.sqrt(torch.square(zr) + torch.square(zi) + epsilon)
            xr = torch.divide(zr, xabs + epsilon) * self.relu(xabs - self.theta[t])
            xi = torch.divide(zi, xabs + epsilon) * self.relu(xabs - self.theta[t])
      
        return xr, xi


# normal l1 methods

def minL1_CVX(y, A, sig):
    M, N = A.shape
    x = cp.Variable(N)
    cost = cp.norm(x, 1)
    constraint = [cp.norm(A @ x - y.flatten(), 2) <= 1.01*sig]
    # constraint = [A @ x == y]#[:,0]
    prob = cp.Problem(cp.Minimize(cost), constraint)
    prob.solve()
    S_l1 = np.abs(x.value)
    return S_l1
#OMP算法函数
def cs_omp(y,D):    
    L=math.floor(3*(y.shape[0])/4)
    residual=y  #初始化残差
    index=np.zeros((L),dtype=int)
    for i in range(L):
        index[i]= -1
    result=np.zeros((256))
    for j in range(L):  #迭代次数
        product=np.abs(np.dot(D.T,residual))
        pos=np.argmax(product)  #最大投影系数对应的位置        
        index[j]=pos
        my=np.linalg.pinv(D[:,index>=0]) #最小二乘,看参考文献1           
        a=np.dot(my,y) #最小二乘,看参考文献1     
        residual=y-np.dot(D[:,index>=0],a)
    result[index>=0]=a
    return  result

def Omp(y,A,K,N):
    cols=A.shape[1]#传感矩阵A的列数 800
    res=y #初始化残差r0 值为y
    indexs=[]#用来保存索引的数组
    A_c=A.copy()#传感矩阵A的拷贝
    
    
    #进行K次迭代
    for i in range(0,K):
        products=[]#用来保存每次迭代产生的内积
        #对于传感矩阵A中每一列进行计算
        for col in range(cols):
            #传感矩阵A第col列与残差的内积    (32,).T*初始残差y(32,)
            products.append(np.dot(A[:,col].T,res))#获得一个内积 放入products数组

        # plt.rcParams.update({'font.size': 16})
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.plot(sl.T, np.abs(products), linewidth=2, label='omp')
        # ax.stem(sl.T, S_test, 'k', markerfmt='ko', linefmt='k--', label='True',  basefmt=" ")
        # y1, y2 = ax.get_ylim()
        # ax.set_ylim(0.001, y2)
        # ax.set_xlabel('Frequency')
        # ax.set_title('')
        # ax.grid(color='#99AABB', linestyle=':')
        # ax.set_facecolor('#CCDDEE')
        # ax.legend()
        # plt.tight_layout()
        # plt.show()
        # plt.savefig(model_name + str(number) +'prouduct.png') 
        
        #一轮迭代products中有800个值
        index=np.argmax(np.abs(products)) # 每列计算完成后 在products找最大内积并返回列索引值    
        
        indexs.append(index)#将最大列索引值加入索引数组indexs[]
        #使用索引集在传感矩阵中获得子集
        inv=np.dot(A_c[:,indexs].T,A_c[:,indexs])#
        
        theta=np.dot(np.dot(np.linalg.inv(inv),A_c[:,indexs].T),y)#利用最小二乘估计 计算一次θ
        print(theta.shape)
        res=y-np.dot(A_c[:,indexs],theta)#更新残差
    
    #迭代8次出来的theta的形状为(8,0)
    
    theta_final=np.zeros(N,)#重建theta 利用对应的索引
    theta_final[indexs]=theta
    return theta_final , indexs