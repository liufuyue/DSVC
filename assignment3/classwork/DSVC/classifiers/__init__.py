#-*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy import optimize
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 解决windows环境下画图汉字乱码问题


def logisticRegression_OneVsAll():
    data = loadmat_data("data_digits.mat") 
    X = data['X']
    y = data['y']
    m,n = X.shape
    num_labels = 10
    rand_indices = [t for t in [np.random.randint(x-x, m) for x in range(100)]]
    display_data(X[rand_indices,:])
    
    Lambda = 0.1
    all_theta = oneVsAll(X, y, num_labels, Lambda)
    p = predict_oneVsAll(all_theta,X)
    print(u"预测准确度为：%f%%"%np.mean(np.float64(p == y.reshape(-1,1))*100))
     
# 加载mat文件
def loadmat_data(fileName):
    return spio.loadmat(fileName)
    
# 显示100个数字
def display_data(imgData):
    sum = 0

    pad = 1
    display_array = -np.ones((pad+10*(20+pad),pad+10*(20+pad)))
    for i in range(10):
        for j in range(10):
            display_array[pad+i*(20+pad):pad+i*(20+pad)+20,pad+j*(20+pad):pad+j*(20+pad)+20] = (imgData[sum,:].reshape(20,20,order="F"))    # order=F指定以列优先，在matlab中是这样的，python中需要指定，默认以行
            sum += 1
            
    plt.imshow(display_array,cmap='gray')   #显示灰度图像
    plt.axis('off')
    plt.show()


def oneVsAll(X,y,num_labels,Lambda):
    # 初始化变量
    m,n = X.shape
    all_theta = np.zeros((n+1,num_labels))
    X = np.hstack((np.ones((m,1)),X))
    class_y = np.zeros((m,num_labels))
    initial_theta = np.zeros((n+1,1))
    
    # 映射y
    for i in range(num_labels):
        class_y[:,i] = np.int32(y==i).reshape(1,-1)
    for i in range(num_labels):
        #optimize.fmin_cg
        result = optimize.fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X,class_y[:,i],Lambda))
        all_theta[:,i] = result.reshape(1,-1)
    # 转置
    all_theta = np.transpose(all_theta)
    return all_theta

# 代价函数
def costFunction(initial_theta,X,y,inital_lambda):
    m = len(y)
    J = 0
    
    h = sigmoid(np.dot(X,initial_theta))
    theta1 = initial_theta.copy()
    theta1[0] = 0   
    
    temp = np.dot(np.transpose(theta1),theta1)
    J = (-np.dot(np.transpose(y),np.log(h))-np.dot(np.transpose(1-y),np.log(1-h))+temp*inital_lambda/2)/m
    return J

# 计算梯度
def gradient(initial_theta,X,y,inital_lambda):
    m = len(y)
    grad = np.zeros((initial_theta.shape[0]))
    
    h = sigmoid(np.dot(X,initial_theta))  # 计算h(z)
    theta1 = initial_theta.copy()
    theta1[0] = 0

    grad = np.dot(np.transpose(X),h-y)/m+inital_lambda/m*theta1
    return grad   

def sigmoid(z):
    h = np.zeros((len(z),1))
    
    h = 1.0/(1.0+np.exp(-z))
    return h

def predict_oneVsAll(all_theta,X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    p = np.zeros((m,1))
    X = np.hstack((np.ones((m,1)),X))
    
    h = sigmoid(np.dot(X,np.transpose(all_theta)))

    p = np.array(np.where(h[0,:] == np.max(h, axis=1)[0]))  
    for i in np.arange(1, m):
        t = np.array(np.where(h[i,:] == np.max(h, axis=1)[i]))
        p = np.vstack((p,t))
    return p
        
        
if __name__ == "__main__":
    logisticRegression_OneVsAll()

