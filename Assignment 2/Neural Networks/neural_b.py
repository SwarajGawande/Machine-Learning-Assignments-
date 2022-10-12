import numpy as np
from numpy.core.fromnumeric import argmax
import sys

def Input(trainfile,testfile):
    train=np.loadtxt(trainfile,delimiter=',').astype('float32')
    Y_train=train[:,-1]
    X_train=train[:,0:-1]
    X_test=np.loadtxt(testfile,delimiter=',').astype('float32')
    return ((X_train,Y_train),X_test)

def softmax(x):
    x = np.exp(x)
    return x / np.sum(x, axis=1,keepdims=True)

def Activation(func,Z):
    if func==0:
        return 1/(1+np.exp(-1*Z))
    elif func==1:
        return np.tanh(Z)
    else:
        return Z.clip(0)

def derivative(func,A):
    if func==0:
        return A*(1-A)
    if func==1:
        return (1-A*A)
    else:
        return (A>0).astype('float32')

def ForwardProp(X_train,params,layers,func,losst):
    Z=[X_train]
    A=[X_train]
    for i in range(layers):
        Z=(A[i]@params[i])
        if i!=layers-1:
            Ac=Activation(func,Z)
            A.append(np.append(np.ones((len(Ac),1)),Ac,axis=1))
        else:
            Ac=Z
            if losst==0:
                Ac=softmax(Z)
            else:
                Ac=Activation(func,Z)
            A.append(Ac)
    return A

def Cost(Y_pred,Y_train):
    N=len(Y_train)
    cost=0
    for i in range(N):
        for j in range(len(Y_pred[0])):
            cost=cost-Y_train[i][j]*np.log(Y_pred[i][j])
    return cost/N

def BackProp(As,Y,params,layers,lr,func,losst):
    delta=0
    if losst==0:
        delta=(As[-1]-Y)/Y.shape[0]
    else:
        delta=((As[-1]-Y)*derivative(func,As[-1]))/Y.shape[0]
    for i in range(1,layers+1):
        delta_ = (delta @params[layers-i].T) * derivative(func,As[layers-i])
        params[layers-i]-=lr*As[layers-i].T @ delta
        delta=np.delete(delta_,0,axis=1)

def Initialize(n,lis,s,seed):
    params=[]
    np.random.seed(seed)
    for i in range(n):
        W=np.random.normal(0,1,size=(s,int(lis[i])))*np.sqrt(2/(s+int(lis[i])))
        params.append(W)
        s=int(lis[i])+1
    return params


if __name__=="__main__":
    arglist=sys.argv
    trainfile=arglist[1]+'train_data_shuffled.csv'
    testfile=arglist[1]+'public_test.csv'
    ((X_train,Y_train),X_test)=Input(trainfile,testfile)
    paramfile=arglist[3]+'param.txt'
    pfile=open(paramfile,'r')
    Lines=pfile.readlines()
    pfile.close()
    X_train=X_train/255
    X_test=X_test/255
    X_train=np.append(np.ones((len(X_train),1)),X_train,axis=1)
    length=len(X_train)
    bs=int(Lines[1])
    b=int(length//bs)
    ll=Lines[2].split('[')[1].split(']')[0].split(',')
    layers=len(ll)
    seed=int(Lines[7])
    params=Initialize(layers,ll,len(X_train[0]),seed)
    it=int(Lines[0])
    losst=int(Lines[6])
    func=int(Lines[5])
    lrs=int(Lines[3])
    lr=float(Lines[4])
    Y=np.zeros((len(X_train),int(ll[-1])))
    print('read')
    for i in range(len(Y_train)):
        j=int(Y_train[i])
        Y[i][j]=1
    for j in range(5):
        if lrs==1:
            lr=lr/np.sqrt(j+1)
        for i in range(b):
            X_trainb=X_train[i*bs:i*bs+bs]
            Yb=Y[i*bs:i*bs+bs]
            A=ForwardProp(X_trainb,params,layers,func,losst)
            BackProp(A,Yb,params,layers,lr,func,losst)
    for i in range(layers):
        np.save(arglist[2]+'w_'+str(i+1),params[i])
    P=ForwardProp(X_test,params,layers,func,losst)
    Predictions=np.zeros((len(P[-1]),1))
    #print(P[-1])
    k=0
    for l in P[-1]:
        Predictions[k]=argmax(l)
        k+=1
    np.save(arglist[2]+'predictions',Predictions)
