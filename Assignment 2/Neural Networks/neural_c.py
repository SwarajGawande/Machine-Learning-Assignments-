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

def ForwardProp(X_train,params,layers,func,losst,momentum,opt):
    Z=[X_train]
    A=[X_train]
    for i in range(layers):
        Z=(A[i]@params[i])
        if opt==2:
            Z=A[i]@(params[i]-0.9*momentum[i])
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

def Loss(Y_pred,Y_train,losst):
    N=Y_train.shape[0]
    #Y_pred=np.zeros((N,1))
    #for l in y_pred:
    #    Y_pred=argmax(l)+1
    N=len(Y_train)
    if losst==0:
        loss=np.sum(np.sum(-1 * Y_train * np.log(np.clip(Y_pred, 10**(-15), 1-10**(-15))), axis=1))
    else:
        loss=np.sum((Y_pred-Y_train)*(Y_pred-Y_train))
    return loss/N

def BackProp(As,Y,params,layers,lr,func,losst,momentum,opt,var,epoch):
    delta=0
    if losst==0:
        delta=(As[-1]-Y)/Y.shape[0]
    else:
        delta=((As[-1]-Y)*derivative(func,As[-1]))/Y.shape[0]
    for i in range(1,layers+1):
        delta_ = (delta @params[layers-i].T) * derivative(func,As[layers-i])
        if opt==0:
            params[layers-i]-=lr*As[layers-i].T @ delta
        elif opt==1 or opt==2:
            momentum[layers-i]=(0.9*momentum[layers-i])+lr*(As[layers-i].T @ delta)
            params[layers-i]-=momentum[layers-i]
        elif opt==3:
            momentum[layers-i]=0.9*momentum[layers-i]+0.1*(As[layers-i].T @ delta)*(As[layers-i].T @ delta)
            params[layers-i]-=lr*(As[layers-i].T @ delta)/np.sqrt(momentum[layers-i]+10**(-15))
        elif opt==4:
            momentum[layers-i]=0.9*momentum[layers-i]+(As[layers-i].T @ delta)
            var[layers-i]=0.9*var[layers-i]+0.1*(As[layers-i].T @ delta)*(As[layers-i].T @ delta)
            m=momentum[layers-i]/(1-0.9**epoch)
            n=var[layers-i]/(1-0.9**epoch)
            params[layers-i]-=lr*m/(np.sqrt(n)+10**(-7))
        elif opt==5:
            momentum[layers-i]=0.9*momentum[layers-i]+(As[layers-i].T @ delta)
            var[layers-i]=0.9*var[layers-i]+0.1*(As[layers-i].T @ delta)*(As[layers-i].T @ delta)
            m=momentum[layers-i]/(1-0.9**epoch)
            n=var[layers-i]/(1-0.9**epoch)
            params[layers-i]-=lr*(0.9*m+((0.1*As[layers-i].T @ delta)/(1-0.9**epoch)))/(np.sqrt(n)+10**(-7))
        delta=np.delete(delta_,0,axis=1)

def Initialize(n,lis,s):
    params=[]
    np.random.seed(1)
    for i in range(n):
        W=np.random.normal(0,1,size=(s,lis[i]))*np.sqrt(2/(s+lis[i]))
        params.append(W)
        #print(W.shape)
        s=lis[i]+1
    return params


if __name__=='__main__':
    arglist=sys.argv
    trainfile=arglist[1]+'train_data_shuffled.csv'
    testfile=arglist[1]+'public_test.csv'
    ((X_train,Y_train),X_test)=Input(trainfile,testfile)
    #print(X_train.shape)
    paramfile='param.txt'
    pfile=open(paramfile,'r')
    Lines=pfile.readlines()
    pfile.close()
    X_train=X_train/255
    X_train=np.append(np.ones((len(X_train),1)),X_train,axis=1)
    length=len(X_train)
    bs=100
    b=int(length//bs)
    ll=Lines[0].split('[')[1].split(']')[0].split(',')
    layers=len(ll)
    params=Initialize(layers,ll,len(X_train[0]))
    losst=0
    func=2
    lrs=0
    opt=2
    lr=0.01
    Y=np.zeros((len(X_train),ll[-1]))
    for i in range(len(Y_train)):
        j=int(Y_train[i])
        Y[i][j]=1
    momentum=[0 for i in range(layers)]
    var=[0 for i in range(layers)]
    epoch=1
    for j in range(20):
        loss=0
        for i in range(b):
            X_trainb=X_train[i*bs:i*bs+bs]
            Y_trainb=Y_train[i*bs:i*bs+bs]
            Yb=Y[i*bs:i*bs+bs]
            A=ForwardProp(X_trainb,params,layers,func,losst,momentum,opt)
            BackProp(A,Yb,params,layers,lr,func,losst,momentum,opt,var,epoch)
            loss+=Loss(A[-1],Yb,losst)
            epoch+=1
        #print(j)
    #print(params)
    #print(np.load('essentials/part_a_and_b/multiclass_dataset/tc_1/ac_w_1.npy'))
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
