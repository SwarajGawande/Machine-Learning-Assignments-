import numpy as np
from numpy.lib.function_base import gradient
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import sys
import time

arglist=sys.argv

part=arglist[1]
if part=='a' or part=='b':
    train_path = arglist[2]
    test_path = arglist[3]
    paramfile=arglist[4]
    outputfile=arglist[5]
    weightfile=arglist[6]
else:
    train_path = arglist[2]
    test_path = arglist[3]
    outputfile=arglist[4]
    weightfile=arglist[5]


train = pd.read_csv(train_path, index_col = 0)    
test = pd.read_csv(test_path, index_col = 0)
    
y_train = np.array(train['Length of Stay'])

train = train.drop(columns = ['Length of Stay'])

#Ensuring consistency of One-Hot Encoding


data = pd.concat([train, test], ignore_index = True)
cols = train.columns
cols = cols[:-1]
data = pd.get_dummies(data, columns=cols, drop_first=True)
data = data.to_numpy()
X_train = data[:train.shape[0], :]
ones=np.ones((len(X_train),1))
X_train=np.append(ones,X_train,axis=1)
X_test = data[train.shape[0]:, :]
ones=np.ones((len(X_test),1))
X_test=np.append(ones,X_test,axis=1)
W=np.zeros((len(X_train[0]),8))
y_train=pd.get_dummies(y_train,columns=['Length of Stay'])
y_train=np.array(y_train)

def L(X_train,Y_train,W):
    return abs(np.sum(np.log(np.clip(np.sum(Y_train*softmax(X_train@W,axis=1),axis=1),10**-(15),1-(10**(-15)))))/Y_train.shape[0])

if part=='a':
    parm=open(paramfile,'r')
    Lines=parm.readlines()
    mode=int(Lines[0])
    alpha=(Lines[1])
    it=int(Lines[2])
    parm.close()
    if mode==1:
        lr=float(alpha)
        for i in range(it):
            prediction=X_train@W
            prediction=softmax(prediction,axis=1)
            grad= (X_train.T @ (prediction - y_train)) / X_train.shape[0]
            W=W-(lr*grad)
    elif mode==2:
        lr0=float(alpha)
        for i in range(it):
            lr=lr0/np.sqrt(i+1)
            prediction=X_train@W
            prediction=softmax(prediction,axis=1)
            grad= (X_train.T @ (prediction - y_train)) / X_train.shape[0]
            W=W-(lr*grad)
    elif mode==3:
        lr0=float(alpha.split(',')[0])
        a=float(alpha.split(',')[1])
        b=float(alpha.split(',')[2])
        for i in range(it):
            lr=lr0
            Li=L(X_train,y_train,W)
            prediction=X_train@W
            prediction=softmax(prediction,axis=1)
            grad= (X_train.T @ (prediction - y_train)) / X_train.shape[0]
            while (L(X_train,y_train,W - lr*grad) > Li - a*lr*(np.linalg.norm(grad)**2)):
                lr = lr * b
            W=W-(lr*grad)


    Output=X_test@W
    Out=open(outputfile,'w')
    for a in Output:
        Out.write(str(np.argmax(a)+1)+'\n')
    Out.close()
    Wt=open(weightfile,'w')
    for l in W:
        for a in l:
            Wt.write(str(a)+'\n')
    Wt.close()

elif part=='b':
    parm=open(paramfile,'r')
    Lines=parm.readlines()
    mode=int(Lines[0])
    alpha=(Lines[1])
    it=int(Lines[2])
    bs=int(Lines[3])
    parm.close()
    length=len(X_train)
    b=int(length//bs)
    if mode==1:
        lr=float(alpha)
        for j in range(it):
            for i in range(b):
                X_trainb=X_train[i*bs:i*bs+bs]
                Y_trainb=y_train[i*bs:i*bs+bs]
                prediction=X_trainb@W
                prediction=softmax(prediction,axis=1)
                grad= (X_trainb.T @ (prediction - Y_trainb)) / X_trainb.shape[0]
                W=W-(lr*grad)
    elif mode==2:
        lr0=float(alpha)
        for j in range(it):
            lr=lr0/np.sqrt(j+1)
            for i in range(b):
                X_trainb=X_train[i*bs:i*bs+bs]
                Y_trainb=y_train[i*bs:i*bs+bs]
                prediction=X_trainb@W
                prediction=softmax(prediction,axis=1)
                grad= (X_trainb.T @ (prediction - Y_trainb)) / X_trainb.shape[0]
                W=W-(lr*grad)
    elif mode==3:
        lr0=float(alpha.split(',')[0])
        a=float(alpha.split(',')[1])
        bt=float(alpha.split(',')[2])
        n=1
        for i in range(it):
            lr=lr0
            Li=L(X_train,y_train,W)
            prediction=X_train@W
            prediction=softmax(prediction,axis=1)
            grad= (X_train.T @ (prediction - y_train)) / X_train.shape[0]
            while (L(X_train,y_train,W - lr*grad) > Li - a*lr*(np.linalg.norm(grad)**2)):
                lr = lr * bt
            for i in range(b):
                X_trainb=X_train[i*bs:i*bs+bs]
                Y_trainb=y_train[i*bs:i*bs+bs]
                prediction=X_trainb@W
                prediction=softmax(prediction,axis=1)
                grad= (X_trainb.T @ (prediction - Y_trainb)) / X_trainb.shape[0]
                W=W-(lr*grad)
            n+=1
    Output=X_test@W
    Out=open(outputfile,'w')
    for a in Output:
        Out.write(str(np.argmax(a)+1)+'\n')
    Out.close()
    Wt=open(weightfile,'w')
    for l in W:
        for a in l:
            Wt.write(str(a)+'\n')
    Wt.close()

elif part=='c':
    bs=80
    lr0=9
    length=len(X_train)
    b=int(length//bs)
    start=time.time()
    ls=start
    j=0
    m=10
    end=start+570
    print(start)
    print(end)
    while(time.time()<start+570):
        lr=lr0/np.sqrt(j+1)
        for i in range(b):
            X_trainb=X_train[i*bs:i*bs+bs]
            Y_trainb=y_train[i*bs:i*bs+bs]
            prediction=X_trainb@W
            prediction=softmax(prediction,axis=1)
            grad= (X_trainb.T @ (prediction - Y_trainb)) / X_trainb.shape[0]
            W=W-(lr*grad)
        loss=L(X_train,y_train,W)
        if loss<m:
            m=loss
            Wm=W
        j=j+1
        print(loss)
        if ls<time.time()-60:
            Output=X_test@Wm
            Out=open(outputfile,'w')
            for a in Output:
                Out.write(str(np.argmax(a)+1)+'\n')
            Out.close()
            Wt=open(weightfile,'w')
            for l in Wm:
                for a in l:
                    Wt.write(str(a)+'\n')
            Wt.close()
            print('saved with loss: '+str(m))
            ls=time.time()
    Output=X_test@Wm
    Out=open(outputfile,'w')
    for a in Output:
        Out.write(str(np.argmax(a)+1)+'\n')
    Out.close()
    Wt=open(weightfile,'w')
    for l in Wm:
        for a in l:
            Wt.write(str(a)+'\n')
    Wt.close()
    print('saved with loss: '+str(m))
    ls=time.time()
elif part=='d':
    bs=80
    lr0=9
    length=len(X_train)
    b=int(length//bs)
    start=time.time()
    ls=start
    j=0
    m=10
    kb= SelectKBest(chi2,k=500) 
    X_train=kb.fit_transform(X_train,y_train)
    X_test=kb.transform(X_test)
    W=np.zeros((500,8))
    while(time.time()<start+570):
        lr=lr0/np.sqrt(j+1)
        for i in range(b):
            X_trainb=X_train[i*bs:i*bs+bs]
            Y_trainb=y_train[i*bs:i*bs+bs]
            prediction=X_trainb@W
            prediction=softmax(prediction,axis=1)
            grad= (X_trainb.T @ (prediction - Y_trainb)) / X_trainb.shape[0]
            W=W-(lr*grad)
        loss=L(X_train,y_train,W)
        if loss<m:
            m=loss
            Wm=W
        j=j+1
        print(loss)
        if ls<time.time()-60:
            Output=X_test@Wm
            Out=open(outputfile,'w')
            for a in Output:
                Out.write(str(np.argmax(a)+1)+'\n')
            Out.close()
            Wt=open(weightfile,'w')
            for l in Wm:
                for a in l:
                    Wt.write(str(a)+'\n')
            Wt.close()
            print('saved with loss: '+str(m))
            ls=time.time()
    Output=X_test@Wm
    Out=open(outputfile,'w')
    for a in Output:
        Out.write(str(np.argmax(a)+1)+'\n')
    Out.close()
    Wt=open(weightfile,'w')
    for l in Wm:
        for a in l:
            Wt.write(str(a)+'\n')
    Wt.close()
    print('saved with loss: '+str(m))