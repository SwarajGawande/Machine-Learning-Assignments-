import sys 
import numpy as np
arglist=sys.argv
def split10(X_train,Y_train,i,n):
    l=X_train.shape[0]
    d=l//n
    testx=X_train[i*d:i*d+d]
    testy=Y_train[i*d:i*d+d]
    trainx=np.append(X_train[:i*d],X_train[i*d+d:],axis=0)
    trainy=np.append(Y_train[:i*d],Y_train[i*d+d:],axis=0)
    return (trainx,trainy),(testx,testy)


def TargetEncoder(lis,xtrain,ytrain,dict):
    d1=[]
    n1=[]
    for i in range(len(xtrain[0])):
        d={}
        d1.append(d)
        n={}
        n1.append(n)
    for i in lis:
      d={}
      n={}
      for j in range(len(xtrain)):
          if xtrain[j][i] in d:
              d[xtrain[j][i]] += ytrain[j][0]
              n[xtrain[j][i]] += 1
          else:
              d[xtrain[j][i]] = ytrain[j][0]
              n[xtrain[j][i]] = 1
      for j in range(len(xtrain)):
          if not xtrain[j][i] in dict[i]:
              dict[i][xtrain[j][i]]=d[xtrain[j][i]]/n[xtrain[j][i]]
          xtrain[j][i]=(d[xtrain[j][i]]/n[xtrain[j][i]])
    return dict

def TransformTrain(X_train,Y_train):
    typ=X_train[:,19]
    #em=X_train[:,30]
    #gender=X_train[:,8]
    name=X_train[:,5]
    typ2=X_train[:,17]
    #X_train=np.delete(X_train,23,axis=1)
    X_train=np.delete(X_train,21,axis=1)
    X_train=np.delete(X_train,19,axis=1)
    X_train=np.delete(X_train,17,axis=1)
    X_train=np.delete(X_train,15,axis=1)
    #X_train=np.delete(X_train,11,axis=1)
    X_train=np.delete(X_train,7,axis=1)
    X_train=np.delete(X_train,5,axis=1)
    X_train=np.delete(X_train,3,axis=1)
    X_train=np.delete(X_train,1,axis=1)
    days=X_train[:,7]
    d=np.ones((len(X_train),1))
    n=np.ones((len(X_train),1))
    t=np.ones((len(X_train),1))
    t2=np.ones((len(X_train),1))
    for i in range(len(days)):
        d[i][0]=days[i]
        n[i][0]=name[i]
        t[i][0]=typ[i]
        t2[i][0]=typ2[i]
    X_train=np.append(X_train,d*d,axis=1)
    X_train=np.append(X_train,n*n,axis=1)
    X_train=np.append(X_train,t*t,axis=1)
    X_train=np.append(X_train,t2*t2,axis=1)
    X_train=np.append(X_train,d*t,axis=1)
    X_train=np.append(X_train,d*t2,axis=1)
    X_train=np.append(X_train,t*t2,axis=1)
    X_train=np.append(X_train,n*t2,axis=1)
    X_train=np.append(X_train,t*n,axis=1)
    X_train=np.append(X_train,d*n,axis=1)
    X_train=np.append(X_train,d+1000*t,axis=1)
    X_train=np.append(X_train,1000*d+n,axis=1)
    X_train=np.append(X_train,1000*t+t2,axis=1)
    X_train=np.append(X_train,1000*n+t2,axis=1)
    X_train=np.append(X_train,1000*n+t,axis=1)
    X_train=np.append(X_train,1000*d+t2,axis=1)
    #print(X_train.shape)
    lis=[]
    for i in range(len(X_train[0])):
        s={}
        lis.append(s)
    TargetEncoder([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38],X_train,Y_train,lis)
    #print(X_train)
    #pca=PCA(n_components=18)
    #X_train=pca.fit_transform(X_train)
    #poly = PolynomialFeatures(2)
    #X_train=poly.fit_transform(X_train)
    return X_train,lis

def TestEncoder(lis,xtrain,dict):
    for i in lis:
        for j in range(len(xtrain)):
            if xtrain[j][i] in dict[i]:
                xtrain[j][i]=dict[i][xtrain[j][i]]

def TransformTest(X_train,lis):
    typ=X_train[:,19]
    #em=X_train[:,30]
    #gender=X_train[:,8]
    name=X_train[:,5]
    typ2=X_train[:,17]
    #X_train=np.delete(X_train,23,axis=1)
    X_train=np.delete(X_train,21,axis=1)
    X_train=np.delete(X_train,19,axis=1)
    X_train=np.delete(X_train,17,axis=1)
    X_train=np.delete(X_train,15,axis=1)
    #X_train=np.delete(X_train,11,axis=1)
    X_train=np.delete(X_train,7,axis=1)
    X_train=np.delete(X_train,5,axis=1)
    X_train=np.delete(X_train,3,axis=1)
    X_train=np.delete(X_train,1,axis=1)
    days=X_train[:,7]
    d=np.ones((len(X_train),1))
    n=np.ones((len(X_train),1))
    t=np.ones((len(X_train),1))
    t2=np.ones((len(X_train),1))
    for i in range(len(days)):
        d[i][0]=days[i]
        n[i][0]=name[i]
        t[i][0]=typ[i]
        t2[i][0]=typ2[i]
    X_train=np.append(X_train,d*d,axis=1)
    X_train=np.append(X_train,n*n,axis=1)
    X_train=np.append(X_train,t*t,axis=1)
    X_train=np.append(X_train,t2*t2,axis=1)
    X_train=np.append(X_train,d*t,axis=1)
    X_train=np.append(X_train,d*t2,axis=1)
    X_train=np.append(X_train,t*t2,axis=1)
    X_train=np.append(X_train,n*t2,axis=1)
    X_train=np.append(X_train,t*n,axis=1)
    X_train=np.append(X_train,d*n,axis=1)
    X_train=np.append(X_train,d+1000*t,axis=1)
    X_train=np.append(X_train,1000*d+n,axis=1)
    X_train=np.append(X_train,1000*t+t2,axis=1)
    X_train=np.append(X_train,1000*n+t2,axis=1)
    X_train=np.append(X_train,1000*n+t,axis=1)
    X_train=np.append(X_train,1000*d+t2,axis=1)
    #print(X_train.shape)
    TestEncoder([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38],X_train,lis)
    #pca=PCA(n_components=18)
    #X_train=pca.fit_transform(X_train)
    #poly = PolynomialFeatures(2)
    #X_train=poly.fit_transform(X_train)
    return X_train

if arglist[1]=='a':
    trainfile=arglist[2]
    Inp=np.loadtxt(trainfile,delimiter=',',skiprows=1)
    X_train=Inp[:,1:-1].astype('float64')
    o=np.ones((len(Inp),1))
    X_train=np.append(o,X_train,axis=1)
    #print(X_train)
    Y=Inp[:,-1].astype('float64')
    Y_train=np.array([Y]).T
    #print(Y_train)
    Theta=np.linalg.inv(X_train.T@X_train) @ (X_train.T@Y_train)
    testfile=arglist[3]
    Inp=np.loadtxt(testfile,delimiter=',',skiprows=1)
    X_test=Inp[:,1:].astype('float64')
    o=np.ones((len(Inp),1))
    X_test=np.append(o,X_test,axis=1)
    #print(X_test.shape)
    weightfile=arglist[5]
    fout=open(weightfile,'w')
    #print(Theta)
    for t in Theta:
        fout.write(str(t[0])+'\n')
    fout.close()
    Output=X_test @ Theta
    predfile=arglist[4]
    fout=open(predfile,'w')
    for t in Output:
        fout.write(str(t[0])+'\n')
    fout.close()
elif arglist[1]=='b':
    trainfile=arglist[2]
    testfile=arglist[3]
    lamdafile=arglist[4]
    outputfile=arglist[5]
    weightfile=arglist[6]
    bestparameter=arglist[7]
    ld=np.loadtxt(lamdafile,delimiter='\n')
    Inp=np.loadtxt(trainfile,delimiter=',',skiprows=1)
    X_train=Inp[:,1:-1].astype('float64')
    o=np.ones((len(Inp),1))
    X_train=np.append(o,X_train,axis=1)
    #print(X_train)
    Y=Inp[:,-1].astype('float64')
    Y_train=np.array([Y]).T
    #print(Y_train)
    Inp=np.loadtxt(testfile,delimiter=',',skiprows=1)
    X_test=Inp[:,1:].astype('float64')
    o=np.ones((len(Inp),1))
    X_test=np.append(o,X_test,axis=1)
    lmda=0.01
    mx=1
    n=10
    for lmd in ld:
        s=0
        t=0
        for i in range(10):
            train,test=split10(X_train,Y_train,i,n)
            #print(train[0].shape)
            #print(test[0].shape)
            #print(X_train.shape)
            I=np.identity(train[0].shape[1],dtype=np.float64)
            Theta=np.linalg.inv(lmd*I+train[0].T@train[0]) @ (train[0].T@train[1])
            Output=test[0] @ Theta
            diff=Output-test[1]
            diff=np.multiply(diff,diff)
            y=np.multiply(test[1],test[1])
            s=s+sum(diff)/sum(y)
        #print('{:.12f}'.format(s[0]/10))
        if mx>s[0]/10:
            mx=s[0]/10
            lmda=lmd
    I=np.identity(X_train.shape[1],dtype=np.float64)
    Theta=np.linalg.inv(lmda*I+X_train.T@X_train) @ (X_train.T@Y_train)
    fout=open(weightfile,'w')
    #print(Theta)
    for t in Theta:
        fout.write(str(t[0])+'\n')
    fout.close()
    Output=X_test @ Theta
    fout=open(outputfile,'w')
    for t in Output:
        fout.write(str(t[0])+'\n')
    fout.close()
    fout=open(bestparameter,'w')
    fout.write(str(lmda)+'\n')
    fout.close()
elif arglist[1]=='c':
    trainfile=arglist[2]
    testfile=arglist[3]
    outputfile=arglist[4]
    Inp=np.loadtxt(trainfile,delimiter=',',skiprows=1)
    X_train=Inp[:,1:-1].astype('float64')
    o=np.ones((len(Inp),1))
    X_train=np.append(o,X_train,axis=1)
    #print(X_train)
    Y=Inp[:,-1].astype('float64')
    Y_train=np.array([Y]).T
    #print(Y_train)
    Inp=np.loadtxt(testfile,delimiter=',',skiprows=1)
    X_test=Inp[:,1:].astype('float64')
    o=np.ones((len(Inp),1))
    X_test=np.append(o,X_test,axis=1)
    X_train,dict=TransformTrain(X_train,Y_train)
    X_test=TransformTest(X_test,dict)
    I=np.identity(X_train.shape[1],dtype=np.float64)
    Theta=np.linalg.pinv(0.01*I+X_train.T@X_train) @ (X_train.T@Y_train)
    Output=X_test @ Theta
    fout=open(outputfile,'w')
    for t in Output:
        fout.write(str(t[0])+'\n')
    fout.close()
