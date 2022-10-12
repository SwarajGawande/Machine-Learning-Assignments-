import numpy as np
import pandas as pd
import cv2
import os
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
import random
import argparse
from torchvision.transforms import transforms
transform = transforms.Compose([transforms.ToTensor()])
keypoint_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,num_keypoints=17)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
keypoint_model.to(device).eval()

argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--traininput', type=str, help='Input csv')
parser.add_argument('--trainoutput', type=str, help='Output Folder')

args = parser.parse_args()
traininput = args.traininput
trainoutput = args.trainoutput



class customdata(Dataset):
    def __init__(self,keypoints,labels,train=True):
        super().__init__()
        self.keypoints = keypoints
        self.labels = labels
        self.train = train


    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, index):
        
        if(self.train):
            y = self.labels[index]
        else:
            y = -1

        return (self.keypoints[index][:,:2].T,y)

class Archi(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(2,16,3,1,1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(16,64,3,1,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1088,272),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(272,num_classes),
            
        )
        


    def forward(self,x):
        y = self.model(x)
        return y


def train(model, device, train_loader, optimizer,criterion, epoch):
    model.train()
    train_loss = 0.
    for batch_idx, (X,y,) in enumerate(tqdm(train_loader)):
        X,y = X.to(device),y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y.long())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_loader)
    print('Training Loss at epoch {} : {}'.format(epoch,train_loss))

def test(model,device,test_loader):   
    model.eval()
    for batch_idx, (X,y,) in enumerate(tqdm(test_loader)):
        X,y = X.to(device),y.to(device)
        output = model(X)
        preds = np.argmax(output.detach().cpu().numpy(),axis=1)

    return preds

def formatPredictions(preds,invalid):
    pred_final=[]
    i=1
    for pred in preds:
        for p in pred:
            pred_final.append(p)
            #print(i)
            i+=1
    for  inv in invalid:
        pred_final.insert(inv,np.random.randint(19))
    return pred_final








data = pd.read_csv(traininput)

paths = data['name'].to_numpy()[:]
labels_train = data['category'].to_numpy()[:]
keypoints_train = []
valid_i_train = []
invalid_train = []
for i,path in enumerate(tqdm(paths)):
    image = cv2.imread(path)
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = keypoint_model(image)
    
    if(len(outputs[0]['keypoints'])>0):
        keypoints_train.append(outputs[0]['keypoints'][0].cpu().detach().numpy())
        valid_i_train.append(i)
    else:
        invalid_train.append(i)

np.save(os.path.join(trainoutput,'valid_keys_train'),keypoints_train)
np.save(os.path.join(trainoutput,'valid_i_train'),valid_i_train)
np.save(os.path.join(trainoutput,'invalid_train'),invalid_train)


keypoints_train = np.load(os.path.join(trainoutput,'valid_keys_train.npy'))

valid_i_train = np.load(os.path.join(trainoutput,'valid_i_train.npy'))

datatrain = pd.read_csv(traininput)
labels_train_all = datatrain['category'].to_numpy()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(labels_train_all)
labels_train = labels_train_all[valid_i_train]
labels_train = le.transform(labels_train)

train_data = customdata(keypoints_train,labels_train)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader = DataLoader(train_data,batch_size=64,shuffle=True,drop_last=True,num_workers=20)

model = Archi(len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=3.e-4,weight_decay=1.e-4)
criterion = nn.CrossEntropyLoss()
epochs = 50
for epoch in range(epochs):
    train(model,device,train_loader,optimizer,criterion,epoch)
np.save(os.path.join(trainoutput,'classes.npy'), le.classes_)
torch.save(model.state_dict(),os.path.join(trainoutput,'model.pth'))