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
parser.add_argument('--testinput', type=str, help='Input csv')
parser.add_argument('--testoutput', type=str, help='Output csv')
parser.add_argument('--modelpath', type=str, help='Model folder')
args = parser.parse_args()
testinput = args.testinput
testoutput = args.testoutput
modelpath = args.modelpath





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



data = pd.read_csv(testinput)

paths = data['name'].to_numpy()[:-1]
keypoints_test = []
valid_i_test = []
invalid_test = []
for i,path in enumerate(tqdm(paths)):
    image = cv2.imread('../input/col341-a3/'+path[1:])
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = keypoint_model(image)
    
    if(len(outputs[0]['keypoints'])>0):
        keypoints_test.append(outputs[0]['keypoints'][0].cpu().detach().numpy())
        valid_i_test.append(i)
    else:
        invalid_test.append(i)

np.save(os.path.join(modelpath,'valid_keys_test'),keypoints_test)
np.save(os.path.join(modelpath,'valid_i_test'),valid_i_test)
np.save(os.path.join(modelpath,'invalid_test'),invalid_test)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.classes_ = np.load(os.path.join(modelpath,'classes.npy'))

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Archi(len(le.classes_)).to(device)
model.load_state_dict(torch.load(os.path.join(modelpath,'model.pth')))
datatest = pd.read_csv(testinput,index_col=0)
valid_i_test = np.load(os.path.join(modelpath,'valid_i_test.npy'))
keypoints_test = np.load(os.path.join(modelpath,'valid_keys_test.npy'))
test_data = customdata(keypoints_test,None,train=False)
test_loader = DataLoader(test_data,batch_size=len(test_data),shuffle=False,num_workers=0)
import random
preds = test(model,device,test_loader)
actual_preds = le.inverse_transform(preds)
datatest['category'] = random.choice(le.classes_)
for i,j in enumerate(valid_i_test):
    datatest['category'][j] = actual_preds[i]

datatest.to_csv(testoutput)


