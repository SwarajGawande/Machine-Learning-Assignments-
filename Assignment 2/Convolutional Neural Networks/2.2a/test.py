import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score
import sys

from IPython.display import Image
arglist=sys.argv
print('loaded')
class DevanagariDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform = None):
        
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,:-1].to_numpy()
            labels = data.iloc[:,-1].astype(int)
        else:
            images = data.iloc[:,:].to_numpy()
            labels = None
        
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = np.array(image).astype(np.uint8).reshape(32, 32, 1)
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
        sample = {"images": image, "labels": label}
        return sample

print('class created')

BATCH_SIZE = 200 
NUM_WORKERS = 20

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

# Test DataLoader
test_data = arglist[1] # Path to test csv file
test_dataset = DevanagariDataset(data_csv = test_data, train=False, img_transform=img_transforms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

class Network(Module):   
    def __init__(self):
        super(Network, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 32, kernel_size=3, stride=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(32, 64, kernel_size=3, stride=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 256, kernel_size=3, stride=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=1),
            Conv2d(256, 512, kernel_size=3, stride=1),
            ReLU(inplace=True)
        )
        
        self.linear_layers = Sequential(
            Linear(512,256),
            ReLU(inplace=True),
            Dropout(p=0.2),
            Linear(256, 46)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

torch.manual_seed(51)
# defining the model
model = Network()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.0001)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load(arglist[2]))

def test():
    model.eval()
    for batch_idx, sample in enumerate(test_loader):
        images = sample['images']
        with torch.no_grad():
            output = model(images.cuda())
        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)
        return predictions

predictions=test()
np.savetxt(arglist[3],predictions)
