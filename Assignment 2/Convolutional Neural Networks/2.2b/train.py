import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


import matplotlib.pyplot as plt # for plotting
import numpy as np
import pandas as pd
import glob
from scipy.special import softmax
from sklearn.metrics import accuracy_score
import sys
arglist=sys.argv

from IPython.display import Image

print('loaded')

# DataLoader Class
# if BATCH_SIZE = N, dataloader returns images tensor of size [N, C, H, W] and labels [N]
class ImageDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform=None):
        """
        Dataset init function
        
        INPUT:
        data_csv: Path to csv file containing [data, labels]
        train: 
            True: if the csv file has [labels,data] (Train data and Public Test Data) 
            False: if the csv file has only [data] and labels are not present.
        img_transform: List of preprocessing operations need to performed on image. 
        """
        
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,1:].to_numpy()
            labels = data.iloc[:,0].astype(int)
        else:
            images = data.iloc[:,:]
            labels = None
        
        self.images = images
        self.labels = labels
        print("Total Images: {}, Data Shape = {}".format(len(self.images), images.shape))
    def __len__(self):
        """Returns total number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Loads image of the given index and performs preprocessing.
        
        INPUT: 
        idx: index of the image to be loaded.
        
        OUTPUT:
        sample: dictionary with keys images (Tensor of shape [1,C,H,W]) and labels (Tensor of labels [1]).
        """
        image = self.images[idx]
        image = np.array(image).astype(np.uint8).reshape((32, 32, 3),order='F')
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
        
        sample = {"images": image, "labels": label}
        return sample

# Data Loader Usage

BATCH_SIZE = 200 # Batch Size. Adjust accordingly
NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

# Train DataLoader
train_data = arglist[1] # Path to train csv file
#train_data = "/mnt/scratch1/siy197580/COL341/cifar/train_data.csv"
#test_data = "/mnt/scratch1/siy197580/COL341/cifar/public_test.csv"
train_dataset = ImageDataset(data_csv = train_data, train=True, img_transform=img_transforms)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

# Test DataLoader
test_data = arglist[2] # Path to test csv file
#train_data = "/mnt/scratch1/siy197580/COL341/cifar/train_data.csv"
#test_data = "/mnt/scratch1/siy197580/COL341/cifar/public_test.csv"
test_dataset = ImageDataset(data_csv = test_data, train=True, img_transform=img_transforms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 32, kernel_size=3, stride=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 64, kernel_size=3, stride=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 512, kernel_size=3, stride=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(512, 1024, kernel_size=2, stride=1),
            ReLU(inplace=True)
        )
        
        self.linear_layers = Sequential(
            Linear(1024,256),
            ReLU(inplace=True),
            Dropout(p=0.2),
            Linear(256, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
torch.manual_seed(51)
# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.0001)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
train_losses=[]
val_losses=[]

def train(epoch,train_x,train_y):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    #x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        
    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    output_train = model(x_train)
    #output_val = model(x_val)
     # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    #loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    #val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    #val_loss = loss_val.item()
    return tr_loss
    
    
def calcAcc():
    model.eval()
    acc=0
    for batch_idx, sample in enumerate(test_loader):
        images = sample['images']
        labels = sample['labels']
        with torch.no_grad():
            output = model(images.cuda())
        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)
        acc+=accuracy_score(labels,predictions)
    return acc/len(test_loader)

lossf=open(arglist[4],'w')
accf=open(arglist[5],'w')
loss=[]
accuracy=[]

for i in range(5):
    tr_loss=0
    for batch_idx, sample in enumerate(train_loader):
        images = sample['images']
        labels = sample['labels']
        #print(batch_idx)
        tr_loss+=train(i,images,labels)
    acc=calcAcc()
    print('Epoch : ',i+1, '\t', 'loss :', tr_loss/len(train_loader),acc)
    lossf.write(str(tr_loss/len(train_loader))+'\n')
    accf.write(str(acc)+'\n')
    loss.append(tr_loss/len(train_loader))
    accuracy.append(acc)
lossf.close()
accf.close()
torch.save(model.state_dict(),arglist[3])
X=[1,2,3,4,5]

"""plt.plot(loss,X)
plt.ylabel('training loss')
plt.ylabel('epochs')
#plt.xticks(X,X)
plt.show()

plt.plot(accuracy,X)
plt.ylabel('testing accuracy')
plt.ylabel('epochs')
#plt.xticks(X,X)
plt.show()"""
