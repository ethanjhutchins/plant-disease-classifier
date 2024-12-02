import os   #for working with files
from IPython.display import display   #for displaying data
import numpy as np      #for numerical computations
import pandas as pd     #for working with dataframes
import torch    #Pytorch module
import matplotlib
import matplotlib.pyplot as plt     #for plotting information on graphs and images using tensors
import torch.nn as nn   #for creating neural networks
from torch.utils.data import DataLoader     #for dataloaders
from PIL import Image   #for checking images
import torch.nn.functional as F     #for functions for calculating loss
import torchvision.transforms as transforms     #for transforming images into tensors
from torchvision.utils import make_grid     #for checking data
from torchvision.datasets import ImageFolder    #for working with classes and images
from torchsummary import summary    #for getting summary of the model

def explore_data():
    print(diseases)
    print("Total disease classes are: {}".format(len(diseases)))

    plants = []
    NumberOfDiseases = 0
    for plant in diseases:
        if plant.split('___')[0] not in plants:
            plants.append(plant.split('___')[0])
        if plant.split('___')[1] != 'healthy':
            NumberOfDiseases += 1

    #unique plants in dataset
    print(f"Unique Plants are: \n{plants}")
    #number of unique plants
    print("Number of plants: {}".format(len(plants)))
    #number of unique diseases
    print("Number of diseases: {}".format(NumberOfDiseases))

    #number of images for each disease
    nums = {}
    for disease in diseases:
        nums[disease] = len(os.listdir(train_dir + '/' + disease))

    #converting the nums directory to pandas datafrome passing index as 
    # plant name and nuumber of images as column
    img_per_class = pd.DataFrame(nums.values(), index=nums.keys(), columns=['no. of images'])
    display(img_per_class)

    n_train = 0
    for value in nums.values():
        n_train += value
    print(f"There are {n_train} images for training")

    #plotting number of images available for each disease
    index = [n for n in range(38)]
    plt.figure(figsize=(20, 5))
    plt.bar(index, height=[n for n in nums.values()], width=.3)
    plt.xlabel("Plant/Diseases", fontsize=10)
    plt.ylabel("No of images avalable", fontsize=10)
    plt.xticks(index, diseases, fontsize=5, rotation=90)
    plt.title("Images per each class of plant disease")

def show_image(img, lbl):
    print("Label: " + train.classes[lbl] + "(" + str(lbl) + ")")
    plt.imshow(img.permute(1, 2, 0))

def show_batch(data):
    for images, labels in data:
        fig, ax = plt.subplots(figsize=(30, 30))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break

#helper functions
#for moving data into GPU (if available)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

#for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

#for loading in the device (GPU if available else CPU)
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x);
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x  #relu can be applied before or after adding the input

#for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

#base class for the model
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  #generate predictions
        loss = F.cross_entropy(out, labels) #calculate loss 
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  #generate predictions
        loss = F.cross_entropy(out, labels) #calculate loss
        acc = accuracy(out, labels) #calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc }

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_accuracy).mean() #combine loss
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} #combine accuracies
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], las_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['lrs'][-1], result['train_loss'], result['val_loss']. result['val_accuracy']))

#Architecture for training
#convolution block with BatchNormalisation
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

#resnet architecture
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  #out dimensions: 128x64x64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True) #out dimensions:256x16x16
        self.conv4 = ConvBlock(256, 512, pool=True) #out dimensions:512x4x44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_diseases))

    def forward(self, xb):  #xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

matplotlib.use('TkAgg')

data_dir = "./plant dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
diseases = os.listdir(train_dir)

#datasets for validation and training
train = ImageFolder(train_dir, transform=transforms.ToTensor())
valid = ImageFolder(valid_dir, transform=transforms.ToTensor())

random_seed = 7
torch.manual_seed(random_seed)
batch_size = 32

#dataloaders for training and validation
train_dl = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_dl = DataLoader(valid, batch_size=batch_size, num_workers=8, pin_memory=True)

device = get_default_device()

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)

# model = to_device(ResNet9(3, len(train.classes)), device)
model = to_device(ResNet9(3, len(train.classes)), torch.device("cpu"))
print(model)

print(torch.cuda.is_available())

#getting summary of the model
INPUT_SHAPE = (3, 256, 256)
# print(summary(model.cuda(), (INPUT_SHAPE)))

# explore_data()
# show_image(*train[0])
# show_batch(train_dl)  #images for a training batch

plt.show()