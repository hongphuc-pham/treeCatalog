from ray import tune
from functools import partial
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import torch, os, random
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

# To avoid non-essential warnings 
import warnings
warnings.filterwarnings('ignore')

from torchvision import datasets, transforms, models 
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split, Dataset
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt


from datetime import datetime, timezone
import pytz

# Timestamp tag for check point and best model
# ||||||||||||||||||      FILE NAME GENERATION      ||||||||||||||||||

def getCTimeStr():
    """ Get time string"""
    py_timezone = pytz.timezone('Australia/Adelaide')
    dt_string = datetime.now(py_timezone).strftime("(%H%M%S%b%d%Y)")
    # print("date and time =", dt_string)	
    return str(dt_string)


def cpNameGen( modelName = 'best_weights', fType = 'cp', eNo = None):
    """ 
    Generate the checkpoint or model file name with the timestamp
    - modelName (str): model name
    - fType (str): file type | either 'cp' - check point or 'mdl' - model
    - eNo (int): epoch No
    - dir (str): drirectoy
    """

    if eNo:
        return str(f'{modelName}_CP_e{eNo}{getCTimeStr()}.pth')

    return str(f'{modelName}_CP{getCTimeStr()}.pth')




# ||||||||||||||||||      DATA LOADER CLASS      ||||||||||||||||||
class LoadDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x = dataset[index][0]
        if self.transform:
            x = self.transform(dataset[index][0])
            
        y = dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(dataset)


# ||||||||||||||||||      BASELINE MODEL CLASS FOR TRAINING AND VALIDATION      ||||||||||||||||||
# ||||||||||||||||||         WITH EVALUATION METRIC FUNCTION - ACCURACY         ||||||||||||||||||

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-3 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    """
    with torch.no_grad():
        maxk = 3
        batch_size = target.size(0)

        # st()
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # st()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        # correct = (pred == target.view(1, -1).expand_as(pred))
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)



        correct_3 = correct[:3].reshape(-1).float().sum(0, keepdim=True)

        return correct_3.mul_(1.0 / batch_size)
#def accuracy(outputs, labels):
 #   _, preds = torch.max(outputs, dim=1)
  #  return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss, Hints: the loss function can be changed to improve the accuracy
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels, (5))           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


 # To check wether Google Colab GPU has been assigned/not. 
def get_default_device():

    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return None
    
def to_device(data, device):

    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

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


# ||||||||||||||||||      GOOGLE COLB GPU ASSIGNMENT CHECK      ||||||||||||||||||

def get_default_device():

    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return None
    
def to_device(data, device):

    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

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


# ||||||||||||||||||      FUNCTIONS FOR EVALUATION AND TRAINING     ||||||||||||||||||


@torch.no_grad()
def evaluate(model, val_loader):

    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, scheduler, opt_func=torch.optim.SGD, cpPath = None):

    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        
        # Check point saving
        checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'history': history
        # add any other information you want to save
        }

        torch.save(checkpoint, cpNameGen(cpPath, eNo=epoch))
        scheduler.step()

    return history

def continue_fit(epochs, lr, model, train_loader, val_loader, scheduler, optFunc=None, cpPath = None, startEpoch = 0):
    history = []
    if optFunc is None:
        optimizer = torch.optim.SGD(model.parameters(), lr)
    else:
        optimizer = optFunc

    for epoch in range(startEpoch, epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

        checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'history': history
        # add any other information you want to save
        }

        torch.save(checkpoint, cpNameGen(cpPath, eNo=epoch))
        scheduler.step()
        
    return history


def load_cp(cpPath, model, optimizer, scheduler=None):

    checkpoint = torch.load(cpPath)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict']) if scheduler is not None else None
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    history = checkpoint.get('history')

    return epoch, loss, history

# ||||||||||||||||||      FUNCTIONS FOR PLOTTING     ||||||||||||||||||

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()
    
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()


