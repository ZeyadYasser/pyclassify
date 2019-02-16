import os
import time
import pickle

import torch
import torchvision.datasets as datasets

from torchvision import transforms
from PIL import Image
from pyclassify.models import get_backend

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def data_loader(data_path, batch_size, num_workers):
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(train_dir, train_transform),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir, eval_transform),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train_model(train_loader, model, loss_fn, optim, epoch, device=None):
    losses = AverageMeter()
    top1 = AverageMeter()

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    model.train() # Train mode

    num_batches = len(train_loader)
    print_freq = int(max(1, num_batches/32))

    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = loss_fn(output, target)

        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % 16 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss ({loss.avg:.4f})\t'
                  'Acc ({top1.avg:.3f}%)\t'.format(
                   epoch, i+1, len(train_loader),
                   loss=losses, top1=top1))

def val_model(val_loader, model, loss_fn, device=None):
    losses = AverageMeter()
    top1 = AverageMeter()

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    model.eval() # Evaluation mode

    for i, (input, target) in enumerate(val_loader):
        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = loss_fn(output, target)

        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    print('Validation: '
          'Loss ({loss.avg:.4f})\t'
          'Acc ({top1.avg:.3f}%)\t'.format(
           i+1, len(val_loader), loss=losses,
           top1=top1))    

def save_model(model, save_dir, metadata):
    # Save model metadata
    with open(os.path.join(save_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    # Save checkpoint
    save_path = os.path.join(save_dir, 'model.pt')
    torch.save(model.state_dict(), save_path)
    print('Checkpoint save at {0}'.format(save_path))

def load_model(model_dir):
    with open(os.path.join(model_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    model = get_backend(metadata['backend'], metadata['classes'])
    
    model_path = os.path.join(model_dir, 'model.pt')
    model.load_state_dict(torch.load(model_path))

    return model, metadata

def classify_img(model, img_path, device):
    img = Image.open(img_path)
    img_tensor = eval_transform(img).view(1, 3, 224, 224)
    img_tensor = img_tensor.to(device)
    class_idx = model(img_tensor).argmax()
    return model.classes[class_idx]

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
