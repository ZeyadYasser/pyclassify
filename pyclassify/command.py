import os
import pickle
import argparse
import warnings

import torch
from torchvision import transforms
import torchvision.datasets as datasets
from pyclassify.models import get_backend
import pyclassify.utils as utils

def train_cmd():
    warnings.filterwarnings("ignore") # Skip Pytorch warnings

    parser = argparse.ArgumentParser(description='PyClassify for transfer learning',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--checkpoint-dir', default=None, type=str,
                        help='Directory of latest checkpoint (default: none)')
    parser.add_argument('--save-dir', required=True, type=str,
                        help='Directory to save model checkpoints')
    parser.add_argument('--data-dir', required=True, type=str,
                        help='Directory to data:\n'
                             '\troot/train/dog/xxx.png\n'
                             '\troot/train/cat/yyy.png\n\n'
                             '\troot/val/dog/xxx.png\n'
                             '\troot/val/cat/yyy.png\n')
    parser.add_argument('--epochs', default=50, type=int,
                        help='# of epochs to run (default: 50)')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size (default: 32)')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='# of CPU workers that prefetch data (default: 4)')
    parser.add_argument('--device', default='cpu', type=str, choices=['cuda', 'cpu'],
                        help='cuda or cpu (default: cpu)')
    parser.add_argument('--backend-model', default='squeeze_net', type=str,
                        help='Backend model to use (default: squeeze_net)')
    parser.add_argument('--model-name', default='image_classifer', type=str,
                        help='Name for your new model (default: image_classifier)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help='Model learning rate (default: 0.001)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Model momentum (default: 0.9)')
    parser.add_argument('--weight-decay', default=0.0002, type=float,
                        help='Model weight decay (default: 0.0002)')

    args = parser.parse_args()

    device = torch.device(args.device)

    trainloader, testloader = utils.data_loader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.checkpoint_dir is not None:
        model, metadata = utils.load_model(args.checkpoint_dir)
    else:
        classes = trainloader.dataset.classes
        metadata = {
            'backend': args.backend_model,
            'model_name': args.model_name,
            'classes': classes,
        }

    model = get_backend(metadata['backend'], metadata['classes'])
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        utils.train_model(trainloader, model, loss_fn, optimizer, epoch+1 ,device)
        print('Running Validation...')
        utils.val_model(testloader, model, loss_fn, device)
        utils.save_model(model, args.save_dir, metadata)

def run_cmd():
    warnings.filterwarnings("ignore") # Skip Pytorch warnings

    parser = argparse.ArgumentParser(description='PyClassify for transfer learning')
    parser.add_argument('checkpoint_dir', help='Directory of model checkpoint')
    parser.add_argument('image_dir', help='Path to image')
    parser.add_argument('--device', default='cpu', type=str, choices=['cuda', 'cpu'],
                        help='cuda or cpu (default: cpu)')

    args = parser.parse_args()

    device = torch.device(args.device)

    model, metadata = utils.load_model(args.checkpoint_dir)
    model.to(device)

    print('Model: "{0}" is loaded'.format(metadata['model_name']))
    model.eval()
    img_class = utils.classify_img(model, args.image_dir, device)
    print(img_class)
