from data import *
from utils.augmentations import SSDAugmentation
from utils.utils import *
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from tensorboardX import SummaryWriter
import numpy as np
import argparse
import ipdb
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='ICDAR', choices=['ICDAR', 'COCO_TEXT'],
                    type=str, help='ICDAR or COCO_TEXT')
# parser.add_argument('--dataset_root', default=ICDAR_ROOT,
#                     help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    cfg = cfg300
    if args.dataset == 'ICDAR':
        dataset = ICDARDataset(root=ICDAR_ROOT,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))  
    elif args.dataset == 'COCO_TEXT':
        dataset = COCOTEXTDataset(root=COCO_ROOT,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS)) 
        
    directory = os.path.join('run', args.dataset)
    runs = sorted(glob.glob(os.path.join(directory, 'experiment_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    experiment_dir = os.path.join(directory, 'experiment_{}'.format(str(run_id)))
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    writer = SummaryWriter(log_dir=experiment_dir)
    
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Epoch size is ', epoch_size)
    print('Using the specified args:')
    print(args)
    save_experiment_config(args, experiment_dir)
        
    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    t0 = time.time()
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            targets = [ann for ann in targets]
        
        # forward
        out = net(images)
        # backprop
        optimizer.zero_grad()         
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (time.time() - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')
            writer.add_scalar("train/loss", loss.item(), iteration)
            t0 = time.time()
        
        if iteration != 0 and (iteration % epoch_size == 0):
            loc_loss = 0
            conf_loss = 0
            epoch += 1
            batch_iterator = iter(data_loader)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalars("train/total_loss", 
                              {'loss_l': loss_l.item(),
                               'loss_c': loss_c.item()}, epoch)
            
        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 
                       '{}/ssd{}_{}_'.format(experiment_dir, cfg['min_dim'], args.dataset) +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), os.path.join(experiment_dir, args.dataset + '.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    train()
