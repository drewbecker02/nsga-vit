import os
import sys
# update your project root path before running
sys.path.insert(0, '/home/ajb46717/workDir/projects/nsgaformer')
sys.path.insert(1, '/mnt/data4TBa/ajb46717/imagenet')
import numpy as np
import time
import torch
from misc import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import default_collate
import torch.backends.cudnn as cudnn
import pytorch_warmup as warmup
from torchvision.transforms import v2

from torch.optim.lr_scheduler import LambdaLR

import models.micro_genotypes as genotypes
from torch.autograd import Variable
from models.micro_models import ViTNetworkImageNet as Network

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='/mnt/data4TBa/ajb46717/imagenet', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=3e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1000, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=512, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='NSGA_ViT', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=1., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--patchify', action='store_true', default=True, help='patchify stem')
parser.add_argument('--resume', type=str, default=None, help='directory to resume from')
parser.add_argument('--warmup', type=int, default=None, help='Number of warmup steps')
parser.add_argument('--mixup', type=float, default=None, help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--randaug', type=float, default=None, help='randaugment magnitude')

args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
    
def collate_fn(batch):

    randaug = transforms.RandAugment(num_ops=2, magnitude=args.randaug)
            # aug += randaug
    return randaug(*default_collate(batch))
        

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    if args.parallel:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        args.learning_rate,
        # momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              normalize,
        ]))
    
    if args.mixup or args.randaug:
        train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=12, collate_fn = collate_fn)
    else:
        train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=12)
        
    valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    

            

    #Set Scheduler
    num_steps =  args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    if args.warmup:    
        warmup_scheduler = warmup.LinearWarmup(optimizer, args.warmup)
    else:
        warmup_scheduler=None
    # optionally resume from a checkpoint

    if args.resume:
        checkpoint = utils.load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc_top1 = checkpoint['best_acc_top1']
        scheduler.load_state_dict(checkpoint['scheduler'])
        if args.warmup:
            warmup_schedule.load_state_dict(checkpoint['warmup'])

    else:
        start_epoch = 0
        best_acc_top1 = 0
        
    for epoch in range(start_epoch, args.epochs):
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_prob = args.drop_path_prob #* epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer,
                                     scheduler, warmup_scheduler)
        logging.info('train_acc %f', train_acc)

        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc_top1 %f', valid_acc_top1)
        logging.info('valid_acc_top5 %f', valid_acc_top5)

        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
            
        if args.warmup:
            with warmup_scheduler.dampening():
                scheduler.step()
        else:
            scheduler.step()
        

        
        if args.warmup:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'warmup': warmup_scheduler.state_dict(),
              }, is_best, args.save)
        else:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
              }, is_best, args.save)    


def train(train_queue, model, criterion, optimizer, scheduler, warmup_scheduler):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
        
    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True) #non_blocking=True
        input = input.cuda()
        input = Variable(input)
        target = Variable(target)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda(non_blocking=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main() 
