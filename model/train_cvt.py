##### train_vit for experiment

import argparse

import os
import gc
import time
import json
import shutil
import logging
import functools

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch import optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from transformers import AutoTokenizer, ViTFeatureExtractor
from transformers import ViTFeatureExtractor, ViTForImageClassification, CvtForImageClassification
from torch.utils.tensorboard import SummaryWriter
#import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

NUM_CLASSES = 4

logger = logging.getLogger(__name__)

def create_dir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def create_directory_info(args, create_dir=True):

    model_dir = os.path.join(args.output_dir, args.vision_model)
    if args.dir_suffix is not None:
        model_dir = '_'.join([model_dir, args.dir_suffix])
    weights_dir = os.path.join(model_dir, "weights")
    logs_dir = os.path.join(model_dir, "logs")

    path_info = {
        'model_dir': model_dir,
        'weights_dir': weights_dir,
        'logs_dir': logs_dir,
    }

    if create_dir:
        for k, v in path_info.items():
            create_dir_if_not_exist(v)

    path_info['best_model_path'] = os.path.join(weights_dir, "best_model.pth")
    path_info['ckpt_path'] = os.path.join(weights_dir, "checkpoint.pth")
    return path_info

class AverageMeter(object):
    """Computes and stores the average and current value"""

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

def save_checkpoint(state, is_best, filename='checkpoint.pth', best_filename='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train_path",
                        default="./train", type=str)
    parser.add_argument("--validation_path",
                        default="./val", type=str)
    parser.add_argument("--image_root_dir",
                        default=None, type=str)

    # model
    parser.add_argument("--vision_model",
                        default="google/vit-base-patch16-384", type=str)
    
    parser.add_argument("--num_classes", default=NUM_CLASSES, type=int,
                        help="number of classes")

    parser.add_argument("--dir_suffix",
                        default=None, type=str)
    parser.add_argument("--output_dir",
                        default="output", type=str)

    # resume
    parser.add_argument("--resume", default=None, type=str,
                        help="path to checkpoint.")
    parser.add_argument("--hf_path", default=None, type=str,
                        help="path to score huggingface model")
                    

    # default settings for training, evaluation
    parser.add_argument("--batch_size", default=8,
                        type=int, help="mini batch size")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="number of workers")
    parser.add_argument("--print_freq", default=1000,
                        type=int, help="print frequency")
    parser.add_argument("--global_steps", default=0,
                        type=int, help="variable for global steps")

    # default settings for training
    parser.add_argument("--epochs", default=1, type=int,
                        help="number of epochs for training")
    parser.add_argument("--start_epoch", default=0,
                        type=int, help="start epoch")
    parser.add_argument("--save_freq", default=1000,
                        type=int, help="steps to save checkpoint")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=2e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup", default=0.1,
                        type=float, help="warm-up proportion for linear scheduling")
    parser.add_argument("--logit_temperature", default=1.0,
                        type=float, help="temperature for logits")
    parser.add_argument("--label_smoothing", default=0.1,
                        type=float, help="label smoothing for cross entropy")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--off_scheduling", action='store_false',
                        help="off_scheduling")
    parser.add_argument("--max_validation_steps", default=1000, type=int,
                        help="max steps for validation")
    
    # ddp settings for sync
    parser.add_argument("--seed", default=0,
                        type=int, help="seed for torch manual seed")
    parser.add_argument("--deterministic", action='store_true',
                        help="deterministic")
    parser.add_argument("--save_every_epoch", action='store_true',
                        help="save check points on every epochs")
    parser.add_argument("--freeze_lm", action='store_true',
                        help="freeze language model")

    args = parser.parse_args()

    # create directory and summary logger
    best_acc = 0
    path_info = create_directory_info(args)
    summary_logger = SummaryWriter(path_info["logs_dir"])
    path_info = create_directory_info(args, create_dir=False)
    
    # device
    device = torch.device('cuda')

    # deterministic seed
    if args.deterministic:
        torch.manual_seed(args.seed)
        data_seed = args.seed
    else:
        data_seed = torch.randint(9999, (1,), device=device, requires_grad=False)
        data_seed = data_seed.cpu().item()
        logger.info("[rank {}]seed for data: {}".format(0, data_seed))

    # update batch_size per a device
    args.batch_size = int(
        args.batch_size / args.gradient_accumulation_steps)

    if(args.vision_model == 'google/vit-base-patch16-384'):
        model = ViTForImageClassification.from_pretrained(args.vision_model)
    else:
        model = CvtForImageClassification.from_pretrained(args.vision_model)
    model.classifier = torch.nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model = model.to(device)

    # get optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay
    )

    loss_function = CrossEntropyLoss()

    # Load the dataset and define the optimizer and loss function
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # create dataset
    train_dataset = torchvision.datasets.ImageFolder(
            args.train_path, transform=transform)

        
    val_dataset = torchvision.datasets.ImageFolder(
            args.validation_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               args.batch_size, 
                                               shuffle=False, 
                                               num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             args.batch_size, 
                                             shuffle=False, 
                                             num_workers=args.num_workers)
    
    
    # learning rate scheduler
    scheduler = None
    if args.off_scheduling:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=args.epochs,
            last_epoch=-1,
            steps_per_epoch=int(len(train_loader)/args.gradient_accumulation_steps),
            pct_start=args.warmup,
            anneal_strategy="linear"
        )

    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                
                checkpoint = torch.load(
                    args.resume, map_location=lambda storage, loc: storage.cuda(args.local_rank))

                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint and scheduler is not None:
                    if checkpoint['scheduler'] is not None:
                        scheduler.load_state_dict(checkpoint['scheduler'])

                args.start_epoch = checkpoint['epoch']
                if args.resume.endswith('-train'):
                    args.global_steps = checkpoint['global_step']
                    logger.info("=> global_steps '{}'".format(args.global_steps))
                    args.start_epoch-=1

                best_acc = checkpoint['best_acc'] if 'best_acc' in checkpoint else 0
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            elif args.resume.lower()=='true':
                args.resume = path_info['ckpt_path']
                resume()
            elif args.resume.lower()=='best':
                args.resume = path_info['best_model_path']
                resume()
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
        resume()
    
    optimizer.param_groups[0]['capturable'] = True

    # save model as huggingface model
    if args.hf_path:
        if args.hf_path.lower()=='default':
            args.hf_path = os.path.join(path_info["model_dir"], "hf")

        model.module.save_pretrained(args.hf_path)
        logger.info('hf model is saved in {}'.format(args.hf_path))
        exit()

    train_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(args.start_epoch, args.epochs):
        total_preds = []
        total_labels = []
        
        # Training
        train(train_losses, train_acc, train_loader, model, optimizer, loss_function, scheduler, epoch, args, path_info, summary_logger=summary_logger)
        args.global_steps = 0

        # Validation
        scores = validate(val_acc, val_loader, model, epoch, args, total_labels, total_preds)

        ckpt_path = os.path.join(path_info["weights_dir"], "ckpt-{}.pth".format(epoch)) if args.save_every_epoch else path_info["ckpt_path"]

        is_best = scores["accuracy"] > best_acc
        if scores["accuracy"] > best_acc:
            best_acc = scores["accuracy"]
            best_epoch = epoch+1
        best_acc = max(scores["accuracy"], best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'scheduler': scheduler.state_dict() if scheduler is not None else scheduler,
        }, is_best, ckpt_path, path_info["best_model_path"])

        summary_logger.add_scalar('eval/accuracy', scores['accuracy'], epoch)

        total_labels = torch.cat(total_labels).cpu().numpy()
        total_preds = torch.cat(total_preds).cpu().numpy()
        
        print(confusion_matrix(total_labels, total_preds))
        target_name = ['N3(3)', 'N1+N2(4+5)', 'wake(6)', 'REM(7)']
        print(classification_report(total_labels, total_preds, target_names = target_name))

    summary_logger.close()
    print('검증 세트 기준 best accuracy : ', best_acc)
    print('검증 세트 기준 best epoch : ', best_epoch)

def train(train_losses, train_acc, train_loader, model, optimizer, loss_function, scheduler, epoch, args, path_info, summary_logger=None):
    batch_time = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = 100 * correct / total
    train_acc.append(train_accuracy)

    with torch.no_grad():
        batch_time.update((time.time() - end) / args.print_freq)
        end = time.time()

        summary_logger.add_scalar('train/loss', train_loss, epoch)
        summary_logger.add_scalar('train/accuracy', train_accuracy, epoch)

        score_log = "loss\t{:.3f}\t accuracy\t{:.3f}".format(train_loss, train_accuracy)

        logger.info('-----Training----- \nEpoch: [{0}]\t'.format(
                        epoch) + score_log)

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else scheduler,
    }, False, path_info["ckpt_path"] + "-train", path_info["best_model_path"])


def validate(val_acc, val_loader, model, epoch, args, total_labels, total_preds):
    batch_time = AverageMeter()

    # switch to evaluate mode (for drop out)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        end = time.time()

        for inputs, labels in val_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_labels.append(labels)
            total_preds.append(predicted)

        val_accuracy = 100 * correct / total
        val_acc.append(val_accuracy)

        batch_time.update((time.time() - end) / args.print_freq)
        end = time.time()

        score_log = "accuracy\t{:.3f}".format(val_accuracy)

        logger.info('-----Evaluation----- \nEpoch: [{0}]\t'.format(
                epoch) + score_log)

    scores = {
        "accuracy": val_accuracy
    }
    score_log = "accuracy\t{:.3f}\n".format(scores["accuracy"])

    logger.info('-----Evaluation----- \nEpoch: [{0}]\t'.format(epoch) + score_log)

    return scores



if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)

    main()

