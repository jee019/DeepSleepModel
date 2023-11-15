##### test_vit for experiment

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

def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_path",
                        default='./test',
                        type=str, help="path to the test data")
    parser.add_argument("--image_root_dir",
                        default=None, type=str)

    # model
    parser.add_argument("--vision_model",
                        default="google/vit-base-patch16-384", type=str)
    
    parser.add_argument("--num_classes", default=NUM_CLASSES, type=int,
                        help="number of classes")

    parser.add_argument("--dir_suffix",
                        default=None, type=str)
    
    parser.add_argument("--hf_path",
                        default='./output/vit/weights/',
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
    parser.add_argument("--learning_rate", default=1e-4,
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
        checkpoint_path = "./output/google/vit-base-patch16-384/weights/best_model.pth"

    else:
        model = CvtForImageClassification.from_pretrained(args.vision_model)
        checkpoint_path = "./output/microsoft/cvt-13/weights/best_model.pth"

    model.classifier = torch.nn.Linear(model.classifier.in_features, NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    loss_function = CrossEntropyLoss()


    # Load the dataset and define the optimizer and loss function
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    # create dataset
    test_dataset = torchvision.datasets.ImageFolder(
            args.data_path, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers)

    for epoch in range(args.start_epoch, args.epochs):
        model.eval()
        total_labels = []
        total_preds = []
        val_losses = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = model(inputs)
                loss = loss_function(outputs.logits, labels)
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs.logits.data, 1)

                total_labels.append(labels)
                total_preds.append(predicted)
        
        total_labels = torch.cat(total_labels).cpu().numpy()
        total_preds = torch.cat(total_preds).cpu().numpy()
        target_name = ['N3(3)', 'N1+N2(4+5)', 'wake(6)', 'REM(7)']
        report = classification_report(total_labels, total_preds, target_names=target_name)
        confusion = confusion_matrix(total_labels, total_preds)

        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", confusion)

if __name__ == "__main__":
    main()