import argparse
from model.densenet import get_multi_label_densenet
from lib.utils.avgmeter import AverageMeter
from lib.dataloader import chexpert_dataset
import os
from os import path
import time
import shutil
import ast
from sklearn.metrics import roc_auc_score
from lib.criterion import ClsCriterion
from collections import defaultdict
import re
import numpy as np
from lib.utils.utils import get_score_label_array_from_dict


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


parser = argparse.ArgumentParser(description='CheXpert Classifier')
# Dataset Parameters
parser.add_argument('-bp', '--base_path', default="/data/fhz")
parser.add_argument('--dataset', default="CheXpert", type=str, help="The dataset name")
parser.add_argument('-is', "--image-size", default=[320, 320], type=arg_as_list,
                    metavar='Image Size List', help='the size of h * w for image')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# Train Strategy Parameters
parser.add_argument('-t', '--train-time', default=1, type=int,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--dp', '--data-parallel', action='store_false', help='Use Data Parallel')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-arg', action='store_false', help='if we not resume the argument')
# Mixup Strategy Parameters
# Not use now
# parser.add_argument('--mixup', default=False, type=bool, help="use mixup method")
# parser.add_argument('--manifold-mixup', default=False, type=bool, help="use manifold mixup method")
# parser.add_argument('--mll', "--mixup-layer-list", default=[0, 2], type=arg_as_list,
#                     help="The mixup layer list for manifold mixup strategy")
# parser.add_argument('--ma', "--mixup-alpha", default=0.2, type=float, help="the lambda for mixup method")
# Deep Learning Model Parameters
parser.add_argument('--net-name', default="densenet121", type=str, help="the name for network to use")
# parser.add_argument('--depth', default=28, type=int, metavar='D', help="the depth of neural network")
# parser.add_argument('--width', default=2, type=int, metavar='W', help="the width of neural network")
parser.add_argument('--dr', '--drop-rate', default=0, type=float, help='dropout rate')
# Optimizer Parameters
parser.add_argument('--optimizer', default="Adam", type=str, metavar="Optimizer Name")
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M', help='Momentum in SGD')
parser.add_argument('--nesterov', action='store_true', help='nesterov in sgd')
parser.add_argument('-ad', "--adjust-lr", default=[20, 40], type=arg_as_list,
                    help="The milestone list for adjust learning rate")
parser.add_argument('--lr-decay-ratio', default=0.1, type=float)
parser.add_argument('--wd', '--weight-decay', default=0, type=float)
# parser.add_argument('--wul', '--warm-up-lr', default=0.02, type=float, help='the learning rate for warm up method')
# GPU Parameters
parser.add_argument("--gpu", default="0,1", type=str, metavar='GPU plans to use', help='The GPU id plans to use')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR


def main(args=args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # build dataset
    if args.dataset == "CheXpert":
        dataset_base_path = path.join(args.base_path, "dataset", "CheXpert")
        train_dataset = chexpert_dataset(dataset_base_path, args.image_size)
        valid_dataset = chexpert_dataset(dataset_base_path, args.image_size, train_flag=False)
        train_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                   pin_memory=True, shuffle=True)
        valid_dloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                   pin_memory=True, shuffle=True)
        class_config = (2, 2, 2, 2, 2)
    else:
        raise NotImplementedError("Dataset {} Not Implemented".format(args.dataset))
    if "densenet" in args.net_name:
        model = get_multi_label_densenet(args.net_name, drop_rate=args.dr, data_parallel=args.dp,
                                         class_config=class_config)
    else:
        raise NotImplementedError("model {} not implemented".format(args.net_name))
    model = model.cuda()

    input("Begin the {} time's training, Dataset:{}".format(args.train_time, args.dataset))
    criterion = ClsCriterion()
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd,
                                    nesterov=args.nesterov)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.99), weight_decay=args.wd)
    else:
        raise NotImplementedError("{} not find".format(args.optimizer))
    scheduler = MultiStepLR(optimizer, milestones=args.adjust_lr, gamma=args.lr_decay_ratio)
    writer_log_dir = "{}/{}/runs/train_time:{}".format(args.base_path, args.dataset, args.train_time)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.resume_arg:
                args = checkpoint['args']
                args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise FileNotFoundError("Checkpoint Resume File {} Not Found".format(args.resume))
    else:
        if os.path.exists(writer_log_dir):
            flag = input("{} train_time:{} will be removed, input yes to continue:".format(
                args.dataset, args.train_time))
            if flag == "yes":
                shutil.rmtree(writer_log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=writer_log_dir)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step(epoch)
        # if epoch == 0:
        #     # do warmup
        #     modify_lr_rate(opt=optimizer, lr=args.wul)
        train(train_dloader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, writer=writer)
        valid(valid_dloader, model=model, criterion=criterion, epoch=epoch, writer=writer)
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            "state_dict": model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
        # if epoch == 0:
        #     modify_lr_rate(opt=optimizer, lr=args.lr)


def train(train_dloader, model, criterion, optimizer, epoch, writer):
    # some records
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    for i, (image, index, image_name, label_weight, label) in enumerate(train_dloader):
        data_time.update(time.time() - end)
        image = image.float().cuda()
        label = label.long().cuda()
        label_weight = label_weight.cuda()
        prediction_list = model(image)
        loss = criterion(prediction_list, label_weight, label)
        loss.backward()
        losses.update(float(loss.item()), image.size(0))
        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            train_text = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                         'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})'.format(
                epoch, i + 1, len(train_dloader), batch_time=batch_time, data_time=data_time,
                cls_loss=losses)
            print(train_text)
    writer.add_scalar(tag="Train/cls_loss", scalar_value=losses.avg, global_step=epoch + 1)
    return losses.avg


def valid(valid_dloader, model, criterion, epoch, writer):
    """
    Here valid dataloader we may need to organize it with each study
    """
    model.eval()
    # calculate score and label for different dataset
    atelectasis_score_dict = defaultdict(list)
    atelectasis_label_dict = defaultdict(list)
    cardiomegaly_score_dict = defaultdict(list)
    cardiomegaly_label_dict = defaultdict(list)
    consolidation_score_dict = defaultdict(list)
    consolidation_label_dict = defaultdict(list)
    edema_score_dict = defaultdict(list)
    edema_label_dict = defaultdict(list)
    pleural_effusion_score_dict = defaultdict(list)
    pleural_effusion_label_dict = defaultdict(list)
    # calculate index for valid dataset
    losses = AverageMeter()
    for idx, (image, index, image_name, label_weight, label) in enumerate(valid_dloader):
        image = image.float().cuda()
        label = label.long().cuda()
        label_weight = label_weight.cuda()
        with torch.no_grad():
            prediction_list = model(image)
            loss = criterion(prediction_list, label_weight, label)
            losses.update(float(loss.item()), image.size(0))
        for i, img_name in enumerate(image_name):
            study_name = re.match(r"(.*)patient(.*)\|(.*)", img_name).group(2)
            for j, prediction in enumerate(prediction_list):
                score = prediction[i, 1].item()
                item_label = label[i, j].item()
                if j == 0:
                    atelectasis_label_dict[study_name].append(item_label)
                    atelectasis_score_dict[study_name].append(score)
                elif j == 1:
                    cardiomegaly_label_dict[study_name].append(item_label)
                    cardiomegaly_score_dict[study_name].append(score)
                elif j == 2:
                    consolidation_label_dict[study_name].append(item_label)
                    consolidation_score_dict[study_name].append(score)
                elif j == 3:
                    edema_label_dict[study_name].append(item_label)
                    edema_score_dict[study_name].append(score)
                else:
                    pleural_effusion_label_dict[study_name].append(item_label)
                    pleural_effusion_score_dict[study_name].append(score)
    writer.add_scalar(tag="Valid/cls_loss", scalar_value=losses.avg, global_step=epoch + 1)
    # Calculate AUC ROC
    # Here we use the max method to get the score and label list of each study
    atelectasis_score, atelectasis_label = get_score_label_array_from_dict(atelectasis_score_dict,
                                                                           atelectasis_label_dict)
    cardiomegaly_score, cardiomegaly_label = get_score_label_array_from_dict(cardiomegaly_score_dict,
                                                                             cardiomegaly_label_dict)

    consolidation_score, consolidation_label = get_score_label_array_from_dict(consolidation_score_dict,
                                                                               consolidation_label_dict)
    edema_score, edema_label = get_score_label_array_from_dict(edema_score_dict, edema_label_dict)
    pleural_effusion_score, pleural_effusion_label = get_score_label_array_from_dict(pleural_effusion_score_dict,
                                                                                     pleural_effusion_label_dict)
    atelectasis_auc = roc_auc_score(atelectasis_label, atelectasis_score)
    cardiomegaly_auc = roc_auc_score(cardiomegaly_label, cardiomegaly_score)
    consolidation_auc = roc_auc_score(consolidation_label, consolidation_score)
    edema_auc = roc_auc_score(edema_label, edema_score)
    pleural_effusion_auc = roc_auc_score(pleural_effusion_label, pleural_effusion_score)
    writer.add_scalar(tag="Valid/Atelectasis_AUC", scalar_value=atelectasis_auc, global_step=epoch)
    writer.add_scalar(tag="Valid/Cardiomegaly_AUC", scalar_value=cardiomegaly_auc, global_step=epoch)
    writer.add_scalar(tag="Valid/Consolidation_AUC", scalar_value=consolidation_auc,
                      global_step=epoch)
    writer.add_scalar(tag="Valid/Edema_AUC", scalar_value=edema_auc, global_step=epoch)
    writer.add_scalar(tag="Valid/Pleural_Effusion_AUC", scalar_value=pleural_effusion_auc,
                      global_step=epoch)
    return [atelectasis_auc, cardiomegaly_auc, consolidation_auc, edema_auc, pleural_effusion_auc]


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    :param state: a dict including:{
                'epoch': epoch + 1,
                'args': args,
                "state_dict": model.state_dict(),
                'optimizer': optimizer.state_dict(),
        }
    :param filename: the filename for store
    :return:
    """
    filefolder = "{}/{}/parameter/train_time:{}".format(args.base_path, args.dataset, args.train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


def modify_lr_rate(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
