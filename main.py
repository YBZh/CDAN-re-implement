##############################################################################
#
# All the codes about the model constructing should be kept in the folder ./models/
# All the codes about the data process should be kept in the folder ./data/
# The file ./opts.py stores the options.
# The file ./trainer.py stores the training and test strategy
# The ./main.py should be simple
#
##############################################################################
import os
import json
import shutil
import torch.optim
import torch.nn as nn
import random
import numpy as np
import torch.backends.cudnn as cudnn
from models.model_construct import Model_Construct
from trainer import train  # For the training process
from trainer import validate  # For the validate (test) process
from trainer import download_domain_scores
from opts import opts  # The options for the project
from data.prepare_data import generate_dataloader  # Prepare the data and dataloader
import time
import ipdb

best_prec1 = 0

def main():
    global args, best_prec1, current_epoch, epoch_count_dataset
    current_epoch = 0
    epoch_count_dataset = 'source'
    args = opts()
    # ipdb.set_trace()
    # args = parser.parse_args()
    model_source = Model_Construct(args)
    # define-multi GPU
    model_source = torch.nn.DataParallel(model_source).cuda()

    # define loss function(criterion) and optimizer

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_bce = nn.BCEWithLogitsLoss().cuda()
    np.random.seed(1)  ### fix the test data.
    random.seed(1)
    # optimizer = torch.optim.SGD(model.parameters(),
    # To apply different learning rate to different layer
    if args.domain_feature == 'original':
        print('domain feature is original')
        optimizer_feature = torch.optim.SGD([
            {'params': model_source.module.base_conv.parameters(), 'name': 'conv'},
            {'params': model_source.module.domain_classifier.parameters(), 'name': 'do_cl'},
            {'params': model_source.module.fc.parameters(), 'name': 'ca_cl'}
        ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        optimizer_domain = torch.optim.SGD([
            {'params': model_source.module.base_conv.parameters(), 'name': 'conv'},
            {'params': model_source.module.domain_classifier.parameters(), 'name': 'do_cl'},
            {'params': model_source.module.fc.parameters(), 'name': 'ca_cl'}
        ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    elif args.domain_feature == 'full_bilinear' or args.domain_feature == 'random_bilinear':
        print('the domain feature is full bilinear')
        optimizer_feature = torch.optim.SGD([
            {'params': model_source.module.base_conv.parameters(), 'name': 'conv'},
            {'params': model_source.module.domain_classifier.parameters(), 'name': 'do_cl'},
            {'params': model_source.module.fc.parameters(), 'name': 'ca_cl'}
        ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        optimizer_domain = torch.optim.SGD([
            {'params': model_source.module.base_conv.parameters(), 'name': 'conv'},
            {'params': model_source.module.domain_classifier.parameters(), 'name': 'do_cl'},
            {'params': model_source.module.fc.parameters(), 'name': 'ca_cl'}
        ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise ValueError('the requested domain feature is not available', args.domain_feature)


    if args.resume:
        if os.path.isfile(args.resume):
            # raise ValueError('the resume function is not finished')
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            current_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model_source.load_state_dict(checkpoint['source_state_dict'])
            print("==> loaded checkpoint '{}'(epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('The file to be resumed from is not exited', args.resume)
    else:
        if not os.path.isdir(args.log):
            os.makedirs(args.log)
        log = open(os.path.join(args.log, 'log.txt'), 'w')
        state = {k: v for k, v in args._get_kwargs()}
        log.write(json.dumps(state) + '\n')
        log.close()

    log = open(os.path.join(args.log, 'log.txt'), 'a')
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log.write(local_time)
    log.close()

    cudnn.benchmark = True
    # process the data and prepare the dataloaders.
    # train_loader_source, val_loader_source, train_loader_target, val_loader_target = generate_dataloader(args)
    # train_loader, val_loader = generate_dataloader(args)
    train_loader_source, train_loader_target, val_loader_target, val_loader_source = generate_dataloader(args)

    # print('this is the first validation')
    # validate(val_loader_source, val_loader_target, model_source, model_target, criterion, 0, args)
    print('begin training')
    train_loader_source_batch = enumerate(train_loader_source)
    train_loader_target_batch = enumerate(train_loader_target)
    batch_number_s = len(train_loader_source)
    batch_number_t = len(train_loader_target)
    if batch_number_s < batch_number_t:
        epoch_count_dataset = 'target'

    for epoch in range(args.start_epoch, 1000000000000000000):
        # train for one epoch
        train_loader_source_batch, train_loader_target_batch, current_epoch, new_epoch_flag = train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model_source, criterion, criterion_bce, optimizer_feature, optimizer_domain, epoch, args, current_epoch, epoch_count_dataset)
        # train(train_loader, model, criterion, optimizer, epoch, args)
        # evaluate on the val data
        if new_epoch_flag:
            prec1 = validate(val_loader_target, model_source, criterion, current_epoch, args)
            # prec1 = 1
            # record the best prec1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best:
                log = open(os.path.join(args.log, 'log.txt'), 'a')
                log.write('                        Target_T1 acc: %3f' % (best_prec1))
                log.close()
                save_checkpoint({
                    'epoch': current_epoch + 1,
                    'arch': args.arch,
                    'source_state_dict': model_source.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, args)
            if (current_epoch + 1) % args.domain_freq == 0:
                download_domain_scores(val_loader_target, val_loader_source,model_source, criterion, current_epoch, args)
        if current_epoch > args.epochs:
            break

def save_checkpoint(state, is_best, args):
    if is_best:
        filename = 'final_checkpoint.pth.tar'
        dir_save_file = os.path.join(args.log, filename)
        torch.save(state, dir_save_file)
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()





