import time
import torch
import os
import math
import copy
import ipdb
import torch.nn as nn


def train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model_source, criterion,criterion_bce, optimizer_feature, optimizer_domain, epoch, args, current_epoch, epoch_count_dataset):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_source = AverageMeter()
    top1_source = AverageMeter()
    top5_source = AverageMeter()
    losses_target = AverageMeter()
    losses_label = AverageMeter()
    top1_target = AverageMeter()
    top5_target = AverageMeter()
    model_source.train()

    adjust_learning_rate(optimizer_feature, current_epoch, args)
    domain_weight = adjust_weight_for_domain(current_epoch, args)
    # domain_weight = 1
    # adjust_learning_rate_domain(optimizer_domain, current_epoch, args, domain_weight)


    end = time.time()

    new_epoch_flag = False
    # for i, (input_source, target_source) in enumerate(train_loader_source):  # the iterarion in the source dataset.
    # prepare the data for the model forward and backward
    try:
        (input_source, target_source) = train_loader_source_batch.__next__()[1]
    except StopIteration:
        train_loader_source_batch = enumerate(train_loader_source)
        if epoch_count_dataset == 'source':
            print('the weight for domain loss is:', domain_weight)
            current_epoch = current_epoch + 1
            new_epoch_flag = True
        (input_source, target_source) = train_loader_source_batch.__next__()[1]


    try:
        (input_target, target_target_not_use) = train_loader_target_batch.__next__()[1]
    except StopIteration:
        train_loader_target_batch = enumerate(train_loader_target)
        if epoch_count_dataset == 'target':
            print('the weight for domain loss is:', domain_weight)
            current_epoch = current_epoch + 1
            new_epoch_flag = True
        (input_target, target_target_not_use) = train_loader_target_batch.__next__()[1]

    num_source = len(input_source)
    num_target = len(input_target)

    # dlabel_src = torch.zeros(num_source).long().cuda()
    # dlabel_tar = torch.ones(num_target).long().cuda()
    dlabel_src = torch.zeros(num_source, 1).cuda()  ## label for BCE loss
    dlabel_tar = torch.ones(num_target, 1).cuda()
    dlabel_src_var = torch.autograd.Variable(dlabel_src)
    dlabel_tar_var = torch.autograd.Variable(dlabel_tar)

    target_source = target_source.cuda(async=True)
    input_source_var = torch.autograd.Variable(input_source)
    target_source_var = torch.autograd.Variable(target_source)
    input_target_var = torch.autograd.Variable(input_target)

    data_time.update(time.time() - end)

    # calculate for the source data #######################################################
    output_source, domain_source = model_source(input_source_var, domain_weight)
    # print('domain score for source data', domain_source)
    loss_source = criterion(output_source, target_source_var)
    loss_source_domain_dis = criterion_bce(domain_source, dlabel_src_var)

    prec1, prec5 = accuracy(output_source.data, target_source, topk=(1, 5))
    losses_label.update(loss_source.data[0], input_source.size(0))
    losses_source.update(loss_source_domain_dis.data[0], input_source.size(0))
    top1_source.update(prec1[0], input_source.size(0))
    top5_source.update(prec5[0], input_source.size(0))

    # calculate for the target data#####################################################
    _, domain_target = model_source(input_target_var, domain_weight)
    # print('domain socre for target data', domain_target)
    loss_target_domain_dis = criterion_bce(domain_target, dlabel_tar_var)
    losses_target.update(loss_target_domain_dis, input_target.size(0))

    total_loss = loss_source + loss_source_domain_dis + loss_target_domain_dis

    model_source.zero_grad()
    total_loss.backward()
    optimizer_feature.step()
    model_source.zero_grad()
    # loss_for_feature.backward(retain_graph = True)
    # optimizer_feature.step()
    # model_source.zero_grad()
    # loss_total_domain_dis.backward()
    # # optimizer_domain.step()
    # model_source.zero_grad()

    batch_time.update(time.time() - end)

    print('Tr epoch [{0}/{1}]\t'
          'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'S@1 {source_top1.val:.3f} ({source_top1.avg:.3f})\t'
          'S@5 {source_top5.val:.3f} ({source_top5.avg:.3f})\t'
          'LS {source_loss.val:.4f} ({source_loss.avg:.4f})\t'
          'LT {target_loss.val:.4f} ({target_loss.avg:.4f})'.format(
           current_epoch, args.epochs, batch_time=batch_time,
           data_time=data_time, loss=losses_label, source_top1=top1_source, source_top5=top5_source, source_loss=losses_source,
           target_loss=losses_target))
    if new_epoch_flag:
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write("\n")
        log.write("Tr:epoch: %d, real_loss: %4f, source_T1 acc: %3f, source_T5 acc: %3f, source_loss: %4f, target_loss: %4f"
                  % (current_epoch, losses_label.avg, top1_source.avg, top5_source.avg, losses_source.avg, losses_target.avg))
        log.close()
    
    return train_loader_source_batch, train_loader_target_batch, current_epoch, new_epoch_flag


def validate(val_loader_target, model_source, criterion, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_source = AverageMeter()
    top1_source = AverageMeter()
    top5_source = AverageMeter()
    losses_target = AverageMeter()
    # losses_real = AverageMeter()
    top1_target = AverageMeter()
    top5_target = AverageMeter()
    model_source.eval()

    end = time.time()

    for i, (input_source, target_source) in enumerate(val_loader_target):  # the iterarion in the source dataset.
        data_time.update(time.time() - end)
        target_source = target_source.cuda(async=True)
        input_var = torch.autograd.Variable(input_source)  # volatile is fast in the evaluate model.
        target_var_source = torch.autograd.Variable(target_source)
        with torch.no_grad():
            output_source, _ = model_source(input_var, 1)
            # calculate for the source data #######################################################
            loss_source = criterion(output_source, target_var_source)
        prec1, prec5 = accuracy(output_source.data, target_source, topk=(1, 5))
        losses_source.update(loss_source.data[0], input_source.size(0))
        top1_source.update(prec1[0], input_source.size(0))
        top5_source.update(prec5[0], input_source.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Te S: [{0}][{1}/{2}]\t'
                  'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'D {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'S@1 {source_top1.val:.3f} ({source_top1.avg:.3f})\t'
                  'S@5 {source_top5.val:.3f} ({source_top5.avg:.3f})\t'
                  'LS {source_loss.val:.4f} ({source_loss.avg:.4f})'.format(
                   epoch, i, len(val_loader_target), batch_time=batch_time,
                   data_time=data_time,  source_top1=top1_source, source_top5=top5_source,
                   source_loss=losses_source,
                ))

    

    print(' * Source model Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1_source, top5=top5_source))

    log = open(os.path.join(args.log, 'log.txt'), 'a')

    log.write("\n")
    log.write("                               Test on Source:epoch: %d, loss: %4f, Top1 acc: %3f, Top5 acc: %3f" %\
                  (epoch, losses_source.avg, top1_source.avg, top5_source.avg))
    return top1_source.avg


def download_domain_scores(val_loader_target, val_loader_source, model_source, criterion, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_source = AverageMeter()
    top1_source = AverageMeter()
    top5_source = AverageMeter()

    model_source.eval()
    end = time.time()
    sigmoid = nn.Sigmoid()
    num_source_images = len(val_loader_source.dataset)
    num_target_images = len(val_loader_target.dataset)
    domain_score_source = torch.zeros(num_source_images)
    domain_score_target = torch.zeros(num_target_images)

    index = 0
    for i, (input_target, target_source) in enumerate(val_loader_target):  # the iterarion in the source dataset.
        data_time.update(time.time() - end)
        target_source = target_source.cuda(async=True)
        input_var = torch.autograd.Variable(input_target, volatile=True)  # volatile is fast in the evaluate model.
        target_var_source = torch.autograd.Variable(target_source)
        # with torch.no_grad():
        output_source, score_domain = model_source(input_var, 1)
        # calculate for the source data #######################################################
        loss_source = criterion(output_source, target_var_source)
        prec1, prec5 = accuracy(output_source.data, target_source, topk=(1, 5))
        losses_source.update(loss_source.data[0], input_target.size(0))
        top1_source.update(prec1[0], input_target.size(0))
        top5_source.update(prec5[0], input_target.size(0))
        score_domain = sigmoid(score_domain)
        for j in range(input_target.size(0)):
            # ipdb.set_trace()
            domain_score_target[index] = score_domain[j].data.cpu()[0]
            index = index + 1
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('do_score: [{0}][{1}/{2}]\t'
                  'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'D {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'S@1 {source_top1.val:.3f} ({source_top1.avg:.3f})\t'
                  'S@5 {source_top5.val:.3f} ({source_top5.avg:.3f})\t'
                  'LS {source_loss.val:.4f} ({source_loss.avg:.4f})'.format(
                epoch, i, len(val_loader_target), batch_time=batch_time,
                data_time=data_time, source_top1=top1_source, source_top5=top5_source,
                source_loss=losses_source,
            ))
    domain_score_target = domain_score_target[0:index]

    index = 0
    for i, (input_source, target_source) in enumerate(val_loader_source):  # the iterarion in the source dataset.
        data_time.update(time.time() - end)
        target_source = target_source.cuda(async=True)
        input_var = torch.autograd.Variable(input_source, volatile=True)  # volatile is fast in the evaluate model.
        target_var_source = torch.autograd.Variable(target_source)
        # with torch.no_grad():
        output_source, score_domain = model_source(input_var, 1)
        # calculate for the source data #######################################################
        loss_source = criterion(output_source, target_var_source)
        prec1, prec5 = accuracy(output_source.data, target_source, topk=(1, 5))
        losses_source.update(loss_source.data[0], input_source.size(0))
        top1_source.update(prec1[0], input_source.size(0))
        top5_source.update(prec5[0], input_source.size(0))
        score_domain = sigmoid(score_domain)
        for j in range(input_source.size(0)):
            domain_score_source[index] = score_domain[j].data.cpu()[0]
            index = index + 1
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('do_score: [{0}][{1}/{2}]\t'
                  'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'D {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'S@1 {source_top1.val:.3f} ({source_top1.avg:.3f})\t'
                  'S@5 {source_top5.val:.3f} ({source_top5.avg:.3f})\t'
                  'LS {source_loss.val:.4f} ({source_loss.avg:.4f})'.format(
                epoch, i, len(val_loader_source), batch_time=batch_time,
                data_time=data_time, source_top1=top1_source, source_top5=top5_source,
                source_loss=losses_source,
            ))
    domain_score_source = domain_score_source[0:index]
    save_socre = {
        'source': domain_score_source,
        'target': domain_score_target,
    }
    filename_to_save = 'domain_score_epoch' + str(epoch) + '.pth.tar'
    dir_save_file = os.path.join(args.log, filename_to_save)
    torch.save(save_socre, dir_save_file)


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


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according the epoch"""
    if args.lrplan == 'step':
        raise ValueError('the required parameter group is not exist.')
    elif args.lrplan == 'dao':
        lr = args.lr / math.pow((1 + 10 * epoch/args.epochs), 0.75)
        for param_group in optimizer.param_groups:
           if param_group['name'] == 'conv':
               param_group['lr'] = lr * 0.1
           elif param_group['name'] == 'do_cl':
               param_group['lr'] = lr
           elif param_group['name'] == 'ca_cl':
               param_group['lr'] = lr
           else:
               raise ValueError('the required parameter group is not exist.')

def adjust_learning_rate_domain(optimizer, epoch, args, weight_for_da=1):
    """Adjust the learning rate according the epoch"""
    if args.lrplan == 'step':
        raise ValueError('the required parameter group is not exist.')
    elif args.lrplan == 'dao':
        lr = args.lr / math.pow((1 + 10 * epoch/args.epochs), 0.75)
        for param_group in optimizer.param_groups:
           if param_group['name'] == 'conv':
               param_group['lr'] = lr * 0.1 * weight_for_da
           elif param_group['name'] == 'ca_cl':
               param_group['lr'] = lr * weight_for_da
           elif param_group['name'] == 'do_cl':
               param_group['lr'] = lr
           else:
               raise ValueError('the required parameter group is not exist.')

def adjust_weight_for_domain(epoch, args):
    lambda_weight = 2 / (1 + math.exp(-10 * (epoch / args.epochs))) - 1
    return lambda_weight

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
