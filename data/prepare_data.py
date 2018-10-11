import os
import shutil
import torch
import torchvision.transforms as transforms
#import torchvision.datasets as datasets
from data.folder_new import ImageFolder_new

def generate_dataloader(args):
    # Data loading code
    traindir = os.path.join(args.data_path_source, args.src)
    traindir_t = os.path.join(args.data_path_source_t, args.src_t)
    valdir = os.path.join(args.data_path_target, args.tar)

    if not os.path.isdir(traindir):
        raise ValueError ('the require data path is not exist, please download the dataset')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # the mean and std of ImageNet
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ImageFolder_new(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            # transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers = args.workers, pin_memory=True, sampler=None, drop_last=True
    )

    train_dataset_doscore = ImageFolder_new(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader_doscore = torch.utils.data.DataLoader(
        train_dataset_doscore, batch_size=args.batch_size, shuffle=True,
        num_workers = args.workers, pin_memory=True, sampler=None, drop_last=True
    )

    train_t_dataset = ImageFolder_new(
        traindir_t,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            # transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_t_loader = torch.utils.data.DataLoader(
        train_t_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers = args.workers, pin_memory=True, sampler=None, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        ImageFolder_new(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    return train_loader, train_t_loader, val_loader, train_loader_doscore

