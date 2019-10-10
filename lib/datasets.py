import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import lib.cifar
import lib.mnist
from PIL import ImageFile

def get_batches(iterable, n=1):
    l = len(iterable)
    batches = [iterable[ndx:min(ndx + n, l)] for ndx in range(0, l, n)]
    return batches

def split(dataset_size, split_size):
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    strides = get_batches(indices, split_size)
    return strides

class Dataset(object):
    # TODO: remove this first_split_size nonsense
    def __init__(self, split_size):
        self._split_size = split_size
        self.first_split = True

    @property
    def split_size(self):
        if self._split_size is None:
            return self.num_training_images
        else:
            return self._split_size

    def get_split_size(self, first_split_size):
        if self.first_split and first_split_size is not None and first_split_size > 0:
            return first_split_size
        else:
            return self.split_size

    def get_dataset_splits(self, first_split_size=None):
        split_size = self.get_split_size(first_split_size)
        splits =  split(self.num_training_images, split_size)
        self.first_split = False
        return splits

class CIFAR10(Dataset):
    def __init__(self, model, test_batch_size, augment, split_size, randomize_labels):

        super(CIFAR10, self).__init__(split_size)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.model = model
        self.augment = augment

        # Testing set
        transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = lib.cifar.CIFAR10(root='./data',
                                    train=False,
                                    download=False,
                                    transform=transform_test,
                                    randomize_labels=randomize_labels)
        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=test_batch_size,
                                                      shuffle=False,
                                                      num_workers=2)

        # Training set
        if self.augment:
            print("Performing data augmentation on CIFAR10")
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            print("Not performing data augmentation on CIFAR10")
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.trainset = lib.cifar.CIFAR10(root='./data',
                                          train=True,
                                          download=False,
                                          transform=transform_train,
                                          randomize_labels=randomize_labels)

        self.num_training_images = len(self.trainset)

        self.unnormalizer = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
                                       transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                            std = [ 1., 1., 1. ])
                                      ])

class MNIST(Dataset):
    def __init__(self, device, split_size, test_batch_size):

        super(MNIST, self).__init__(split_size)

        self.model = MNISTNet().to(device)

        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        # Testing set
        self.testloader = torch.utils.data.DataLoader(
            lib.mnist.MNIST('../data', train=False, download=True,
                           transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size=test_batch_size, shuffle=False, num_workers=2)

        # Training set
        self.trainset = lib.mnist.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        self.unnormalizer = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.3081]),
                                       transforms.Normalize(mean = [ -0.1307],
                                                            std = [ 1., 1., 1. ])
                                      ])

        self.num_training_images = len(self.trainset)

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

from torch.utils.data import ConcatDataset
from torchvision import datasets
class IndexedSVHN(datasets.SVHN):
    def __getitem__(self, index):
        retval = super(IndexedSVHN, self).__getitem__(index)
        return retval + (index,)

class SVHN(Dataset):
    def __init__(self, model, test_batch_size, split_size, augment):

        super(SVHN, self).__init__(split_size)

        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        self.model = model
        self.augment = augment

        # Testing set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = IndexedSVHN(root='./svhn_data',
                              split='test',
                              download=False,
                              transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=test_batch_size,
                                                      shuffle=False,
                                                      num_workers=0)

        # Training set
        if self.augment:
            print("Performing data augmentation on SVHN")
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            print("Not performing data augmentation on SVHN")
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.trainset1 = IndexedSVHN(root='./svhn_data',
                                     split='train',
                                     download=False,
                                     transform=transform_train)
        self.trainset2 = IndexedSVHN(root='./svhn_data',
                                     split='extra',
                                     download=False,
                                     transform=transform_train)
        self.trainset = ConcatDataset([self.trainset1, self.trainset2])

        self.num_training_images = len(self.trainset)

        self.unnormalizer = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
                                       transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                            std = [ 1., 1., 1. ])
                                      ])

class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        retval = super(IndexedImageFolder, self).__getitem__(index)
        return retval + (index,)

class ImageNet(Dataset):
    def __init__(self, model, test_batch_size, traindir, valdir, split_size):

        super(ImageNet, self).__init__(split_size)

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.classes = [str(i) for i in range(1000)]
        self.model = model
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])


        # Testing set
        testset = IndexedImageFolder(valdir, transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize]))
        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=test_batch_size,
                                                      shuffle=False,
                                                      num_workers=2)

        # Training set
        print("Performing data augmentation on ImageNet")
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.trainset = IndexedImageFolder(traindir,
                                           transform_train)
        self.num_training_images = len(self.trainset)
        print(self.num_training_images)
        self.unnormalizer = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ])
                                               ])
