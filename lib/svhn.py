from torch.utils.data import ConcatDataset
from torchvision import datasets

class SVHN(datasets.SVHN):
    def __getitem__(self, index):
        retval = super(SVHN, self).__getitem__(index)
        return retval + (index,)
