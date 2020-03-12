## Accelerating Deep Learning by Focusing on the Biggest Losers

SelectiveBackprop accelerates training by dynamically prioritizing useful
examples with high loss. SelectiveBackprop can reduce training time by up to
3.5x faster than standard training with SGD. [Our
paper](https://arxiv.org/abs/1910.00762) describes the system in detail.

## Prerequisites

* Nvidia GPU with CUDA
* PyTorch 0.4.0 or newer
* python2.7
* numpy, torch, torchvision, tqdm, scipy, sklearn

## Example Commands

### Train using SelectiveBackprop
```
$ python examples/train.py --strategy sb --dataset <cifar10|cifar100|svhn>

Files already downloaded and verified
Files already downloaded and verified
...
...
============ EPOCH 0 ============
FPs: 50000 / 50000
BPs: 50000 / 50000
Test loss: 0.010174
Test acc: 56.380
Time elapsed: 41.24s
============ EPOCH 1 ============
FPs: 100000 / 100000
BPs: 62366 / 100000
Test loss: 0.008055
Test acc: 68.580
Time elapsed: 57.07s
...
...
```

### Train using SelectiveBackprop with staleness
```
$ python examples/train.py --strategy sb --dataset <cifar10|cifar100|svhn> --fp_selector stale
```

## Example Usage

To use SelectiveBackprop with your own PyTorch training code, first create a
SelctiveBackpropper object and call train with a typical PyTorch dataloader.

```python
'''
SelectiveBackpropper
   :param cnn: pytorch network
   :param cnn optimizer: pytorch optimizer
   :param prob_pow: int for beta dictating SB's selectivity (higher is more selective)
   :param batch_size: int for batch size
   :param lr_sched: string for path to learning rate schedule or None
   :param num_classes: int for number of classes
   :param num_examples: int for number of training examples
   :param forwardlr: boolean to set learning rate changes based on epochs
   :param strategy: string in ['nofilter', 'sb']
   :param calculator: string in ['relative', 'random', 'hybrid']
   :param fp_selector: string in ['alwayson', 'stale']
   :param staleness: int for number of epochs to use cached losses in StaleSB
'''

sb = SelectiveBackpropper(cnn,
                          cnn_optimizer,
                          prob_pow,
                          batch_size,
                          lr_sched,
                          num_classes,
                          num_examples,
                          forwardlr,
                          strategy,
                          calculator,
                          fp_selector,
                          staleness)

while True:
    sb.trainer.train(train_loader)
    sb.next_epoch()
    sb.next_partition()

```

