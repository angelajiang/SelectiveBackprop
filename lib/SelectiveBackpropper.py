from . import backproppers
from . import calculators
from . import fp_selectors
from . import loggers
from . import selectors
import time
import torch
import torch.nn as nn
from . import trainer as trainer

start_time_seconds = time.time()

class SelectiveBackpropper:

    def __init__(self,
                 model,
                 optimizer,
                 prob_pow,
                 batch_size,
                 lr_sched,
                 num_classes,
                 num_training_images,
                 forwardlr,
                 strategy,
                 calculator="relative",
                 fp_selector_type="alwayson",
                 staleness=2):

        ## Hardcoded params
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert device == "cuda"
        self.num_training_images = num_training_images
        num_images_to_prime = self.num_training_images
        #num_images_to_prime = 0

        log_interval = 1
        sampling_min = 0
        sampling_max = 1
        max_history_len = 1024
        prob_loss_fn = nn.CrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss
        sample_size = 0 # only needed for topk, lowk

        # Params for resuming from checkpoint
        start_epoch = 0
        start_num_backpropped = 0
        start_num_skipped = 0

        self.selector = None
        self.fp_selector = None
        if strategy == "nofilter":
            self.backpropper = backproppers.SamplingBackpropper(device,
                                                                model,
                                                                optimizer,
                                                                loss_fn)
            self.trainer = trainer.NoFilterTrainer(device,
                                                   model,
                                                   self.backpropper,
                                                   batch_size,
                                                   loss_fn,
                                                   lr_schedule=lr_sched,
                                                   forwardlr=forwardlr)
        else:
            probability_calculator = calculators.get_probability_calculator(calculator,
                                                                            device,
                                                                            prob_loss_fn,
                                                                            sampling_min,
                                                                            sampling_max,
                                                                            num_classes,
                                                                            max_history_len,
                                                                            prob_pow)
            self.selector = selectors.get_selector("sampling",
                                                   probability_calculator,
                                                   num_images_to_prime,
                                                   sample_size)

            self.fp_selector = fp_selectors.get_selector(fp_selector_type,
                                                         num_images_to_prime,
                                                         staleness=staleness)

            self.backpropper = backproppers.SamplingBackpropper(device,
                                                                model,
                                                                optimizer,
                                                                loss_fn)

            self.trainer = trainer.MemoizedTrainer(device,
                                                   model,
                                                   self.selector,
                                                   self.fp_selector,
                                                   self.backpropper,
                                                   batch_size,
                                                   loss_fn,
                                                   lr_schedule=lr_sched,
                                                   forwardlr=forwardlr)

        self.logger = loggers.Logger(log_interval = log_interval,
                                     epoch=start_epoch,
                                     num_backpropped=start_num_backpropped,
                                     num_skipped=start_num_skipped,
                                     start_time_seconds = start_time_seconds)

        self.trainer.on_backward_pass(self.logger.handle_backward_batch)
        self.trainer.on_forward_pass(self.logger.handle_forward_batch)

    def next_epoch(self):
        self.logger.next_epoch()

    def next_partition(self):
        if self.selector is not None:
            self.selector.next_partition(self.num_training_images)
        if self.fp_selector is not None:
            self.fp_selector.next_partition(self.num_training_images)
        self.logger.next_partition()
