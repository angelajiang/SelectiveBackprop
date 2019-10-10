import json
import numpy as np
import torch
import torch.nn as nn
import forwardproppers
import sb_util


class ExampleAndMetadata(object):
    def __init__(self, example, metadata):
        self.example = example
        self.metadata = metadata

class Example(object):
    # TODO: Add ExampleCollection class
    def __init__(self,
                 loss=None,
                 output=None,
                 softmax_output=None,
                 target=None,
                 datum=None,
                 image_id=None,
                 select_probability=None):
        if loss is not None:
            self.loss = loss.detach().cpu()
        if output is not None:
            self.output = output.detach().cpu()
        if softmax_output is not None:
            self.softmax_output = softmax_output.detach().cpu()
        self.forward_select = True
        self.target = target.detach().cpu()
        self.datum = datum.detach().cpu()
        self.image_id = image_id
        self.select_probability = select_probability
        self.backpropped_loss = None   # Populated after backprop

    def __str__(self):
        string = "Image {}\n\ndatum:{}\ntarget:{}\nsp:{}\n".format(self.image_id,
                                                                   self.datum,
                                                                   self.target,
                                                                   self.select_probability)
        if hasattr(self, 'loss'):
            string += "loss:{}\n".format(self.loss)
        if hasattr(self, 'output'):
            string += "output:{}\n".format(self.output)
        if hasattr(self, 'softmax_output'):
            string += "softmax_output:{}\n".format(self.softmax_output)

        return string


    @property
    def predicted(self):
        _, predicted = self.softmax_output.max(0)
        return predicted

    @property
    def is_correct(self):
        return self.predicted.eq(self.target)


class Trainer(object):
    def __init__(self,
                 device,
                 net,
                 selector,
                 backpropper,
                 batch_size,
                 loss_fn,
                 max_num_backprops=float('inf'),
                 lr_schedule=None,
                 forwardlr=False):
        self.device = device
        self.net = net
        self.selector = selector
        self.backpropper = backpropper
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.backprop_queue = []
        self.forward_pass_handlers = []
        self.backward_pass_handlers = []
        self.global_num_backpropped = 0
        self.global_num_forwards = 0
        self.global_num_analyzed = 0
        self.forwardlr = forwardlr
        self.max_num_backprops = max_num_backprops
        self.on_backward_pass(self.update_num_backpropped)
        self.on_forward_pass(self.update_num_forwards)
        self.on_forward_pass(self.update_num_analyzed)
        self.example_metadata = {}
        if lr_schedule:
            self.load_lr_schedule(lr_schedule)
            self.on_backward_pass(self.update_learning_rate)

    def update_num_backpropped(self, batch):
        self.global_num_backpropped += sum([1 for em in batch if em.example.select])

    def update_num_forwards(self, batch):
        self.global_num_forwards += sum([1 for em in batch if em.example.forward_select])

    def update_num_analyzed(self, batch):
        self.global_num_analyzed += len(batch)

    def on_forward_pass(self, handler):
        self.forward_pass_handlers.append(handler)

    def on_backward_pass(self, handler):
        self.backward_pass_handlers.append(handler)

    def emit_forward_pass(self, batch):
        for handler in self.forward_pass_handlers:
            handler(batch)

    def emit_backward_pass(self, batch):
        for handler in self.backward_pass_handlers:
            handler(batch)

    # TODO move to a LRScheduler object or to backpropper
    def load_lr_schedule(self, schedule_path):
        with open(schedule_path, "r") as f:
            data = json.load(f)
        self.lr_schedule = {}
        for k in data:
            self.lr_schedule[int(k)] = data[k]

    def set_learning_rate(self, lr):
        print("Setting learning rate to {} at {} backprops".format(lr,
                                                                   self.global_num_backpropped))
        for param_group in self.backpropper.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def counter(self):
        if self.forwardlr:
            counter = self.global_num_analyzed
        else:
            counter = self.global_num_backpropped
        return counter

    def update_learning_rate(self, batch):
        for start_num_backprop in reversed(sorted(self.lr_schedule)):
            lr = self.lr_schedule[start_num_backprop]
            if self.counter >= start_num_backprop:
                if self.backpropper.optimizer.param_groups[0]['lr'] is not lr:
                    self.set_learning_rate(lr)
                break

    @property
    def stopped(self):
        return self.counter >= self.max_num_backprops

    def train(self, trainloader):
        for i, batch in enumerate(trainloader):
            if self.stopped: break
            if i == len(trainloader) - 1:
                self.train_batch(batch, final=True)
            else:
                self.train_batch(batch, final=False)

    def train_batch(self, batch, final):
        pass

    def forward_pass(self, data, targets, image_ids):
        pass

    def get_batch(self, final):
        num_images_to_backprop = 0
        for index, em in enumerate(self.backprop_queue):
            num_images_to_backprop += int(em.example.select)
            if num_images_to_backprop == self.batch_size:
                # Note: includes item that should and shouldn't be backpropped
                backprop_batch = self.backprop_queue[:index+1]
                self.backprop_queue = self.backprop_queue[index+1:]
                return backprop_batch
        if final:
            def get_num_to_backprop(batch):
                return sum([1 for em in batch if em.example.select])
            backprop_batch = self.backprop_queue
            self.backprop_queue = []
            if get_num_to_backprop(backprop_batch) == 0:
                return None
            return backprop_batch
        return None

    def create_example_batch(self, data, targets, image_ids):
        batch = []
        for target, datum, image_id in zip(targets, data, image_ids):
            image_id = image_id.item()
            if image_id not in self.example_metadata:
                self.example_metadata[image_id] = {"epochs_since_update": 0}
            example = Example(target=target, datum=datum, image_id=image_id, select_probability=1)
            example.select = True
            batch.append(ExampleAndMetadata(example, self.example_metadata[image_id]))
        return batch

class MemoizedTrainer(Trainer):
    def __init__(self,
                 device,
                 net,
                 selector,
                 fp_selector,
                 backpropper,
                 batch_size,
                 loss_fn,
                 max_num_backprops=float('inf'),
                 lr_schedule=None,
                 forwardlr=False):

        super(MemoizedTrainer, self).__init__(device,
                                net,
                                selector,
                                backpropper,
                                batch_size,
                                loss_fn,
                                max_num_backprops,
                                lr_schedule,
                                forwardlr)

        self.fp_selector = fp_selector
        self.forward_queue = []
        self.forward_batch_size = batch_size
        self.forwardpropper = forwardproppers.CutoutForwardpropper(device,
                                                                   net,
                                                                   loss_fn)

    def train_batch(self, batch, final):
        EMs = self.create_example_batch(*batch)
        batch_marked_for_fp = self.fp_selector.mark(EMs)
        self.forward_queue += batch_marked_for_fp
        batch_to_fp = self.get_forward_batch(final)
        if batch_to_fp:
            forward_pass_batch = self.forwardpropper.forward_pass(batch_to_fp)
            annotated_forward_batch = self.selector.mark(forward_pass_batch)
            self.emit_forward_pass(annotated_forward_batch)
            self.backprop_queue += annotated_forward_batch
            backprop_batch = self.get_batch(final)
            if backprop_batch:
                annotated_backward_batch = self.backpropper.backward_pass(backprop_batch)
                self.emit_backward_pass(annotated_backward_batch)

    def get_forward_batch(self, final):
        num_images_to_fp = 0
        max_queue_size = self.forward_batch_size * 4
        for index, em in enumerate(self.forward_queue):
            num_images_to_fp += int(em.example.forward_select)
            if num_images_to_fp == self.forward_batch_size:
                # Note: includes item that should and shouldn't be forward propped
                forward_batch = self.forward_queue[:index+1]
                self.forward_queue = self.forward_queue[index+1:]
                return forward_batch
        if final or len(self.forward_queue) > max_queue_size:
            forward_batch = self.forward_queue
            self.forward_queue = []
            return forward_batch
        return None

class NoFilterTrainer(Trainer):
    def __init__(self,
                 device,
                 net,
                 backpropper,
                 batch_size,
                 loss_fn,
                 max_num_backprops=float('inf'),
                 lr_schedule=None,
                 forwardlr=False):

        super(NoFilterTrainer, self).__init__(device,
                                net,
                                None,
                                backpropper,
                                batch_size,
                                loss_fn,
                                max_num_backprops,
                                lr_schedule,
                                forwardlr)

    def train_batch(self, batch, final):
        annotated_forward_batch = self.create_example_batch(*batch)
        self.backprop_queue += annotated_forward_batch
        backprop_batch = self.get_batch(final)
        if backprop_batch:
            annotated_backward_batch = self.backpropper.backward_pass(backprop_batch)
            self.emit_backward_pass(annotated_backward_batch)
            self.emit_forward_pass(annotated_backward_batch)

