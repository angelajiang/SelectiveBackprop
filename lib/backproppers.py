import torch
import time
import torch.nn as nn

class PrimedBackpropper(object):
    def __init__(self, initial, final, initial_num_images):
        self.initial = initial
        self.final = final
        self.initial_num_images = initial_num_images
        self.num_trained = 0

    def next_partition(self, partition_size):
        self.num_trained += partition_size

    def get_backpropper(self):
        return self.initial if self.num_trained < self.initial_num_images else self.final

    @property
    def optimizer(self):
        return self.initial.optimizer if self.num_trained < self.initial_num_images else self.final.optimizer

    def backward_pass(self, *args, **kwargs):
        return self.get_backpropper().backward_pass(*args, **kwargs)


class SamplingBackpropper(object):

    def __init__(self, device, net, optimizer, loss_fn):
        self.optimizer = optimizer
        self.net = net
        self.device = device
        self.loss_fn = loss_fn

    def _get_chosen_examples(self, batch):
        return [em for em in batch if em.example.select]

    def _get_chosen_data_tensor(self, batch):
        chosen_data = [em.example.datum for em in batch]
        return torch.stack(chosen_data)

    def _get_chosen_targets_tensor(self, batch):
        chosen_targets = [em.example.target for em in batch]
        return torch.stack(chosen_targets)

    def backward_pass(self, batch):
        self.net.train()

        chosen_batch = self._get_chosen_examples(batch)
        data = self._get_chosen_data_tensor(chosen_batch).to(self.device)
        targets = self._get_chosen_targets_tensor(chosen_batch).to(self.device)

        # Run forward pass
        outputs = self.net(data) 
        losses = self.loss_fn(reduce=False)(outputs, targets)
        softmax_outputs = nn.Softmax()(outputs)             # OPT: not necessary when logging is off
        _, predicted = outputs.max(1)
        is_corrects = predicted.eq(targets)

        # Scale each loss by image-specific select probs
        #losses = torch.div(losses, probabilities.to(self.device))

        # Reduce loss
        loss = losses.mean()

        # Run backwards pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Add for logging selected loss
        for em, loss, is_correct in zip(chosen_batch,
                                        losses,
                                        is_corrects):
            em.example.loss = loss.item()
            em.example.correct = is_correct.item()
            em.metadata["loss"] = em.example.loss

        return batch

class ReweightedBackpropper(SamplingBackpropper):

    def __init__(self, device, net, optimizer, loss_fn):
        super(ReweightedBackpropper, self).__init__(device,
                                                    net,
                                                    optimizer,
                                                    loss_fn)

    def _get_chosen_weights_tensor(self, batch):
        chosen_weights = [torch.tensor(em.example.weight, dtype=torch.float) for em in batch]
        return torch.stack(chosen_weights)

    def backward_pass(self, batch):
        self.net.train()

        chosen_batch = self._get_chosen_examples(batch)
        data = self._get_chosen_data_tensor(chosen_batch).to(self.device)
        targets = self._get_chosen_targets_tensor(chosen_batch).to(self.device)
        weights = self._get_chosen_weights_tensor(chosen_batch).to(self.device)

        # Run forward pass
        outputs = self.net(data) 
        losses = self.loss_fn(reduce=False)(outputs, targets)
        softmax_outputs = nn.Softmax()(outputs)             # OPT: not necessary when logging is off
        _, predicted = outputs.max(1)
        is_corrects = predicted.eq(targets)

        # Scale each loss by image-specific select probs
        losses = torch.mul(losses, weights)

        # Reduce loss
        loss = losses.mean()

        # Run backwards pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Add for logging selected loss
        for em, loss, is_correct in zip(chosen_batch,
                                        losses,
                                        is_corrects):
            em.example.loss = loss.item()
            em.example.correct = is_correct.item()
            em.metadata["loss"] = em.example.loss

        return batch

class AlwaysOnBackpropper(object):

    def __init__(self, device, net, optimizer, loss_fn):
        super(SamplingBackpropper, self).__init__(device,
                                                  net,
                                                  optimizer,
                                                  loss_fn)

    def _get_chosen_examples(self, batch):
        return batch

