import numpy as np
import pickle
import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class ImageWriter(object):
    def __init__(self, data_dir, dataset, unnormalizer):
        self.data_dir = data_dir
        self.dataset = dataset
        self.unnormalizer = unnormalizer
        self.init_data()

    def init_data(self):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        self.output_dir = os.path.join(self.data_dir, "{}_by_id".format(self.dataset))
        print(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def write_partition(self, partition):
        to_pil = torchvision.transforms.ToPILImage()
        for elem in partition:
            img_tensor = elem[0].cpu()
            unnormalized = self.unnormalizer(img_tensor)
            img = to_pil(unnormalized)

            img_id = elem[2]
            img_file = os.path.join(self.output_dir, "image-{}.png".format(img_id))

            img.save(img_file, 'PNG')


class ProbabilityByImageLogger(object):
    def __init__(self, pickle_dir, pickle_prefix, max_num_images=None):
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix
        self.init_data()
        self.max_num_images = max_num_images
        self.probabilities = {}
        self.backward_selects = {}
        self.forward_selects = {}
        self.losses = {}

    def next_epoch(self):
        self.write()

    def init_data(self):
        # Store frequency of each image getting backpropped
        data_pickle_dir = os.path.join(self.pickle_dir, "probabilities_by_image")
        self.probabilities_pickle_file = os.path.join(data_pickle_dir,
                                                      "{}_probabilities".format(self.pickle_prefix))
        self.backward_selects_pickle_file = os.path.join(data_pickle_dir,
                                                "{}_selects".format(self.pickle_prefix))
        self.forward_selects_pickle_file = os.path.join(data_pickle_dir,
                                                "{}_forwardselects".format(self.pickle_prefix))
        self.losses_pickle_file = os.path.join(data_pickle_dir,
                                               "{}_losses".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, image_ids, probabilities, backward_selects, forward_selects, losses):
        for image_id, probability in zip(image_ids, probabilities):
            if image_id not in self.probabilities.keys():
                if self.max_num_images:
                    if image_id >= self.max_num_images:
                        continue
                self.probabilities[image_id] = []
            self.probabilities[image_id].append(probability)

        for image_id, is_selected in zip(image_ids, backward_selects):
            if image_id not in self.backward_selects.keys():
                if self.max_num_images:
                    if image_id >= self.max_num_images:
                        continue
                self.backward_selects[image_id] = []
            self.backward_selects[image_id].append(int(is_selected))

        for image_id, is_selected in zip(image_ids, forward_selects):
            if image_id not in self.forward_selects.keys():
                if self.max_num_images:
                    if image_id >= self.max_num_images:
                        continue
                self.forward_selects[image_id] = []
            self.forward_selects[image_id].append(int(is_selected))

        for image_id, loss in zip(image_ids, losses):
            if image_id not in self.losses.keys():
                if self.max_num_images:
                    if image_id >= self.max_num_images:
                        continue
                self.losses[image_id] = []
            self.losses[image_id].append(loss)

    def handle_backward_batch(self, batch):
        ids = [em.example.image_id for em in batch]
        probabilities = [em.example.get_sp(False) for em in batch]
        backward_selects = [em.example.get_select(False) for em in batch]
        forward_selects = [em.example.get_select(True) for em in batch]
        losses = [em.example.loss for em in batch]
        self.update_data(ids, probabilities, backward_selects, forward_selects, losses)

    def write(self):
        latest_file = "{}.pickle".format(self.probabilities_pickle_file)
        with open(latest_file, "wb") as handle:
            print(latest_file)
            pickle.dump(self.probabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)
        latest_file = "{}.pickle".format(self.backward_selects_pickle_file)
        with open(latest_file, "wb") as handle:
            print(latest_file)
            pickle.dump(self.backward_selects, handle, protocol=pickle.HIGHEST_PROTOCOL)
        latest_file = "{}.pickle".format(self.forward_selects_pickle_file)
        with open(latest_file, "wb") as handle:
            print(latest_file)
            pickle.dump(self.forward_selects, handle, protocol=pickle.HIGHEST_PROTOCOL)


class ImageIdHistLogger(object):

    def __init__(self, pickle_dir, pickle_prefix, num_images, log_interval):
        self.current_epoch = 0
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix
        self.log_interval = log_interval
        self.init_data(num_images)

    def next_epoch(self):
        self.write()
        self.current_epoch += 1

    def init_data(self, num_images):
        # Store frequency of each image getting backpropped
        keys = range(num_images)
        self.data = dict(zip(keys, [0] * len(keys)))
        data_pickle_dir = os.path.join(self.pickle_dir, "image_id_hist")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                                 "{}_images_hist".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, image_ids):
        for chosen_id in image_ids:
            self.data[chosen_id] += 1

    def handle_backward_batch(self, batch):
        ids = [em.example.image_id.item() for em in batch if em.example.select]
        self.update_data(ids)

    def write(self):
        latest_file = "{}.pickle".format(self.data_pickle_file)
        with open(latest_file, "wb") as handle:
            print(latest_file)
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        epoch_file = "{}.epoch_{}.pickle".format(self.data_pickle_file,
                                                 self.current_epoch)
        if self.current_epoch % self.log_interval == 0:
            with open(epoch_file, "wb") as handle:
                print(epoch_file)
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class LossesByEpochLogger(object):

    def __init__(self, pickle_dir, pickle_prefix, log_frequency):
        self.current_epoch = 0
        self.pickle_dir = pickle_dir
        self.log_frequency = log_frequency
        self.pickle_prefix = pickle_prefix
        self.init_data()

    def next_epoch(self):
        self.write()
        self.current_epoch += 1
        self.data = []

    def init_data(self):
        # Store frequency of each image getting backpropped
        self.data = []
        data_pickle_dir = os.path.join(self.pickle_dir, "losses")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                                 "{}_losses".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, losses):
        self.data += losses

    def handle_backward_batch(self, batch):
        losses = [em.example.loss.item() for em in batch]
        self.update_data(losses)

    def write(self):
        epoch_file = "{}.epoch_{}.pickle".format(self.data_pickle_file,
                                                 self.current_epoch)
        if self.current_epoch % self.log_frequency == 0:
            with open(epoch_file, "wb") as handle:
                print(epoch_file)
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class LossesByImageLogger(object):
    def __init__(self, pickle_dir, pickle_prefix, max_num_images=None):
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix
        self.init_data()
        self.max_num_images = max_num_images
        self.data = {}

    def next_epoch(self):
        self.write()

    def init_data(self):
        # Store frequency of each image getting backpropped
        data_pickle_dir = os.path.join(self.pickle_dir, "losses_by_image")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                             "{}_losses".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, image_ids, losses):
        for image_id, loss in zip(image_ids, losses):
            if image_id not in self.data.keys():
                if self.max_num_images:
                    if image_id >= self.max_num_images:
                        continue
                self.data[image_id] = []
            self.data[image_id].append(loss)

    def handle_backward_batch(self, batch):
        ids = [em.example.image_id for em in batch]
        losses = [em.example.loss for em in batch]
        self.update_data(ids, losses)

    def write(self):
        latest_file = "{}.pickle".format(self.data_pickle_file)
        with open(latest_file, "wb") as handle:
            print(latest_file)
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class VariancesByImageLogger(object):
    def __init__(self, pickle_dir, pickle_prefix, max_num_images=None):
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix
        self.init_data()
        self.max_num_images = max_num_images
        self.data = {}

    def next_epoch(self):
        self.write()

    def init_data(self):
        # Store frequency of each image getting backpropped
        data_pickle_dir = os.path.join(self.pickle_dir, "variance_by_image")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                             "{}_variances".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, image_ids, losses):
        for image_id, loss in zip(image_ids, losses):
            if image_id not in self.data.keys():
                if self.max_num_images:
                    if image_id >= self.max_num_images:
                        continue
                self.data[image_id] = []
            self.data[image_id].append(loss)

    def handle_backward_batch(self, batch):
        ids = [em.example.image_id for em in batch]
        losses = [em.example.loss for em in batch]
        self.update_data(ids, losses)

    def write(self):
        variance = {}
        for image_id in self.data.keys():
            variance[image_id] = np.var(self.data[image_id])
        latest_file = "{}.pickle".format(self.data_pickle_file)
        with open(latest_file, "wb") as handle:
            print(latest_file)
            pickle.dump(variance, handle, protocol=pickle.HIGHEST_PROTOCOL)

class VariancesByEpochLogger(object):
    def __init__(self, pickle_dir, pickle_prefix, log_frequency):
        self.current_epoch = 0
        self.pickle_dir = pickle_dir
        self.log_frequency = log_frequency
        self.pickle_prefix = pickle_prefix
        self.init_data()

    def next_epoch(self):
        self.write()
        self.current_epoch += 1
        self.data = []

    def init_data(self):
        # Store frequency of each image getting backpropped
        self.data = []
        data_pickle_dir = os.path.join(self.pickle_dir, "variance_by_epoch")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                                 "{}_variances".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, variance):
        self.data += [variance]

    def handle_backward_batch(self, batch):
        losses = [em.example.loss.item() for em in batch]
        variance = np.var(losses)
        self.update_data(variance)

    def write(self):
        epoch_file = "{}.epoch_{}.pickle".format(self.data_pickle_file,
                                                 self.current_epoch)
        if self.current_epoch % self.log_frequency == 0:
            with open(epoch_file, "wb") as handle:
                print(epoch_file)
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

class VariancesByAverageProbabilityByImageLogger(object):
    def __init__(self, pickle_dir, pickle_prefix, max_num_images=None):
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix
        self.init_data()
        self.max_num_images = max_num_images
        self.data = {"losses": {}, "probabilities": {}}

    def next_epoch(self):
        self.write()

    def init_data(self):
        # Store frequency of each image getting backpropped
        data_pickle_dir = os.path.join(self.pickle_dir, "variance_by_avg_prob")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                             "{}_variances".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, image_ids, probabilities, losses):
        for image_id, prob, loss in zip(image_ids, probabilities, losses):
            if image_id not in self.data["losses"].keys():
                if self.max_num_images:
                    if image_id >= self.max_num_images:
                        continue
                self.data["losses"][image_id] = []
                self.data["probabilities"][image_id] = []
            self.data["losses"][image_id].append(loss)
            self.data["probabilities"][image_id].append(prob)

    def handle_backward_batch(self, batch):
        ids = [em.example.image_id for em in batch]
        losses = [em.example.loss for em in batch]
        probabilities = [em.example.select_probability for em in batch]
        self.update_data(ids, probabilities, losses)

    def write(self):
        out = {}
        for image_id in self.data["losses"].keys():
            var = np.var(self.data["losses"][image_id])
            avg_prob = np.average(self.data["probabilities"][image_id])
            out[image_id] = (avg_prob, var)
        latest_file = "{}.pickle".format(self.data_pickle_file)
        with open(latest_file, "wb") as handle:
            print(latest_file)
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

class Logger(object):

    def __init__(self, log_interval=1, epoch=0, num_backpropped=0, num_skipped=0, num_skipped_fp=0, num_forwards=0, start_time_seconds=None):
        self.current_epoch = epoch
        self.current_batch = 0
        self.log_interval = log_interval

        self.global_num_backpropped = num_backpropped
        self.global_num_skipped = num_skipped
        self.global_num_skipped_fp = num_skipped_fp
        self.global_num_forwards= num_forwards

        self.partition_loss = 0
        self.partition_backpropped_loss = 0
        self.partition_num_backpropped = 0
        self.partition_num_skipped = 0
        self.partition_num_correct = 0

        self.debug = False

        if start_time_seconds is None:
            self.start_time_seconds = time.time()
        else:
            self.start_time_seconds = start_time_seconds

    def next_epoch(self):
        self.current_epoch += 1

    @property
    def partition_seen(self):
        return self.partition_num_backpropped + self.partition_num_skipped

    @property
    def average_partition_loss(self):
        return self.partition_loss / float(self.partition_seen)

    @property
    def average_partition_backpropped_loss(self):
        return self.partition_backpropped_loss / float(self.partition_num_backpropped)

    @property
    def partition_accuracy(self):
        return 100. * self.partition_num_correct / self.partition_seen

    @property
    def train_debug(self):
        return 'train_debug,{},{},{},{},{:.6f},{},{:.6f},{:4f}'.format(
            self.current_epoch,
            self.global_num_backpropped,
            self.global_num_skipped,
            self.global_num_skipped_fp,
            self.average_partition_backpropped_loss,
            self.global_num_forwards,
            self.partition_accuracy,
            time.time() - self.start_time_seconds)

    def next_partition(self):
        self.partition_loss = 0
        self.partition_backpropped_loss = 0
        self.partition_num_backpropped = 0
        self.partition_num_skipped = 0
        self.partition_num_correct = 0

    def handle_forward_batch(self, batch):
        # Populate batch_stats
        # self.partition_loss += sum([example.loss for em in batch])
        num_skipped_fp = sum([int(not em.example.forward_select) for em in batch])
        self.global_num_skipped_fp += num_skipped_fp
        self.global_num_forwards += sum([int(em.example.forward_select) for em in batch])

    def handle_backward_batch(self, batch):

        self.current_batch += 1

        num_backpropped = sum([int(em.example.select) for em in batch])
        num_skipped = sum([int(not em.example.select) for em in batch])
        self.global_num_backpropped += num_backpropped
        self.global_num_skipped += num_skipped

        if self.debug:
            self.partition_num_backpropped += num_backpropped
            self.partition_num_skipped += num_skipped
            self.partition_backpropped_loss += sum([em.example.backpropped_loss
                                                    for em in batch
                                                    if em.example.backpropped_loss])
            chosen = [em for em in batch if em.example.select]
            self.partition_num_correct += sum([1 for em in chosen if em.example.correct])

            self.write()

    def write(self):
        if self.current_batch % self.log_interval == 0:
            print(self.train_debug)


