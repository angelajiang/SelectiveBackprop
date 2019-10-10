from scipy import stats
import collections
import math
import numpy as np
import torch
import torch.nn as nn
from random import shuffle
import lib.predictors
import lib.hist

# TODO: Transform into base classes
def get_probability_calculator(calculator_type,
                               device,
                               prob_loss_fn,
                               sampling_min,
                               sampling_max,
                               num_classes,
                               max_history_len,
                               prob_pow):
    ## Setup Trainer:ProbabilityCalculator ##
    if prob_pow:
        prob_transform = lambda x: torch.pow(x, prob_pow)
    else:
        prob_transform = None

    if calculator_type == "vanilla":
        probability_calculator = BatchedSelectProbabilityCalculator(sampling_min,
                                                                    sampling_max,
                                                                    num_classes,
                                                                    device,
                                                                    prob_transform=prob_transform)
    elif calculator_type == "alwayson":
        probability_calculator = BatchedAlwaysOnProbabilityCalculator()
    elif calculator_type == "relative":
        probability_calculator = BatchedRelativeProbabilityCalculator(device,
                                                                      sampling_min,
                                                                      max_history_len,
                                                                      prob_pow)
    elif calculator_type == "random":
        probability_calculator = BatchedRandomProbabilityCalculator(device,
                                                                    sampling_min,
                                                                    prob_pow)
    else:
        print("Use prob-strategy in {vanilla,alwayson,relative,random}")
        exit()
    return probability_calculator

class BatchedRandomProbabilityCalculator(object):
    def __init__(self, device, sampling_min, beta):
        self.device = device
        self.sampling_min = sampling_min
        self.beta = beta

    def calculate_probability(self,):
        random_percentile = np.random.uniform(0, 1)
        return math.pow(random_percentile, self.beta)

    def get_probability(self, examples_and_metadata):
        probs = [max(self.sampling_min, self.calculate_probability()) for i in range(len(examples_and_metadata))]
        return probs

class BatchedRelativeProbabilityCalculator(object):
    def __init__(self, device, sampling_min, history_length, beta):
        self.device = device
        self.historical_losses = lib.hist.UnboundedHistogram(history_length) #collections.deque(maxlen=history_length)
        self.sampling_min = sampling_min
        self.beta = beta

    def update_history(self, losses):
        for loss in losses:
            self.historical_losses.append(loss)

    def calculate_probability(self, loss):
        percentile = self.historical_losses.percentile_of_score(loss)
        return math.pow(percentile / 100., self.beta)

    def get_probability(self, examples_and_metadata):
        losses = [em.example.loss for em in examples_and_metadata]
        self.update_history(losses)
        probs = [max(self.sampling_min, self.calculate_probability(loss)) for loss in losses]
        return probs

class BatchedSelectProbabilityCalculator(object):
    def __init__(self, sampling_min, sampling_max, num_classes, device, prob_transform=None):
        self.sampling_min = sampling_min
        self.sampling_max = sampling_max
        self.num_classes = num_classes
        self.device = device
        if prob_transform:
            self.prob_transform = prob_transform
        else:
            self.prob_transform  = lambda x: x

    def get_probability(self, examples_and_metadata):
        ts = [em.example.target for em in examples_and_metadata]
        ss = [em.example.softmax_output for e in examples_and_metadata]
        targets = torch.stack(ts, dim=0).cpu().numpy()
        softmax_outputs = torch.stack(ss, dim=0).cpu().numpy()
        classes = np.diag(np.arange(self.num_classes))
        target_tensor = classes[targets]
        l2_dist = np.linalg.norm(target_tensor - softmax_outputs, axis=1)
        l2_dist = np.square(l2_dist)
        base = np.clip(self.prob_transform(l2_dist), self.sampling_min, self.sampling_max)
        return np.clip(base, self.sampling_min, self.sampling_max)

class BatchedAlwaysOnProbabilityCalculator(object):
    def get_probability(self, examples_and_metadata):
        return [1] * len(examples_and_metadata)


################### Deprecated Calculators #########################

class HistoricalProbabilityCalculator(object):
    def __init__(self, calculator_type, std_multiplier=None, bp_probability_calculator=None):
        self.type = calculator_type
        if self.type == "alwayson":
            self.calculator = BatchedAlwaysOnProbabilityCalculator()
        elif self.type == "mean":
            self.calculator = MeanHistoricalCalculator()
        elif self.type == "gp":
            self.calculator = GPHistoricalCalculator(std_multiplier,
                                                     bp_probability_calculator)
        elif self.type == "rto":
            self.calculator = RTOHistoricalCalculator(std_multiplier,
                                                      bp_probability_calculator)

    def get_probability(self, example):
        return self.calculator.get_probability(example)

class VanillaHistoricalCalculator(object):
    def get_probability(self, example):
        return 1

class MeanHistoricalCalculator(VanillaHistoricalCalculator):
    def __init__(self):
        self.history = {}
        self.history_length = 5

    def update_history(self, example):
        if example.image_id not in self.history.keys():
            self.history[example.image_id] = collections.deque(maxlen=self.history_length)
            # First example won't have a loss yet
        else:
            previous_sp = example.get_sp(False)
            if previous_sp:
                if len(self.history[example.image_id]) == 0:
                    self.history[example.image_id].append(previous_sp)
                else:
                    # Check that loss has been updated
                    if previous_sp != self.history[example.image_id][-1]:
                        self.history[example.image_id].append(previous_sp)

    def get_probability(self, example):
        self.update_history(example)
        hist = self.history[example.image_id]
        if not example.get_select(True):
            return 1
        if len(hist) >= self.history_length:
            #print(hist)
            if all(h < 0.001 for h in hist):
                return 0
        return 1


class ProportionalProbabiltyCalculator(object):
    def __init__(self, sampling_min, sampling_max, num_classes, device,
                 prob_transform=None):
        self.sampling_min = sampling_min
        self.sampling_max = sampling_max
        self.num_classes = num_classes
        self.device = device

        self.theoretical_max = 2

        # prob_transform should be a function f where f(x) <= 1
        if prob_transform:
            self.prob_transform = prob_transform
        else:
            self.prob_transform  = lambda x: x

    def get_probability(self, example):
        target = example.target
        softmax_output = example.softmax_output
        target_tensor = example.hot_encoded_target
        l2_dist = torch.dist(target_tensor.to(self.device), softmax_output)
        l2_dist *= l2_dist
        prob = l2_dist / float(self.theoretical_max)
        transformed_prob = self.prob_transform(prob)
        clamped_prob = torch.clamp(transformed_prob,
                                   min=self.sampling_min)
        return clamped_prob.item()


class GPHistoricalCalculator(VanillaHistoricalCalculator):
    def __init__(self, std_multiplier, bp_selector):
        self.history = {}
        self.gps = {}
        self.xs = {}
        self.std_multiplier = std_multiplier
        self.bp_selector = bp_selector
        self.min_history = 0
        self.timeout_multiplier = 1
        self.retrain_every = 10

    def update_history(self, example):
        if example.image_id not in self.history.keys():
            # First example won't have a loss yet
            self.history[example.image_id] = []
            predictor = lib.predictors.GPPredictor()
            self.gps[example.image_id] = predictor
            self.xs[example.image_id] = 0
        else:
            previous_sp = example.get_sp(False)
            if len(self.history[example.image_id]) == 0 or previous_sp != self.history[example.image_id][-1]:
                self.history[example.image_id].append(previous_sp)
                ys = self.history[example.image_id]
                if len(ys) >= self.min_history and len(ys) % self.retrain_every == 0:
                    predictor = self.gps[example.image_id]
                    X = np.array(range(len(ys))).reshape(-1, 1)
                    predictor.update(X, ys)
        self.xs[example.image_id] += 1


    def select(self, y, std):
        draw = np.random.uniform(0, 1)
        if self.timeout_multiplier * (y + (self.std_multiplier*std)) > draw:
            self.timeout_multiplier += 10
            return 1, draw
        else:
            self.timeout_multiplier = 1
            return 0, draw

    def get_probability(self, example):
        self.update_history(example)
        predictor = self.gps[example.image_id]
        hist = self.history[example.image_id]
        if len(hist) < self.min_history:
            return 1

        x = self.xs[example.image_id]
        X = np.array([x]).reshape(-1, 1)
        y, std = predictor.predict(x)
        is_selected, draw = self.select(y, std)
        example.fp_draw = draw
        if is_selected == 0:
            if hasattr(example, "loss"):
                self.bp_selector.update_history(example.loss)
        return is_selected

    '''
    def get_probability(self, example):
        self.update_history(example)
        predictor = self.gps[example.image_id]
        hist = self.history[example.image_id]
        if not example.get_select(True):
            return 1
        if len(hist) >= self.min_history:
            x = self.xs[example.image_id]
            y, _ = predictor.predict(x)
            return y
        return 1
    '''


class RTOHistoricalCalculator(VanillaHistoricalCalculator):
    def __init__(self, std_multiplier, bp_selector):
        self.history = {}
        self.gps = {}
        self.xs = {}
        self.std_multiplier = std_multiplier
        self.bp_selector = bp_selector
        self.min_history = 0
        self.timeout_multiplier = 1
        print("std_multiplier {}".format(self.std_multiplier))

    def update_history(self, example):
        if example.image_id not in self.history.keys():
            # First example won't have a loss yet
            self.history[example.image_id] = []
            predictor = lib.predictors.RTOPredictor()
            self.gps[example.image_id] = predictor
        else:
            predictor = self.gps[example.image_id]
            previous_sp = example.get_sp(False)
            if len(self.history[example.image_id]) == 0 or previous_sp != self.history[example.image_id][-1]:
                self.history[example.image_id].append(previous_sp)
                predictor.update(None, previous_sp)

    def select(self, y, std):
        draw = np.random.uniform(0, 1)
        if self.timeout_multiplier * (y + (self.std_multiplier*std)) > draw:
            self.timeout_multiplier += 2
            return 1, draw
        else:
            self.timeout_multiplier = 1
            return 0, draw

    def get_probability(self, example):
        self.update_history(example)
        predictor = self.gps[example.image_id]
        hist = self.history[example.image_id]
        if len(self.history[example.image_id]) < self.min_history:
            return 1

        y, std = predictor.predict(None)
        is_selected, draw =  self.select(y, std)
        example.fp_draw = draw
        if is_selected == 0:
            if hasattr(example, "loss"):
                self.bp_selector.update_history(example.loss.item())
        return is_selected


