import collections

class BoundedHistogram:
    def __init__(self, max_history):
        self.max_history =  max_history
        self.num_slots = 1000
        self.history = collections.deque(maxlen=self.max_history)
        self.histogram = [0] * self.num_slots
        self.count = 0

    def append(self, value):
        if len(self.history) == self.max_history:
            assert value <= 1 and value >= 0 
            popped_value = self.history[0]
            slot = int(popped_value * self.num_slots) % self.num_slots
            assert self.histogram[slot] > 0
            self.histogram[slot] -= 1
            self.count -= 1

        self.history.append(value)
        self.count += 1
        slot = int(value * self.num_slots) % self.num_slots
        self.histogram[slot] += 1

    def percentile_of_score(self, score):
        slot = int(score * self.num_slots) % self.num_slots
        summed_count = sum(self.histogram[:slot])
        return summed_count * 100. / self.count

class UnboundedHistogram:
    def __init__(self, max_history):
        self.max_history =  max_history
        self.history = collections.deque(maxlen=self.max_history)

    def append(self, value):
        self.history.append(value)

    def get_count(self, it, score):
        count = 0
        for i in it:
            if i < score:
                count += 1
        return count

    def percentile_of_score(self, score):
        num_lower_scores = self.get_count(self.history, score)
        return num_lower_scores * 100. / len(self.history)

def test():

    import datetime
    import timeit
    import random

    def get_epochtime_us():
        return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000000)

    '''
    ah = BoundedHistogram(4)
    ah.append(0.1)
    ah.append(0.2)
    ah.append(0.3)
    ah.append(0.4)
    percentile = ah.percentile_of_score(0.3)
    assert percentile == 50

    ah.append(0.5)
    percentile = ah.percentile_of_score(0.3)
    assert percentile == 25
    '''

    ah = UnboundedHistogram(4)
    ah.append(1)
    ah.append(2)
    ah.append(3)
    ah.append(4)
    percentile = ah.percentile_of_score(3)
    #print(ah.history, 3, percentile)

    ah.append(5)
    percentile = ah.percentile_of_score(3)
    #print(ah.history, 3, percentile)
    num_trials = 1000000

    '''
    t = timeit.Timer('ah.append(1)',
                     setup='from __main__ import UnboundedHistogram; ah = UnboundedHistogram(4);',
                     )
    seconds_per_call = t.timeit(number = num_trials) / float(num_trials)
    print("{} appends, {} us".format(num_trials, seconds_per_call / 1000000.))
    '''

    l1 = range(num_trials)
    start = get_epochtime_us()
    for i in l1:
        ah.append(i)
    end = get_epochtime_us()
    print("{} appends, {} us".format(num_trials, float(end - start) / num_trials))

    l = range(num_trials)
    random.shuffle(l)
    start = get_epochtime_us()
    for i in l:
        ah.percentile_of_score(i)
    end = get_epochtime_us()
    print("{} get_percentile_of_scores, {} us".format(num_trials, float(end - start) / num_trials))

#if __name__ == "__main__":
#    test()





