class SamplingBackpropper(object):

    def __init__(self, schedule_path):
        load_lr_schedule(schedule_path)

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

    def update_learning_rate(self, batch):
        for start_num_backprop in reversed(sorted(self.lr_schedule)):
            lr = self.lr_schedule[start_num_backprop]
            if self.global_num_backpropped >= start_num_backprop:
                if self.backpropper.optimizer.param_groups[0]['lr'] is not lr:
                    self.set_learning_rate(lr)
                break
