import math


class LearningRateScheduler(object):
    def __init__(self, initial_lr, number_batches, scheduling_type='fixed'):
        self.scheduling_functions = {'fixed': {'set': self._fixed_set,
                                               'call': self._fixed_call},
                                     'step': {'set': self._step_set,
                                              'call': self._step_call},
                                     'exp': {'set': self._exp_set,
                                             'call': self._exp_call},
                                     'inv': {'set': self._inv_set,
                                             'call': self._inv_call},
                                     'sigmoid': {'set': self._sigmoid_set,
                                                 'call': self._sigmoid_call},
                                     }
        self.initial_lr = initial_lr
        if scheduling_type not in self.scheduling_functions:
            raise KeyError('The learning rate scheduler {} is not implemented.'.format(scheduling_type))
        self.scheduling_type = scheduling_type
        self.number_batches = number_batches

    def update_lr(self, optimizer, iteration):
        if self.scheduling_type == 'fixed':
            return optimizer

        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = self.scheduling_functions[self.scheduling_type]['call'](iteration, old_lr)
            param_group['lr'] = new_lr

        return optimizer

    def set(self, **kwargs):
        self.scheduling_functions[self.scheduling_type]['set'](**kwargs)

    def _fixed_set(self):
        pass

    def _fixed_call(self, iteration, current_lr):
        return current_lr

    def _step_set(self, step=10, gamma=0.1):
        self.step = step
        self.gamma = gamma

    def _step_call(self, iteration, current_lr):
        return current_lr * self.gamma**(iteration // self.step)

    def _exp_set(self, gamma=0.1):
        self.gamma = gamma

    def _exp_call(self, iteration, current_lr):
        return current_lr * self.gamma**(iteration // self.number_batches)

    def _inv_set(self, gamma=0, power=0):
        self.gamma = gamma
        self.power = power

    def _inv_call(self, iteration, current_lr):
        return current_lr * (1 + self.gamma * (iteration // self.number_batches)) ** (- self.power)

    def _sigmoid_set(self, gamma, stepsize):
        self.gamma = gamma
        self.stepsize = stepsize

    def _sigmoid_call(self, iteration, current_lr):
        return current_lr * (1 / (1 + math.exp(- self.gamma * ((iteration // self.number_batches) - self.stepsize))))


if __name__ == '__main__':
    lrs = LearningRateScheduler(0.1, scheduling_type='sigmoid')
    lrs.set(gamma=0.1, stepsize=13)
    for i in range(1000):
        print(lrs.update_lr(None, i))
