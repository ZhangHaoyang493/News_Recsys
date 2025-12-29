# ...existing code...
import math
from torch.optim.lr_scheduler import _LRScheduler
    

class CosinDecayLR(_LRScheduler):
    def __init__(self, optimizer, lrs = [1e-3, 1e-5], milestones = [2000, 5000]):
        self.lrs = lrs
        self.milestones = milestones

        assert len(lrs) == 2, "Currently only support 2 lrs for CosinDecayLR"
        assert len(self.lrs) == len(self.milestones), "lrs length must be equal to milestones length"

        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.milestones[0]:
            return [self.lrs[0] for _ in self.optimizer.param_groups]
        elif self.last_epoch >= self.milestones[-1]:
            return [self.lrs[-1] for _ in self.optimizer.param_groups]
        elif self.last_epoch >= self.milestones[0] and self.last_epoch < self.milestones[1]:
            # 使用cosin decay，如果step在milestones[0]和milestones[1]之间,则进行cosin decay
            step_in_decay = self.last_epoch - self.milestones[0]
            total_decay_steps = self.milestones[1] - self.milestones[0]
            progress = step_in_decay / max(1, total_decay_steps)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.lrs[1] + (self.lrs[0] - self.lrs[1]) * cosine_factor
            return [lr for _ in self.optimizer.param_groups]
