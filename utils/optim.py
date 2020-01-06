from bisect import bisect_right
import torch
from .radam import RAdam 

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        #print("base_lr:{} warmup_factor:{} gamma:{} ".format(self.base_lrs[0], warmup_factor, self.gamma))
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

def make_optimizer(cfg, model, num_gpus=1):
    params = []
    optim_name = cfg['OPTIM']
    for key, value in model.named_parameters():
        # filter not requires_grad
        if not value.requires_grad:
            continue
        lr = cfg['LR'] * num_gpus 
        # linear scaling rule
        weight_decay = cfg['WEIGHT_DECAY'] 
        if "bias" in key:
            #lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            lr = cfg['LR'] * cfg['BIAS_LR_FACTOR']
            #weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            weight_decay = cfg['WEIGHT_DECAY_BIAS'] 
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if optim_name == 'SGD':
        optimizer = getattr(torch.optim, optim_name)(params, momentum=0.9)
    elif optim_name == 'RAdam':
        optimizer = RAdam(params)
    else:
        optimizer = getattr(torch.optim, optim_name)(params)
    return optimizer

