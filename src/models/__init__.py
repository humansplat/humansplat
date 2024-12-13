from typing import *

from torch import optim
from torch.nn import Parameter
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import LRScheduler

from .human_rec import HumanRec
from .human_splat import HumanSplat
from .lgm import LGM
from .noise_lgm import NoiseLGM
from .sv3d_p import SV3D
# from .eschernet import EscherNet
# from .escherlgm import EscherLGM
from .sv3dlgm import SV3DLGM
from .sv3dlgm2 import SV3DLGM2
from .sv3dlgm3 import SV3DLGM3


def get_optimizer(name: str, params: Parameter, **kwargs) -> Optimizer:
    if name == "adamw":
        return optim.AdamW(params=params, **kwargs)
    else:
        raise NotImplementedError(f"Not implemented optimizer: {name}")


def get_lr_scheduler(name: str, optimizer: Optimizer, **kwargs) -> LRScheduler:
    if name == "one_cycle":
        return lr_scheduler.OneCycleLR(optimizer=optimizer, **kwargs)
    elif name == "constant":
        return lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda _: 1)
    else:
        raise NotImplementedError(f"Not implemented lr scheduler: {name}")
