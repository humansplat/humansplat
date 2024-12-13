import os
from argparse import Namespace
from typing import *

from omegaconf import DictConfig, OmegaConf
from torch.nn import Module

if not OmegaConf.has_resolver("neg"):
    OmegaConf.register_new_resolver("neg", lambda a: -a)
if not OmegaConf.has_resolver("half"):
    OmegaConf.register_new_resolver("half", lambda a: a / 2)
if not OmegaConf.has_resolver("shsdim"):
    OmegaConf.register_new_resolver(
        "shsdim", lambda sh_degree: (sh_degree + 1) ** 2 * 3
    )

from src.options import Options


def get_configs(yaml_path: str, cli_configs: List[str] = [], **kwargs) -> DictConfig:
    yaml_configs = OmegaConf.load(yaml_path)
    cli_configs = OmegaConf.from_cli(cli_configs)

    configs = OmegaConf.merge(yaml_configs, cli_configs, kwargs)
    OmegaConf.resolve(configs)  # resolve ${...} placeholders
    return configs


def save_experiment_params(
    args: Namespace, configs: DictConfig, opt: Options, save_dir: str
) -> Dict[str, Any]:
    params = OmegaConf.merge(configs, {k: str(v) for k, v in vars(args).items()})
    params = OmegaConf.merge(params, OmegaConf.create(vars(opt)))
    OmegaConf.save(params, os.path.join(save_dir, "params.yaml"))
    return dict(params)


def save_model_architecture(model: Module, save_dir: str) -> None:
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    message = (
        f"Number of trainable / all parameters: {num_trainable_params} / {num_params}\n\n"
        + f"Model architecture:\n{model}"
    )

    with open(os.path.join(save_dir, "model.txt"), "w") as f:
        f.write(message)
