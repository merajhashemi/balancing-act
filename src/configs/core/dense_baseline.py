import cooper
import ml_collections as mlc
import torch

import src.cmp as _cmp
import src.configs as _configs
import src.configs.basic as basic_configs

MLC_PH = mlc.config_dict.config_dict.placeholder


def fetch_cmp_class(mitigation_method: str):
    cmp_assets = {"no_mitigation": _cmp.BaselineProblem}
    return cmp_assets[mitigation_method]


def get_config(config_string):

    dataset_name, mitigation_method = config_string.split("-")

    config = getattr(_configs, f"{dataset_name}_common").build_common_config()
    config.sparsity_method = "dense"
    config.task_id = f"{config.data.dataset_name}_{config.sparsity_method}"

    if config.model.model_name == "MLP":
        config.model.linear_layer = torch.nn.Linear
    else:
        config.model.conv_layer = torch.nn.Conv2d
        config.model.norm_layer = torch.nn.BatchNorm2d

    config.mitigation_method = mitigation_method
    config.train.cmp_class = fetch_cmp_class(mitigation_method)
    config.optim.constrained_optimizer_class = cooper.optim.UnconstrainedOptimizer

    if config.mitigation_method != "no_mitigation":
        config.train.cmp_init_kwargs.ema_gamma = MLC_PH(float)
        config.optim.dual = basic_configs.make_dual_optimizer_config()
        config.optim.dual.optimizer = "SGD"
        config.optim.dual.lr = 1e-2
    if config.mitigation_method == "bounded_acc":
        config.train.cmp_init_kwargs.tol = MLC_PH(float)

    return config
