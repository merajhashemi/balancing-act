import cooper
import torch

import src.cmp as _cmp
import src.configs as _configs


def get_config(config_string):
    dataset_name, mitigation_method = config_string.split("-")  # example: utkface-no_mitigation
    assert dataset_name == "utkface"

    # Get the common config for this dataset set the sparsity method
    config = getattr(_configs, f"{dataset_name}_common").build_common_config()
    config.sparsity_method = "dense"
    config.task_id = f"{config.data.dataset_name}_{config.sparsity_method}"

    config.model.conv_layer = torch.nn.Conv2d
    config.model.norm_layer = torch.nn.BatchNorm2d

    config.mitigation_method = mitigation_method
    config.train.cmp_class = _cmp.BaselineProblem

    # Set the best configs based on runid 3470937
    config.train.epochs = 50
    config.data.train_batch_size = 128
    config.data.test_batch_size = 256
    config.data.val_batch_size = 256
    config.data.val_split_ratio = 0.0
    config.data.val_split_seed = 0

    config.optim.constrained_optimizer_class = cooper.optim.UnconstrainedOptimizer
    config.optim.primal.optimizer = "Adam"
    config.optim.primal.lr = 0.001
    config.optim.primal.lr_scheduler_milestones = (0.6, 0.8, 0.9)
    config.optim.primal.lr_scheduler_gamma = 0.1
    config.train.cmp_init_kwargs.weight_decay = 1e-4

    return config
