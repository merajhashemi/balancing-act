import os

import ml_collections as mlc
import torch

MLC_PH = mlc.config_dict.config_dict.placeholder


def make_data_config():
    _config = mlc.ConfigDict()
    _config.dataset_name = MLC_PH(str)
    _config.train_batch_size = MLC_PH(int)
    _config.test_batch_size = MLC_PH(int)
    _config.val_batch_size = MLC_PH(int)
    _config.val_split_ratio = MLC_PH(float)
    _config.val_split_seed = MLC_PH(int)
    _config.augment = MLC_PH(bool)
    _config.dataset_kwargs = mlc.ConfigDict()

    return _config


def make_train_config():
    _config = mlc.ConfigDict()
    _config.cmp_class = MLC_PH(type)
    _config.slurm_exec = MLC_PH(bool)
    _config.seed = MLC_PH(int)
    _config.ensure_reproducibility = True
    _config.epochs = MLC_PH(int)
    _config.checkpoint_dir = MLC_PH(str)
    _config.pretrained_model_runid = MLC_PH(str)

    _config.cmp_init_kwargs = mlc.ConfigDict()
    _config.cmp_init_kwargs.label_smoothing = MLC_PH(float)
    _config.cmp_init_kwargs.weight_decay = MLC_PH(float)
    _config.cmp_init_kwargs.tolerance = MLC_PH(float)
    _config.cmp_init_kwargs.intersectional = MLC_PH(bool)
    _config.cmp_init_kwargs.buffer_memory = MLC_PH(int)

    _config.dtype = torch.float32

    return _config


def fill_default_slurm_train_config_(train_config):
    train_config.slurm_exec = False
    train_config.seed = 0
    train_config.checkpoint_dir = os.environ["CHECKPOINT_DIR"]


def make_sparsity_config():
    _config = mlc.ConfigDict()
    _config.has_scheduler = MLC_PH(bool)
    _config.last_pruning_epoch = MLC_PH(int)
    _config.sparsity_final = MLC_PH(float)
    _config.pruning_frequency = MLC_PH(int)
    _config.sparsity_initial = MLC_PH(float)
    _config.init_pruning_epoch = MLC_PH(int)
    _config.sparsity_type = MLC_PH(str)

    return _config


def make_model_config():
    _config = mlc.ConfigDict()
    _config.model_name = MLC_PH(str)
    _config.input_shape = MLC_PH(tuple)
    _config.num_classes = MLC_PH(int)
    _config.conv_layer = MLC_PH(torch.nn.Module)
    _config.norm_layer = MLC_PH(torch.nn.Module)
    _config.is_first_conv_dense = MLC_PH(bool)
    _config.is_last_fc_dense = MLC_PH(bool)

    return _config


def make_plain_optimizer_config():
    _config = mlc.ConfigDict()
    _config.optimizer = MLC_PH(str)
    _config.kwargs = mlc.ConfigDict()
    _config.lr = MLC_PH(float)
    _config.lr_scheduler_milestones = MLC_PH(tuple)
    _config.lr_scheduler_gamma = MLC_PH(float)

    return _config


def make_dual_optimizer_config():
    _config = mlc.ConfigDict()
    _config.optimizer = MLC_PH(str)
    _config.lr = MLC_PH(float)
    _config.do_dual_restarts = MLC_PH(bool)
    _config.kwargs = mlc.ConfigDict()

    return _config


def build_basic_config():
    config = mlc.ConfigDict()

    # Populate top-level configs which are common to all experiments
    config.task_id = MLC_PH(str)
    config.sparsity_method = MLC_PH(str)
    config.mitigation_method = MLC_PH(str)
    config.data = make_data_config()
    config.train = make_train_config()
    config.model = make_model_config()
    config.optim = mlc.ConfigDict()
    config.optim.constrained_optimizer_class = MLC_PH(type)

    config.optim.primal = make_plain_optimizer_config()

    # Fixed defaults for logging across all experiments
    config.logging = mlc.ConfigDict()
    config.logging.log_level = "INFO"
    config.logging.wandb_mode = "online"
    config.logging.plot_log_freq = 15

    return config
