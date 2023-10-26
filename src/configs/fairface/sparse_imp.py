import cooper
import ml_collections as mlc

import src.cmp as _cmp
import src.configs as _configs
import src.configs.basic as basic_configs
import src.sparse as sparse

MLC_PH = mlc.config_dict.config_dict.placeholder


def fetch_cmp_class(mitigation_method: str):
    cmp_assets = {
        "naive_fine_tuning": _cmp.BaselineProblem,
        "ceag": _cmp.UniformAccuracyGapProblem,
        "equalized_loss": _cmp.EqualizedLossProblem,
    }
    return cmp_assets[mitigation_method]


def get_config(config_string):
    dataset_name, mitigation_method = config_string.split("-")
    assert dataset_name == "fairface"

    # Get the common config for this dataset set the sparsity method
    config = getattr(_configs, f"{dataset_name}_common").build_common_config()
    config.sparsity_method = "imp"
    config.task_id = f"{config.data.dataset_name}_{config.sparsity_method}"

    config.model.conv_layer = sparse.MaskedConv2d
    config.model.norm_layer = sparse.MaskedBatchNorm2d

    config.mitigation_method = mitigation_method
    config.train.cmp_class = fetch_cmp_class(mitigation_method)

    if config.mitigation_method == "no_mitigation":
        config.optim.constrained_optimizer_class = cooper.optim.UnconstrainedOptimizer
        config.data.val_split_ratio = 0
    else:
        config.optim.constrained_optimizer_class = cooper.optim.AlternatingDualPrimalOptimizer
        config.train.cmp_init_kwargs.apply_mitigation_while_pruning = False
        config.train.cmp_init_kwargs.buffer_memory = 40
        config.optim.dual = basic_configs.make_dual_optimizer_config()
        config.optim.dual.optimizer = "SGD"
        config.optim.dual.lr = 1e-4
        config.optim.dual.kwargs.ema_gamma = 0.0
        config.train.cmp_init_kwargs.detach_model_constraint_contribution = False
        config.data.val_split_ratio = 0.0
        if config.mitigation_method == "uniform_accuracy_gap":
            config.train.cmp_init_kwargs.tolerance = 0.01
        if config.mitigation_method == "equalized_loss":
            config.train.cmp_init_kwargs.abs_equality_constraint = True

    config.train.epochs = 47  # 15 (fine-tuning) + 32
    config.data.train_batch_size = 256
    config.data.test_batch_size = 256
    config.data.val_batch_size = 256
    config.data.val_split_seed = 0

    config.optim.primal.optimizer = "SGD"
    config.optim.primal.lr = 0.001
    config.optim.primal.lr_scheduler_milestones = (0.9,)
    config.optim.primal.lr_scheduler_gamma = 0.1
    config.optim.primal.kwargs.momentum = 0.95
    config.optim.primal.kwargs.nesterov = True
    config.train.cmp_init_kwargs.weight_decay = 1e-2

    config.sparsity = basic_configs.make_sparsity_config()
    config.sparsity.has_scheduler = True
    config.sparsity.last_pruning_epoch = 15
    config.sparsity.sparsity_final = 0.99
    config.sparsity.sparsity_initial = 0
    config.sparsity.pruning_frequency = 1
    config.sparsity.init_pruning_epoch = 0
    config.sparsity.sparsity_type = "unstructured"

    return config
