import cooper
import ml_collections as mlc

import src.cmp as _cmp
import src.configs as _configs
import src.configs.basic as basic_configs
import src.sparse as sparse

MLC_PH = mlc.config_dict.config_dict.placeholder


def fetch_cmp_class(mitigation_method: str):
    cmp_assets = {"no_mitigation": _cmp.BaselineProblem}
    return cmp_assets[mitigation_method]


def get_config(config_string):

    dataset_name, mitigation_method = config_string.split("-")

    config = getattr(_configs, f"{dataset_name}_common").build_common_config()
    config.sparsity_method = "mp"
    config.task_id = f"{config.data.dataset_name}_{config.sparsity_method}"

    if config.model.model_name == "MLP":
        config.model.linear_layer = sparse.MaskedLinear
    else:
        config.model.conv_layer = sparse.MaskedConv2d
        config.model.norm_layer = sparse.MaskedBatchNorm2d

    config.mitigation_method = mitigation_method
    config.train.cmp_class = fetch_cmp_class(mitigation_method)

    if config.mitigation_method == "no_mitigation":
        config.optim.constrained_optimizer_class = cooper.optim.UnconstrainedOptimizer
    else:
        config.optim.constrained_optimizer_class = cooper.optim.AlternatingDualPrimalOptimizer
        config.train.cmp_init_kwargs.tolerance = 0.1
        config.optim.dual = basic_configs.make_dual_optimizer_config()
        config.optim.dual.optimizer = "SGD"
        config.optim.dual.lr = 1e-2
        if config.mitigation_method == "equalized_loss":
            config.train.cmp_init_kwargs.abs_equality_constraint = False

    # Perform magnitude pruning only once, at the beginning of training after
    # having loaded the pretrained model.
    # IMPORTANT: sparsity_initial and sparsity_final MUST match
    config.sparsity = basic_configs.make_sparsity_config()
    config.sparsity.has_scheduler = True
    config.sparsity.last_pruning_epoch = 0
    config.sparsity.sparsity_final = MLC_PH(float)
    config.sparsity.sparsity_initial = MLC_PH(float)
    config.sparsity.pruning_frequency = 1
    config.sparsity.init_pruning_epoch = 0
    config.sparsity.sparsity_type = "unstructured"

    return config
