import ml_collections as mlc

import src.configs.basic as basic_configs

MLC_PH = mlc.config_dict.config_dict.placeholder


def build_common_config():
    # Construct a skeleton config
    config = basic_configs.build_basic_config()

    config.data.dataset_name = "utkface"
    config.data.augment = False
    config.data.dataset_kwargs = mlc.ConfigDict({"target_attribute": "gender", "protected_attributes": ("race",)})

    config.model.model_name = "MobileNet_V2"
    config.model.input_shape = (3, 200, 200)
    config.model.is_first_conv_dense = True
    config.model.is_last_fc_dense = True

    # ------------- These configs should be set at the experiment level -------------

    config.task_id = MLC_PH(str)
    config.train.epochs = MLC_PH(int)
    config.data.train_batch_size = MLC_PH(int)
    config.data.test_batch_size = MLC_PH(int)
    config.data.val_batch_size = MLC_PH(int)
    config.data.val_split_ratio = MLC_PH(float)
    config.data.val_split_seed = MLC_PH(int)

    config.optim.constrained_optimizer_class = MLC_PH(type)
    config.optim.primal.optimizer = MLC_PH(str)
    config.optim.primal.lr = MLC_PH(float)
    config.optim.primal.lr_scheduler_milestones = MLC_PH(tuple)
    config.optim.primal.lr_scheduler_gamma = MLC_PH(float)

    # -------------------------------------------------------------------------------

    # Populate slurm_exec, *default* seed, and checkpoint_dir
    # The seed should be overwritten by the grid search config
    basic_configs.fill_default_slurm_train_config_(config.train)

    return config
