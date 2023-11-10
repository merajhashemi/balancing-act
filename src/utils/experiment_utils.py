import copy
import logging
import os
import pickle
import random
import tempfile
from pathlib import Path
from typing import Optional, Union

import cooper
import numpy as np
import torch
import torch.nn as nn

import src.datasets as datasets
import src.models as models
import src.models.utils as model_utils
import src.sparse as sparse
import src.utils.train_utils as train_utils
import wandb

logger = logging.getLogger(__name__)


def set_seed(seed: int, ensure_reproducibility: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enable TF32 on Ampere GPUs to speed up training see:
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if ensure_reproducibility:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def filter_out_nones_from_dict(_dict):
    """Remove all entries in a dictionary whose value is None."""
    if _dict is None:
        return None
    else:
        return {k: v for k, v in _dict.items() if v is not None}


def construct_wandb_run(config) -> tuple[wandb.run, Path, int, int]:
    """Build a WandB run from the config."""

    if config.train.slurm_exec and "SLURM_JOB_ID" not in os.environ.keys():
        raise ValueError("config.train.slurm_exec is True, but 'SLURM_JOB_ID' not found in environment variables.")

    custom_run_id = os.environ["SLURM_JOB_ID"] if config.train.slurm_exec else None
    run_name = os.environ["SLURM_JOB_ID"] if config.train.slurm_exec else config.task_id

    run = wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        mode=config.logging.wandb_mode,
        dir=os.environ["WANDB_DIR"] if "WANDB_DIR" in os.environ.keys() else None,
        name=run_name,
        id=custom_run_id,
        resume="allow",
    )
    wandb.config.update(config.to_dict(), allow_val_change=True)

    run_checkpoint_dir = Path(config.train.checkpoint_dir) / run.id
    wandb.config.update({"run_checkpoint_dir": run_checkpoint_dir}, allow_val_change=True)

    if run.resumed:
        if not run_checkpoint_dir.exists():
            raise RuntimeError(f"Trying to resume run {run.id} but checkpoint dir {run_checkpoint_dir} does not exist!")

        run_metadata = torch.load(run_checkpoint_dir / "metadata.pt")
        # When checkpoint at the end of an epoch, we already add 1 to "epoch" to
        # continue from the next epoch. So, when resuming, no need to add 1 again.
        start_from_epoch = run_metadata["epoch"]
        start_from_step = run_metadata["step"] + 1

        logging.info(f"Resuming run {run.id} from epoch {start_from_epoch}")
    else:
        logging.info(f"Starting new run {run.id} from epoch 0")
        start_from_epoch, start_from_step = 0, 0

    # Define metrics for custom x-axis
    wandb.define_metric("_step")
    wandb.define_metric("_epoch")
    wandb.define_metric("train/batch/*", step_metric="_step")
    wandb.define_metric("model_stats/epoch/*", step_metric="_epoch")
    wandb.define_metric("train/epoch/*", step_metric="_epoch")
    wandb.define_metric("train_eval/epoch/*", step_metric="_epoch")
    wandb.define_metric("val/epoch/*", step_metric="_epoch")

    return run, run_checkpoint_dir, start_from_epoch, start_from_step


def load_dataset(config):
    """Builds dataset."""

    return datasets.load_dataset(
        dataset_name=config.data.dataset_name,
        augment=config.data.augment,
        train_batch_size=config.data.train_batch_size,
        test_batch_size=config.data.test_batch_size,
        val_batch_size=config.data.val_batch_size,
        val_split_ratio=config.data.val_split_ratio,
        val_split_seed=config.data.val_split_seed,
        dataset_kwargs=config.data.dataset_kwargs.to_dict(),
        slurm_exec=config.train.slurm_exec,
    )


def construct_models(
    config,
    num_classes: int,
    input_shape: tuple[int, int, int],
    run_checkpoint_dir: Optional[Path] = None,
    device: Union[torch.device, str] = "cpu",
) -> tuple[nn.Module, Union[nn.Module, None]]:
    """Constructs a model from the model sub-config."""

    assert config.model.num_classes == num_classes
    assert config.model.input_shape == input_shape

    try:
        model_class = models.__dict__[config.model.model_name]
    except KeyError:
        raise ValueError(f"Model {config.model.model_name} not found.")

    kwargs = config.model.to_dict()
    kwargs.pop("model_name")  # model_name is not a valid argument to the model

    # Create a "trainable" model
    model = model_class(**kwargs)

    cmp_class_name = config.train.cmp_class.__name__
    if "Baseline" not in cmp_class_name and config.train.pretrained_model_runid is None:
        raise ValueError(
            f"{cmp_class_name} requires a pre-trained model but config.train.pretrained_model_runid was not provided."
        )

    if config.train.pretrained_model_runid is not None:
        artifact_dict = load_artifact_from_wandb(
            project=os.environ["WANDB_PROJECT"],
            entity=os.environ["WANDB_ENTITY"],
            artifact_name="best_model.pt",
            run_id=config.train.pretrained_model_runid,
            device=device,
        )

        # The pretrained *dense* model is stored in the "best_model" artifact of the run.
        pretrained_state_dict = artifact_dict[config.train.pretrained_model_runid]["best_model"]
        model_utils.populate_dummy_extra_states_(model, pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)
        # Return trainable model and copy of the model to be used as a baseline
        baseline_model = copy.deepcopy(model)

    else:
        baseline_model = None

        if config.data.dataset_name == "fairface":
            model = model_utils.load_fairface_model(
                model_placeholder=model,
                checkpoint_dir=config.train.checkpoint_dir,
                target_attribute=config.data.dataset_kwargs["target_attribute"],
            )
        elif config.data.dataset_name == "cifar100":
            model = model_utils.load_cifar100_model(model_placeholder=model, checkpoint_dir=config.train.checkpoint_dir)
        elif config.data.dataset_name == "utkface":
            model = model_utils.load_utkface_model(
                model_placeholder=model,
                checkpoint_dir=config.train.checkpoint_dir,
                target_attribute=config.data.dataset_kwargs["target_attribute"],
            )
        else:
            raise ValueError(f"Unknown dataset {config.data.dataset_name}")

    if run_checkpoint_dir is not None and run_checkpoint_dir.exists():
        logger.info(f"Loading model.pt from existing checkpoint in directory {run_checkpoint_dir}.")
        model.load_state_dict(torch.load(run_checkpoint_dir / "model.pt", map_location=device))

    model.to(device)
    if baseline_model is not None:
        baseline_model.to(device)

    return model, baseline_model


def construct_sparsity_scheduler(config):
    # If experiment does not make use of a sparsity scheduler, return None
    if not hasattr(config, "sparsity") or (not config.sparsity.has_scheduler):
        return None

    if config.sparsity_method == "mp" and (config.sparsity.sparsity_final != config.sparsity.sparsity_initial):
        raise ValueError("One shot magnitude pruning expects matching initial and final sparsity for the scheduler.")

    return sparse.SparsityScheduler(
        last_pruning_epoch=config.sparsity.last_pruning_epoch,
        sparsity_final=config.sparsity.sparsity_final,
        sparsity_initial=config.sparsity.sparsity_initial,
        init_pruning_epoch=config.sparsity.init_pruning_epoch,
        pruning_frequency=config.sparsity.pruning_frequency,
        sparsity_type=config.sparsity.sparsity_type,
    )


def construct_constrained_optimizer(model, cmp, run_checkpoint_dir, config):
    # Construct primal optimizers and schedulers
    primal_optimizers = []
    lr_schedulers = {}

    primal_config = config.optim.primal
    optimizer = torch.optim.__dict__[primal_config.optimizer](
        model.parameters(), lr=primal_config.lr, **filter_out_nones_from_dict(primal_config.kwargs)
    )

    if hasattr(primal_config, "lr_scheduler_milestones") and primal_config.lr_scheduler_milestones is not None:
        # config's milestones are specified as a fraction of the total number of epochs
        milestones = [int(milestone * config.train.epochs) for milestone in primal_config.lr_scheduler_milestones]

        lr_schedulers["params"] = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=primal_config.lr_scheduler_gamma
        )

    primal_optimizers.append(optimizer)
    constrained_optimizer_kwargs = {"primal_optimizers": primal_optimizers}

    # If needed, construct optimizer for dual variables
    if cmp.num_constraints > 0:
        dual_config = config.optim.dual

        dual_variables = []
        for constraint_group in train_utils.ensure_iterable(cmp.constraint_group):
            dual_variables += list(constraint_group.multiplier.parameters())

        dual_optim_kwargs = dual_config.kwargs.to_dict()
        if "ema_gamma" in dual_optim_kwargs:
            assert dual_config.optimizer == "SGD"
            dual_optim_kwargs["dampening"] = dual_optim_kwargs["momentum"] = dual_optim_kwargs.pop("ema_gamma")
        # Setting maximize=True as the dual variables aim to maximize the Lagrangian
        dual_optimizer = torch.optim.__dict__[dual_config.optimizer](
            dual_variables, lr=dual_config.lr, maximize=True, **dual_optim_kwargs
        )

        constrained_optimizer_kwargs["dual_optimizers"] = dual_optimizer
        constrained_optimizer_kwargs["constraint_groups"] = cmp.constraint_group

    if run_checkpoint_dir.exists():
        # If we are resuming from a checkpoint, load the Cooper optimizer state, This
        # loads the dual variables and the state for the primal and dual optimizers.
        cooper_optimizer_state_dict = torch.load(run_checkpoint_dir / "constrained_optimizer.pt")
        cooper_optimizer = cooper.optim.utils.load_cooper_optimizer_from_state_dict(
            cooper_optimizer_state=cooper_optimizer_state_dict, **constrained_optimizer_kwargs
        )

        for scheduler_name, scheduler in lr_schedulers.items():
            scheduler.load_state_dict(torch.load(run_checkpoint_dir / f"{scheduler_name}_scheduler.pt"))

        return cooper_optimizer, lr_schedulers

    # If not resuming, then build a new Cooper optimizer
    if not cmp.num_constraints > 0 and config.optim.constrained_optimizer_class != cooper.optim.UnconstrainedOptimizer:
        raise ValueError(f"Expected an UnconstrainedOptimizer for cmp class {cmp} which has constraints.")

    cooper_optimizer = config.optim.constrained_optimizer_class(**constrained_optimizer_kwargs)

    return cooper_optimizer, lr_schedulers


def save_checkpoint(
    model: nn.Module, constrained_optimizer, schedulers, step: int, epoch: int, run_checkpoint_dir: Path
):
    assert run_checkpoint_dir.exists()

    torch.save(model.state_dict(), run_checkpoint_dir / "model.pt")
    torch.save(constrained_optimizer.state_dict(), run_checkpoint_dir / "constrained_optimizer.pt")
    for scheduler_name, scheduler in schedulers.items():
        torch.save(scheduler.state_dict(), run_checkpoint_dir / f"{scheduler_name}_scheduler.pt")
    torch.save({"run_id": wandb.run.id, "step": step, "epoch": epoch}, run_checkpoint_dir / "metadata.pt")


def upload_checkpoint_to_wandb(config, run_checkpoint_dir):
    # Create symlinks to files located outside the wandb run directory
    os.symlink(run_checkpoint_dir / "best_model.pt", os.path.join(wandb.run.dir, "best_model.pt"))
    wandb.save("best_model.pt", policy="end")
    os.symlink(run_checkpoint_dir / "model.pt", os.path.join(wandb.run.dir, "last_model.pt"))
    wandb.save("last_model.pt", policy="end")
    with open(run_checkpoint_dir / "config_pickle.pkl", "wb") as file:
        pickle.dump(config, file)
    os.symlink(run_checkpoint_dir / "config_pickle.pkl", os.path.join(wandb.run.dir, "config_pickle.pkl"))
    wandb.save("config_pickle.pkl", policy="end")


def load_artifact_from_wandb(project, entity, artifact_name, run_id=None, filters=None, device="cuda"):
    api = wandb.Api(overrides={"entity": entity, "project": project})
    api_key = api.api_key

    if run_id is not None:
        run = api.run(f"{entity}/{project}/{run_id}")
        runs = [run]
    elif filters is not None:
        runs = api.runs(path=f"{entity}/{project}", filters=filters, order="-created_at")
    else:
        raise ValueError("Either run_id or filters must be specified.")

    res = {}
    for _run in runs:
        with tempfile.NamedTemporaryFile("wb", delete=True) as temp_file:
            remote_file_url = _run.file(artifact_name).url
            wandb.util.download_file_from_url(temp_file.name, remote_file_url, api_key)
            loaded_artifact = torch.load(temp_file.name, map_location=device)

            # Remove extension from artifact name (e.g. ".pt"). Preserves all other dots in provided artifact name.
            clean_artifact_name = ".".join(artifact_name.split(".")[:-1])
            res[run.id] = {clean_artifact_name: loaded_artifact, "config": _run.config, "summary": _run.summary}

    return res
