import logging

import torch
from absl import app
from absl.flags import FLAGS
from ml_collections.config_flags import config_flags as MLC_FLAGS

import src.models as models
import src.utils.experiment_utils as exp_utils
import src.utils.train_utils as train_utils
import wandb
from src.utils import BestMeter

# Initialize "FLAGS.config" object with a placeholder config
MLC_FLAGS.DEFINE_config_file("config", default="src/configs/basic.py")

logging.basicConfig()
logger = logging.getLogger()


def main(_):
    config = FLAGS.config
    logger.setLevel(getattr(logging, config.logging.log_level))
    exp_utils.set_seed(seed=config.train.seed, ensure_reproducibility=config.train.ensure_reproducibility)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    run, run_checkpoint_dir, start_epoch, start_step = exp_utils.construct_wandb_run(config)
    resumed = True if run.resumed else False

    dataloaders, num_classes, input_shape, num_protected_groups = exp_utils.load_dataset(config)

    model, pretrained_model = exp_utils.construct_models(config, num_classes, input_shape, run_checkpoint_dir, DEVICE)

    cmp_init_kwargs = {"num_protected_groups": num_protected_groups, "device": DEVICE}
    cmp_init_kwargs.update(exp_utils.filter_out_nones_from_dict(config.train.cmp_init_kwargs))
    cmp_class_name = config.train.cmp_class.__name__
    if "Baseline" not in cmp_class_name:
        cmp_init_kwargs["last_pruning_epoch"] = config.sparsity.last_pruning_epoch
    cmp = config.train.cmp_class(**cmp_init_kwargs)

    # When CMP does not have constraints, this builds an "unconstrained optimizer"
    constrained_optimizer, lr_schedulers = exp_utils.construct_constrained_optimizer(
        model, cmp, run_checkpoint_dir, config
    )

    sparsity_scheduler = exp_utils.construct_sparsity_scheduler(config)

    if hasattr(cmp, "precompute_baseline_metrics") and pretrained_model is not None:
        # Pre-compute per-group statistics for the pretrained model only once
        cmp.precompute_baseline_metrics(pretrained_model, dataloaders, DEVICE)

    best_acc_meter = BestMeter(direction="max")

    if not resumed:
        if sparsity_scheduler is not None:
            models.pruning.magnitude_prune_model_(model, sparsity_scheduler, epoch=0)

        # Log validation metrics at initialization (before any training) so all validation
        # measurements start from the same values

        eval_init_logs = {"train_eval": {}, "val": {}, "test": {}}
        for split, dataloader in dataloaders.items():
            logger.info(f"Evaluating the model at *initialization* on the {split} dataset")

            split = split + "_eval" if split == "train" else split
            eval_init_logs[split] = train_utils.evaluate_one_epoch(
                model=model,
                dataloader=dataloader,
                cmp=cmp,
                device=DEVICE,
                _epoch=-1,
                split_prefix=split,
            )

        logger.info("Logging evaluation metrics at initialization")

        init_model_stats_log = train_utils.extract_model_stats_logs(
            model=model, prefix="model_stats/epoch/", skip_keys=("layer_stats",)
        )
        init_model_stats_log["_epoch"] = -1

        for log_dict in list(eval_init_logs.values()) + [init_model_stats_log]:
            wandb.log(log_dict)

        # Save initial model to WandB. This allows us to fetch the model from WandB for
        # experiments that require a checkpoint from a parent experiment
        if not run_checkpoint_dir.exists():
            run_checkpoint_dir.mkdir(parents=True)
        torch.save(model.state_dict(), run_checkpoint_dir / "best_model.pt")
        torch.save(model.state_dict(), run_checkpoint_dir / "model.pt")

    # Reset step_id to be the last step_id from checkpoint
    _step = start_step

    for epoch in range(start_epoch, config.train.epochs):
        logger.info(f"Epoch {epoch}/{config.train.epochs}")

        if sparsity_scheduler is not None:
            wandb.log({"_epoch": epoch, "scheduler_sparsity": sparsity_scheduler.sparsity_amount(epoch)})

            if epoch == sparsity_scheduler.last_pruning_epoch:
                logger.info(f"Epoch {epoch}/{config.train.epochs}: Pruning has ended")

            models.pruning.magnitude_prune_model_(model, sparsity_scheduler, epoch)

        logger.info(f"Epoch {epoch}/{config.train.epochs}: Training loop started")
        batch_train_logs, train_epoch_logs, _step = train_utils.train_one_epoch(
            model=model,
            train_loader=dataloaders["train"],
            cmp=cmp,
            constrained_optimizer=constrained_optimizer,
            device=DEVICE,
            _step=_step,
            _epoch=epoch,
        )

        for lr_scheduler in lr_schedulers.values():
            lr_scheduler.step()

        eval_epoch_logs = {"train_eval": {}, "val": {}, "test": {}}

        for split, dataloader in dataloaders.items():
            logger.info(f"Epoch {epoch}/{config.train.epochs}: Evaluation loop started for {split} data")

            split = split + "_eval" if split == "train" else split
            eval_epoch_logs[split] = train_utils.evaluate_one_epoch(
                model=model,
                dataloader=dataloader,
                cmp=cmp,
                device=DEVICE,
                _epoch=epoch,
                split_prefix=split,
            )

        logger.info(f"Epoch {epoch}/{config.train.epochs}: Extracting model stats")
        model_stats_log = train_utils.extract_model_stats_logs(
            model=model, prefix="model_stats/epoch/", skip_keys=("layer_stats",)
        )
        model_stats_log["_epoch"] = epoch

        logger.info(f"Epoch {epoch}/{config.train.epochs}: Uploading logs to WandB")
        for log_dict in batch_train_logs + list(eval_epoch_logs.values()) + [train_epoch_logs, model_stats_log]:
            wandb.log(log_dict)

        logger.info(f"Epoch {epoch}/{config.train.epochs}: Saving checkpoint")
        # Add 1 to epoch checkpoint so that, if preempted, we can resume from epoch (k+1) since k is already done
        exp_utils.save_checkpoint(model, constrained_optimizer, lr_schedulers, _step, epoch + 1, run_checkpoint_dir)

        # Updating the best model if the current model is better in test accuracy.
        split_for_model_selection = "test"
        current_accuracy = eval_epoch_logs[split_for_model_selection][f"{split_for_model_selection}/epoch/acc"]
        if best_acc_meter.update(current_accuracy):
            torch.save(model.state_dict(), run_checkpoint_dir / "best_model.pt")

    # Upload best and last models to WandB
    exp_utils.upload_checkpoint_to_wandb(config, run_checkpoint_dir)
    run.finish()


if __name__ == "__main__":
    app.run(main)
