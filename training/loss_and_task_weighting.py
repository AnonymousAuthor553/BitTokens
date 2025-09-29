import logging
from typing import cast

import torch

from dataloader.curriculum_manager import CurriculumManager
from dataloader.dataset_utils import RatioSampler
from training.setup import Trainer
from utils.enums import TrainMetrics
from utils.metrics import inverse_generalized_mean
from utils.train_argument_parser import TrainArgumentParser
from utils.visualizer import save_or_add_to_csv


def perform_loss_and_task_weighting(
        args: TrainArgumentParser,
        trainer: Trainer,
        curriculum_manager: CurriculumManager | None,
        train_metrics: TrainMetrics,
        accs: list[float],
        train_losses: dict[str, float],
    ) -> None:
    """
    Performs dynamic loss weighting and task sampling ratio adjustment during training.
    This function implements adaptive loss weighting strategies to balance multiple tasks
    during multi-task learning. It supports both static loss normalization after warmup
    and dynamic adjustment based on task performance using momentum-based updates.

    The function implements a sophisticated multi-task learning strategy that balances
    tasks by adjusting both loss weights and data sampling ratios to optimize for
    harmonic mean performance across tasks.

    Args:
        args (TrainArgumentParser): Training arguments
        trainer (Trainer): Training object
        train_metrics (TrainMetrics): Training metrics containing token counts per step
        accs (list[float]): Current accuracy values for each task
        train_losses (dict[str, float]): Current training losses for each task

    Behavior:
    ---------
    - **Task Selection**: Determines which tasks to optimize based on `optimize_last` flag
    - **Warmup Phase**: If `reset_loss_after_warmup` is True and warmup tokens reached:
        - Normalizes loss weights to achieve uniform task loss magnitude
        - Sets static loss scaling for remainder of training
    - **Dynamic Phase**: If momentum < 1 and after warmup:
        - Calculates target loss ratios using inverse generalized mean of task accuracies
        - Updates loss weights to maintain equal normalized losses across tasks
        - Adjusts sampling ratios in the data loader to achieve target loss distribution
    - **Logging**: Saves loss weights and training ratios to CSV files for monitoring

    Side Effects:
    -------------
    - Modifies `args.train_set_loss_weights` in-place
    - Updates task sampling ratios in `trainer.train_loader.sampler`
    - Writes training logs to CSV files in `trainer.save_dir`
    - Updates `trainer.loss_weights` and `trainer.train_ratios` dictionaries
    """
    old_weights = args.train_set_loss_weights.clone()
    old_ratios: torch.FloatTensor = cast(RatioSampler, trainer.train_loader.sampler).get_task_ratios()
    if curriculum_manager is not None:
        task_selector, ratio_magnitude = curriculum_manager.get_task_selector(optimize_last=args.optimize_last)
        loss_task_selector, _ = curriculum_manager.compute_true_task_ratios(task_selector, old_ratios[task_selector]) # pyright: ignore[reportArgumentType]
    elif args.optimize_last:
        ratio_magnitude=1
        task_selector = slice(None,len(train_losses)-1)
        loss_task_selector = task_selector
    else:
        ratio_magnitude = 1-old_ratios[-1] 
        task_selector = slice(None,len(train_losses)-2)
        loss_task_selector = task_selector
    target_loss_old = old_ratios.clone()[loss_task_selector] / ratio_magnitude
    if args.reset_loss_after_warmup:
        train_loss_array  = torch.tensor(list(train_losses.values())[1:])
        loss_descaled: torch.FloatTensor = train_loss_array / (old_weights * old_ratios)
        trainer.WARMUP_LOSS_SCALE = args.loss_norm_magnitude / loss_descaled
        # Forward-fill NaN/inf values with the previous valid value
        mask = torch.isnan(trainer.WARMUP_LOSS_SCALE) | torch.isinf(trainer.WARMUP_LOSS_SCALE)
        if mask.any():
            # Create indices and use cummax to forward-fill (assumes index 0 is never NaN)
            indices = torch.arange(len(trainer.WARMUP_LOSS_SCALE))
            indices[mask] = -1  # Mark invalid positions
            trainer.WARMUP_LOSS_SCALE = trainer.WARMUP_LOSS_SCALE[indices.cummax(0).values]
        args.train_set_loss_weights = trainer.WARMUP_LOSS_SCALE.clone()            # Static loss-scale normalisation, s.t. loss(task) = 1
        if train_metrics.num_tokens_per_step[trainer.step] >= args.online_weighting_warmup_tokens:
            args.reset_loss_after_warmup = False # Only do once
            logging.info(f"Normalized loss weights: {args.train_set_loss_weights.tolist()} after {train_metrics.num_tokens_per_step[trainer.step]} tokens")
    elif args.loss_weight_momentum < 1 and train_metrics.num_tokens_per_step[trainer.step] >= args.online_weighting_warmup_tokens:
        # Target loss is chosen to optimize for harmonic mean 
        task_accs: torch.FloatTensor = torch.tensor(accs)[task_selector]
        inv_gen_mean = inverse_generalized_mean(task_accs=task_accs, p=args.task_loss_exponent)
        if curriculum_manager is not None:
            task_selector, inv_gen_mean = curriculum_manager.compute_true_task_ratios(task_selector, inv_gen_mean) # pyright: ignore[reportArgumentType]
        target_loss_new = args.loss_weight_momentum * target_loss_old + (1-args.loss_weight_momentum) * inv_gen_mean
        train_loss_array  = torch.tensor(list(train_losses.values())[1:])
        loss_descaled: torch.FloatTensor = train_loss_array / (old_weights * old_ratios)

        task_mean = trainer.WARMUP_LOSS_SCALE[-1] * loss_descaled[-1] # last task is used as reference, usually text
        args.train_set_loss_weights[:-1] = task_mean / loss_descaled[:-1] # Loss normalization, s.t. loss(task) = MEAN(normalized losses)  # pyright: ignore[reportArgumentType, reportCallIssue] 
        # Forward-fill NaN/inf values with the previous valid value
        mask = torch.isnan(args.train_set_loss_weights) | torch.isinf(args.train_set_loss_weights) # pyright: ignore[reportArgumentType]
        if mask.any():
            # Create indices and use cummax to forward-fill (assumes index 0 is never NaN)
            indices = torch.arange(len(args.train_set_loss_weights))
            indices[mask] = -1  # Mark invalid positions
            args.train_set_loss_weights = args.train_set_loss_weights[indices.cummax(0).values]
        task_ratios: torch.FloatTensor = target_loss_new*ratio_magnitude
        cast(RatioSampler, trainer.train_loader.sampler).update_task_ratio(task_selector, task_ratios) # Loss weighting by adjusting sample ratios
    if args.loss_weight_momentum < 1:
        d = dict(zip(trainer.val_loaders.keys(),[t.item() for t in args.train_set_loss_weights]))
        trainer.loss_weights[train_metrics.num_tokens_per_step[trainer.step]] = {k.removesuffix("_val_10k.csv.gz"): v for k,v in d.items()}
        d = {"step": trainer.step, "tokens": train_metrics.num_tokens_per_step[trainer.step], **d}
        save_or_add_to_csv(d, trainer.save_dir / "loss_weights.csv")

        d = dict(zip(trainer.val_loaders.keys(),[t.item() for t in old_ratios]))
        trainer.train_ratios[train_metrics.num_tokens_per_step[trainer.step]] = {k.removesuffix("_val_10k.csv.gz"): v for k,v in d.items()}
        d = {"step": trainer.step, "tokens": train_metrics.num_tokens_per_step[trainer.step], **d}
        save_or_add_to_csv(d, trainer.save_dir / "train_set_ratios.csv")
