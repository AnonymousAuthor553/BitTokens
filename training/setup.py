import logging
from collections import OrderedDict
from math import ceil
from pathlib import Path
from sys import maxsize
from typing import Optional

import pandas as pd
import torch
import wandb
import wandb.wandb_run
from muon import SingleDeviceMuonWithAuxAdam
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from dataloader.curriculum_manager import CurriculumManager
from dataloader.dataloaders import get_eval_loader, get_train_loader
from networks.models import get_model
from utils.enums import Trainer
from utils.train_argument_parser import TrainArgumentParser
from utils.util_funcs import load_checkpoint, load_metrics, print_and_save_arguments
from utils.warm_start_lr_scheduler import (
    WarmStartLrScheduler,
    WarmStartReduceLROnPlateau,
)


def setup(args: TrainArgumentParser, wandb_run: Optional[wandb.wandb_run.Run] = None) -> Trainer:
    """Sets up the training environment, including loading datasets, initializing the model, optimizer, and learning rate scheduler
    Args:
        args (TrainArgumentParser): Parsed command line arguments.
        wandb_run (Optional[wandb.wandb_run.Run]): Optional Weights & Biases run object for logging.
    Returns:
        trainer (Trainer): An instance of the Trainer class, which encapsulates the model, optimizer, and training loop.
    """
    # Create timestamped save directory
    save_dir = Path(args.save_dir)
    time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_name = "_" +args.run_name if args.run_name else ""
    save_dir = save_dir / f"{time_stamp}_{args.seed}{run_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving to {save_dir}")

    if args.device == torch.device("cuda:0"):
        assert torch.cuda.is_available(), "CUDA is not available"

    tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    tokenizer.padding_side = "left"

    if args.continue_from:
        metrics_df, step = load_metrics(args.continue_from, save_dir)
    else:
        step = 0

    # Load the datasets
    logging.info(f"Loading train dataset from {list(map(str,args.train_set_paths))}")
    
    # Set up curriculum learning if enabled
    curriculum_manager = None
    if args.use_curriculum:
        # Number of tasks is determined by the number of training sets
        num_tasks = len(args.train_set_paths)
        
        curriculum_manager = CurriculumManager(
            num_tasks=num_tasks,
            effective_batch_size=args.effective_batch_size,
            initial_max_difficulty_fraction=args.curriculum_max_difficulty_fraction,
            advancement_threshold=args.curriculum_advancement_threshold,
            new_difficulty_ratio=args.curriculum_new_difficulty_ratio,
            generalized_mean_power=args.curriculum_generalized_mean_power,
            performance_history_size=args.curriculum_performance_history_size,
            advancement_difficulty_window_size=args.curriculum_advancement_difficulty_window_size,
            advancement_performance_window_size=args.curriculum_advancement_performance_window_size,
            advancement_step_size_fraction=args.curriculum_step_size,
            preview_difficulty_ratio=args.preview_difficulty_ratio,
            preview_exponential_decay=args.preview_exponential_decay,
            endgame_switch_step_fraction=args.endgame_switch_step_fraction,
            endgame_switch_lr_fraction=args.endgame_switch_lr_fraction,
            total_training_steps=args.max_train_steps,
            max_lr=args.lr,
            min_difficulty_ratio=args.min_difficulty_ratio,
        )
        logging.info(f"Created curriculum manager with {num_tasks} tasks")
        logging.info(f"Initial max difficulty: {args.curriculum_max_difficulty_fraction}")
    
    train_loader = get_train_loader(
        args.train_dataset_type,
        args.train_dataset_curriculum_types,
        args.train_set_paths,
        tokenizer,
        args.device_batch_size,
        effective_batch_size=args.effective_batch_size,
        context_length=args.context_length,
        num_workers=args.num_workers,
        train_cached_paths=args.train_cache_paths, # pyright: ignore[reportArgumentType]
        num_tokens=args.train_set_tokens,
        unique_samples=args.unique_samples,
        seed=args.seed,
        cache_base_path=args.cache_base_path,
        tokenizer_path=args.tokenizer_dir,
        train_set_ratios=args.train_set_ratios,
        curriculum_manager=curriculum_manager,
        difficulty_column=args.difficulty_column
    )

    val_loaders = OrderedDict()
    for i, (val_set_path, val_cache_paths, val_dataset_type, val_set_tokens) in enumerate(zip(args.val_set_paths, args.val_cache_paths, args.val_dataset_types, args.val_set_tokens)):
        logging.info(f"Loading validation dataset from {val_set_path}")
        val_loader = get_eval_loader(
            val_dataset_type,
            val_set_path,
            tokenizer,
            args.effective_batch_size if val_dataset_type in ["curriculum_number", "efficient_number_prompt", "efficient_number_prompt_pos" ] else args.device_batch_size,
            context_length=args.context_length,
            seed=args.seed,
            num_workers=0,
            val_cached_path=val_cache_paths,
            num_tokens=val_set_tokens,
            unique_samples=args.max_eval_steps*args.effective_batch_size,
            shuffle=True,
            cache_base_path=args.cache_base_path,
            difficulty_column=args.difficulty_column,
            dataset_curriculum_type=args.train_dataset_curriculum_types[i] if i < len(args.train_dataset_curriculum_types) else None
        )
        val_loaders[val_set_path.name.removesuffix("_val_50k.csv.gz")] = val_loader
    logging.info("Preparing first step")
    batch = next(iter(train_loader))
    if "sample_idx" in batch:
        samples = dict()
        for i, train_set_path in enumerate(args.train_set_paths):
            sample_mask = batch["sample_idx"]==i
            if sample_mask.any():
                samples[train_set_path.name.removesuffix("_8M_train.csv.gz")] = batch["input_ids"][sample_mask][0,:30]
    else:
        samples = {"random": batch["input_ids"][0,:30]}
    samples_decoded = ""
    for name, text in zip(samples.keys(), tokenizer.batch_decode(list(samples.values()))):
        samples_decoded += f"\x1B[1m{name}\x1B[0m: \x1B[3m{text}\x1B[0m\n"
    logging.info(f"First step prepared. Sample:\n{samples_decoded}")

    dataset_length = len(train_loader.dataset) # pyright: ignore[reportArgumentType]
    if args.train_token_budget == maxsize:
        if args.max_train_steps == maxsize:
            args.max_train_steps = min(args.max_train_steps, args.num_epochs*len(train_loader))
            args.train_token_budget =dataset_length*args.context_length*args.num_epochs
            logging.warning(f"Setting train_token_budget={args.train_token_budget} and max_train_steps={args.max_train_steps} because of num_epochs. Please set train_token_budget explicitly comparability with other experiments.")
        else:
            args.train_token_budget = args.max_train_steps*args.device_batch_size*args.context_length
            args.num_epochs = ceil(args.train_token_budget / (dataset_length*args.context_length))
            logging.warning(f"Setting train_token_budget={args.train_token_budget} and num_epochs={args.num_epochs} because of max_train_steps. Please set train_token_budget explicitly comparability with other experiments.")
    else:
        assert args.max_train_steps == maxsize, "max_train_steps and train_token_budget cannot be set at the same time"
        num_samples = args.train_token_budget / args.context_length

        args.max_train_steps = ceil(num_samples / args.device_batch_size)
        args.num_epochs = ceil( args.max_train_steps / dataset_length)
    if curriculum_manager is not None:
        curriculum_manager.total_training_steps = args.max_train_steps

    logging.info(f"Total training steps: {args.max_train_steps}")

    if args.continue_from:
        # TODO Handle dicts
        raise NotImplementedError("Handle dicts")
    else:
        if args.from_pretrained:
            model = get_model(
                tokenizer=tokenizer,
                args=args,
                pretrained_model_dir=args.from_pretrained,
                device=args.device
            )
        else:
            model = get_model(tokenizer=tokenizer,args=args,device=args.device)
        # collect the parameters to optimize
        hidden_matrix_params = [p for p in model.transformer.h.parameters() if p.ndim >= 2]
        embed_params = [p for p in model.transformer.wte.parameters()]
        scalar_params = [p for p in model.parameters() if p.ndim < 2]
        head_params = [model.lm_head.weight] if not model.config.tie_word_embeddings else []
        param_ids = {id(param) for param in hidden_matrix_params + scalar_params + head_params + embed_params}
        remaining_params = [p for p in model.parameters() if id(p) not in param_ids]
        head_params += remaining_params
        param_groups = [
            dict(params=hidden_matrix_params, use_muon=True,
                lr=args.lr, momentum=0.95),
            dict(params=head_params, use_muon=False,
                lr=0.2*args.lr, betas=(0.9, 0.95), weight_decay=0),
            dict(params=embed_params, use_muon=False,
                lr=15*args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay),
            dict(params=scalar_params, use_muon=False,
                lr=args.lr, betas=(0.9, 0.95), weight_decay=0)
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        # Create a scheduler for each optimizer
        lr_schedulers = []
        if args.lr_scheduler_type == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                WarmStartLrScheduler(optimizer, total_iters=ceil(args.num_warmup_steps / (args.effective_batch_size//args.device_batch_size))),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    ceil((args.max_train_steps-args.num_warmup_steps) / (args.effective_batch_size//args.device_batch_size)),
                    eta_min=args.lr/10
                )
            ],
            [ceil(args.num_warmup_steps / (args.effective_batch_size//args.device_batch_size))]
        )
        elif args.lr_scheduler_type == "plateau":
            lr_scheduler = WarmStartReduceLROnPlateau(
                optimizer,
                warmup_iters=ceil(args.num_warmup_steps / (args.effective_batch_size//args.device_batch_size)),
                mode="max",
                factor=2/3, # 5 reduction steps
                patience=20 * (args.eval_every_k_steps / (args.effective_batch_size//args.device_batch_size)),
                threshold=0.01,
                threshold_mode="abs",
                cooldown=0,
                eps=1e-8
            )
        else:
            raise ValueError(f"Unknown lr_scheduler_type: {args.lr_scheduler_type}")
        lr_schedulers.append(lr_scheduler)

    if args.from_pretrained is not None:
        load_checkpoint(
            optimizer=optimizer,
            checkpoint_dir=args.from_pretrained,
            device=args.device,
            lr_schedulers=lr_schedulers,
            curriculum_manager=curriculum_manager
        )

    print_and_save_arguments(args, save_dir)

    # Print the number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters in the model: {total_params}")
    logging.info(f"Trainable parameters in the model: {trainable_params}")
    logging.info(f"Saving to {save_dir}")
    eval_use_linear_prob: dict[str, list[float]] = {val_set_path.name.removesuffix("_1M_val.csv.gz"): [] for val_set_path in args.val_set_paths}
    eval_use_linear_prob_agg: dict[str, list[float]] = {val_set_path.name.removesuffix("_1M_val.csv.gz"): [] for val_set_path in args.val_set_paths}

    return Trainer(
        model=model,
        optimizer=optimizer,
        tt=tqdm(range(step, args.max_train_steps), initial=step, total=args.max_train_steps, desc="Training", unit="step", leave=False) if args.tqdm else None,
        step=step,
        train_loader=train_loader,
        val_loaders=val_loaders,
        lr_schedulers=lr_schedulers,
        tokenizer=tokenizer,
        eval_use_linear_prob=eval_use_linear_prob,
        eval_use_linear_prob_agg=eval_use_linear_prob_agg,
        save_dir=save_dir,
        curriculum_manager=curriculum_manager
    )
