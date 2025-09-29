from argparse import Namespace
from typing import cast

from dotenv import load_dotenv

from utils.base_argument_parser import BaseArgumentParser
from utils.enums import DATASET_CURRICULUM_TYPE
from utils.eval_argument_parser import EvalArgumentParser
from utils.metrics import MetricFunction
from utils.train_argument_parser import TrainArgumentParser

load_dotenv()
import os  # noqa: E402

PROJECT_PATH = os.getenv("PROJECT_PATH")
DATA_PATH = os.getenv("DATA_PATH")
############################################
# Base configuration
############################################
base_config: BaseArgumentParser = Namespace()

# Model architecture parameters
base_config.tokenizer_dir = f"{PROJECT_PATH}/tokenizers/num_text/base10_gpt2"
base_config.model = "rope_stem"
base_config.num_embedding_type = "base10"
base_config.normalize_num_embedding = True
base_config.add_reciprocal = False
base_config.combine_strategy = "prod"
base_config.num_loss_type = "cs"

# Dataset and caching parameters
base_config.cache_base_path = f"{DATA_PATH}/cache"

# Resource allocation parameters
base_config.compile = True
base_config.num_workers = 16
base_config.verbose = True


############################################
# Training configuration
############################################
train_config = cast(TrainArgumentParser, Namespace(**vars(base_config)))

# Training data parameters
train_config.train_set_paths = [
    f"{DATA_PATH}/Multiplication_decimal_uniform_train_30M.csv.gz",
]
train_config.train_dataset_curriculum_types = [DATASET_CURRICULUM_TYPE.CURRICULUM]
train_config.use_curriculum = True
train_config.optimize_last = True
train_config.difficulty_column = "difficulty_sd"

# Data mixing parameters
train_config.train_set_ratios = [1]
train_config.num_loss_weight = 1
train_config.train_dataset_type = "pretokenized_number"

# Validation data parameters
train_paths_metrics_dataset_types: dict[str, tuple[str, str]] = {
    f"{DATA_PATH}/Multiplication_decimal_uniform_val_10k.csv.gz": (MetricFunction.LOG_SMAPE, "curriculum_number"),
}
train_config.val_set_paths = list(train_paths_metrics_dataset_types.keys())
train_config.val_set_metrics = [v[0] for v in train_paths_metrics_dataset_types.values()]
train_config.val_dataset_types = [v[1] for v in train_paths_metrics_dataset_types.values()]

train_config.val_additional_metrics = [
    MetricFunction.EXACT_NUMBER_ACC,
    MetricFunction.SIG_BITS_ACC,
    MetricFunction.LOG_SMAPE_BASE2
]

# Training hyperparameters
train_config.save_dir = f"{PROJECT_PATH}/trained/soloTask/base10"
train_config.train_token_budget = 10_000_000_000  # Rougly equals 3 epochs (4_717_802_025)
train_config.num_warmup_tokens = train_config.train_token_budget//20
# train_config.eval_every_k_tokens = 800*64*1024 
train_config.eval_every_k_tokens = 100*64*1024  # 100 steps or 1% of the token budget
train_config.max_eval_steps = 2
train_config.no_save_latest = True
train_config.lr_scheduler_type = "cosine"

# Dynamic loss weighting parameters
train_config.loss_weight_momentum = 1
# train_config.online_weighting_warmup_tokens = 1600*64*1024 
train_config.online_weighting_warmup_tokens = 100*64*1024  # 100 steps
train_config.reset_loss_after_warmup = False #True
train_config.grad_clip = -1

# WandB parameters
train_config.wandb_project = "STEM"
train_config.wandb_group = "solo_task_base10"


############################################
# Evaluation configuration
############################################
# Add evaluation configuration if needed, similar to the reference
eval_config = cast(EvalArgumentParser, Namespace(**vars(base_config)))
test_paths_metrics_dataset_types_save_pred: dict[str, tuple[str, str, bool]] = {
    f"{DATA_PATH}/Multiplication_f64_test_10k.csv": (MetricFunction.LOG_SMAPE, "efficient_number_prompt", True),
}
eval_config.test_set_paths = list(test_paths_metrics_dataset_types_save_pred.keys())
eval_config.test_set_metrics = [v[0] for v in test_paths_metrics_dataset_types_save_pred.values()]
eval_config.test_dataset_types = [v[1] for v in test_paths_metrics_dataset_types_save_pred.values()]
eval_config.save_testset_predictions = [v[2] for v in test_paths_metrics_dataset_types_save_pred.values()]

eval_config.additional_metrics = [
    MetricFunction.LOG_SMAPE,
    MetricFunction.EXACT_NUMBER_ACC,
    MetricFunction.SIG_BITS_ACC,
    MetricFunction.LOG_SMAPE_BASE2
]
