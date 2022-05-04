from arguments import get_args

import datasets
import logging
import os
import random
import sys

import torch
import transformers
from transformers import set_seed, Trainer
from transformers.trainer_utils import get_last_checkpoint

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.utils import get_model, TaskType, LinearWeightedSum
from tasks.superglue.dataset import SuperGlueDataset
from training.trainer_base import BaseTrainer
from training.trainer_exp import ExponentialTrainer

from tasks.utils import *

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    args = get_args()

    _, data_args, training_args, _ = args

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if data_args.task_name.lower() == "superglue":
        assert data_args.dataset_name.lower() in SUPERGLUE_DATASETS
        from tasks.superglue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "glue":
        assert data_args.dataset_name.lower() in GLUE_DATASETS
        from tasks.glue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "ner":
        assert data_args.dataset_name.lower() in NER_DATASETS
        from tasks.ner.get_trainer import get_trainer

    elif data_args.task_name.lower() == "srl":
        assert data_args.dataset_name.lower() in SRL_DATASETS
        from tasks.srl.get_trainer import get_trainer

    elif data_args.task_name.lower() == "qa":
        assert data_args.dataset_name.lower() in QA_DATASETS
        from tasks.qa.get_trainer import get_trainer

    else:
        raise NotImplementedError(
            'Task {} is not implemented. Please choose a task from: {}'.format(data_args.task_name, ", ".join(TASKS)))

    set_seed(training_args.seed)

    trainer, predict_dataset = get_trainer(args)

    model = trainer.model
    print('weight 0', model.weighted_sum.weights[0])
    # for layer in model.children():
    #     print(type(layer))
    #     if isinstance(layer, model.sequence_classification.LinearWeightedSum):
    #         print(layer.state_dict()['weight'])
    #         print(layer.state_dict()['bias'])

