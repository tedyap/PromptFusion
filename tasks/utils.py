from tasks.glue.dataset import task_to_keys as glue_tasks
from tasks.superglue.dataset import task_to_keys as superglue_tasks
import os
import torch

GLUE_DATASETS = list(glue_tasks.keys())
SUPERGLUE_DATASETS = list(superglue_tasks.keys())
NER_DATASETS = ["conll2003", "conll2004", "ontonotes"]
SRL_DATASETS = ["conll2005", "conll2012"]
QA_DATASETS = ["squad", "squad_v2"]


TASKS = ["glue", "superglue", "ner", "srl", "qa"]

DATASETS = GLUE_DATASETS + SUPERGLUE_DATASETS + NER_DATASETS + SRL_DATASETS + QA_DATASETS

ADD_PREFIX_SPACE = {
    'bert': False,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': True,
}

USE_FAST = {
    'bert': True,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': False,
}


def get_prompts():
    prompt_file_paths = os.listdir('/scratch/mc8895/prompts')

    prompts = []
<<<<<<< HEAD
    for idx, file_path in enumerate(prompt_file_paths):
        prompts.append(torch.load('prompts/' + file_path))
        print(idx, file_path)

=======
    for file_path in prompt_file_paths:
        prompts.append(torch.load('/scratch/mc8895/prompts/' + file_path))
    
>>>>>>> 4d52f9e3a97ccc2958f91b7b00299962f561d85a
    prompts = torch.stack(prompts)

    return prompts