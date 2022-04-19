# PromptFusion


Source codes and data for NLU project.

### Setup
We conduct our experiment with Anaconda3. If you have installed Anaconda3, then create the environment for PromptFusion:

```shell
conda create -n pt2 python=3.8.5
conda activate pt2
```

After we setup basic conda environment, install pytorch related packages via:

```shell
conda install -n pt2 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

Finally, install other python packages we need:

```shell
pip install -r requirements.txt
```

### Data
For SuperGLUE and SQuAD datasets, we download them from the Huggingface Datasets APIs (embedded in our codes).

For sequence tagging (NER, SRL) datasets, we prepare a non-official packup [here](https://zenodo.org/record/6318701/files/P-tuning-v2_data.tar.gz?download=1). 
After downloading, unzip the packup to the project root.
Please use at your own risk.

### First Stage: P-tuning V2 Training
Run training scripts in [run_script](run_script) (e.g., RoBERTa for RTE):

```shell
bash run_script/run_rte_roberta.sh
```

### Save Trained Prompts on Disk
Run save prompt scripts in [save_promt](save_prompt) (e.g., RoBERTa for RTE):

```shell
bash save_prompt/run_rte_roberta.sh
```

### Second Stage: PromptFusion Training
Run save prompt scripts in [run_fusion](run_fusion) (e.g., RoBERTa for RTE):

```shell
bash run_fusion/run_rte_roberta.sh
```

### Implemented Results
Currently we have released our reimplementation on following tasks and datasets.

Released results on RoBERTa-large

|              | BoolQ | COPA | RTE  | WiC  | WSC  | CoNLL03 | CoNLL04 | OntoNotes 5.0 | CoNLL12 | CoNLL05 WSJ | CoNLL05 Brown | SQuAD 1.1 | SQuAD 2.0 |
|--------------|-------|------|------|------|------|---------|---------|---------------|---------|-------------|---------------|-----------|-----------|
| Results      |   |  |  |  |  |     |     |           |     |        |       |  | |
| Total Epochs |    |   |   |    |    |       |      |            |      |         |            |     |        |
| Best Epoch   |     |    |    |    |     |      |       |            |      |           |            |       |         |

