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
Run save prompt scripts in [save_prompt](save_prompt) (e.g., RoBERTa for RTE):

```shell
bash save_prompt/run_rte_roberta.sh
```

### Second Stage: PromptFusion Training [Scalar, Data Dependent (Attention1), More Attention (Attention2)]
Run train PromptFusion scripts in [run_fusion_scalar](run_fusion_scalar) (e.g., RoBERTa for RTE):

```shell
bash run_fusion_scalar/run_rte_roberta.sh
```

### Implemented Results
Released results on Scalar trained with RoBERTa-base

|              | BoolQ | COPA | RTE  | WiC  | WSC  | CoNLL03 | CoNLL04 | CoNLL12 | CoNLL05 | SQuAD 1.1 | SQuAD 2.0 |
|--------------|-------|------|------|------|------|---------|---------|---------|-------------|-----------|-----------|
| Results      |   0.785|  0.670|0.679 |0.560  |0.635  |0.978     |0.850|0.911     |0.920 |0.840  | 0.705|

Released results on Data Dependent trained with RoBERTa-base

|              | BoolQ | COPA | RTE  | WiC  | WSC  | CoNLL03 | CoNLL04 | CoNLL12 | CoNLL05 | SQuAD 1.1 | SQuAD 2.0 |
|--------------|-------|------|------|------|------|---------|---------|---------|-------------|-----------|-----------|
| Results      |   0.622|  0.630|0.617 |0.544  |0.635  |0.854 |-|-     |0.410 |-  | 0.505|

Released results on More Attention trained with RoBERTa-base

|              | BoolQ | COPA | RTE  | WiC  | WSC  | CoNLL03 | CoNLL04 | CoNLL12 | CoNLL05 | SQuAD 1.1 | SQuAD 2.0 |
|--------------|-------|------|------|------|------|---------|---------|---------|-------------|-----------|-----------|
| Results      |   0.622|  0.620|0.617 |0.555  |0.635  |0.842     |0.768|-     |0.495 |-  | 0.618|

Results for SQuAD 1.1 are missing due to run-time issues and results for CoNLL 04and CoNLL 12 are
missing due to memory issues.