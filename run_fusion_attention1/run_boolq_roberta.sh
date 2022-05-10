export TASK_NAME=superglue
export DATASET_NAME=boolq
export CUDA_VISIBLE_DEVICES=0

bs=32
lr=7e-3
dropout=0.1
psl=128
epoch=100
checkpoints=/scratch/mc8895/promptFusion_new/checkpoints

python3 train_fusion.py \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir $checkpoints/$DATASET_NAME-roberta-fusion-attention1/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --fusion_attention1 > log_boolq.txt
