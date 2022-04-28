export TASK_NAME=qa
export DATASET_NAME=squad
export CUDA_VISIBLE_DEVICES=0

bs=8
lr=5e-3
dropout=0.2
psl=128
epoch=30

python3 train_fusion.py \
  --model_name_or_path checkpoints/$DATASET_NAME-roberta/ \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-roberta-fusion-scalar/ \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --fusion_scalar > log.txt
