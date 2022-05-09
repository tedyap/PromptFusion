export TASK_NAME=superglue
export DATASET_NAME=rte
export CUDA_VISIBLE_DEVICES=0

bs=16
lr=1e-5
dropout=0.1
psl=128
epoch=100
checkpoints=/scratch/mc8895/checkpoints_1012

python3 train_fusion.py \
<<<<<<< HEAD
  --model_name_or_path roberta-large \
=======
  --model_name_or_path roberta-base \
>>>>>>> 4d52f9e3a97ccc2958f91b7b00299962f561d85a
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
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --fusion_attention1 > log_fusion.txt
