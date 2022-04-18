export TASK_NAME=srl
export DATASET_NAME=conll2005
export CUDA_VISIBLE_DEVICES=0

bs=16
lr=6e-3
dropout=0.1
psl=224
epoch=15

python3 get_prompt.py \
  --model_name_or_path checkpoints/$DATASET_NAME-roberta/ \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME/ \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix > log.txt
