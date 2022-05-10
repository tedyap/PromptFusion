export TASK_NAME=ner
export DATASET_NAME=conll2003
export CUDA_VISIBLE_DEVICES=0

bs=16
epoch=30
psl=128
lr=3e-2
dropout=0.1

python3 train_fusion.py \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 152 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-roberta-fusion-attention2/ \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --fusion_attention2 > logs/$DATASET_NAME-roberta-fusion-attention2.txt
