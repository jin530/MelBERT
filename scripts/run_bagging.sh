#!/bin/bash

INDEXES=$(seq 0 9)
for i in $INDEXES
do
    echo "Running bagging for index $i"
    python main.py --data_dir data/VUA20 --task_name vua --model_type MELBERT --train_batch_size 32 --learning_rate 3e-5 --warmup_epoch 2 --num_bagging 10 --bagging_index $i
done