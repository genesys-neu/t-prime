#!/bin/bash


python /home/jgroen/NEU/dstl/transformer/examples/pytorch/image-pretraining/run_mae.py \
    --dataset_name /home/jgroen/NEU/DSTL_1_0_specgram_cropped \
    --output_dir /home/jgroen/NEU/dstl/transformer/outputs \
    --remove_unused_columns False \
    --label_names pixel_values \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --do_train \
    --do_eval \
    --base_learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --num_train_epochs 800 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337
