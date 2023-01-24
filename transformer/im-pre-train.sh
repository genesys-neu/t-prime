#!/bin/bash


python /home/jgroen/NEU/dstl/transformer/examples/pytorch/image-pretraining/run_mae.py \
    --dataset_name /home/jgroen/NEU/DSTL_1_0_specgram_cropped \
    --output_dir /home/jgroen/NEU/dstl/transformer/outputs \
    --remove_unused_columns False \
    --label_names pixel_values \
    --do_train \
    --do_eval
