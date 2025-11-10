#!/bin/bash
RAWRATIO=0.5
POSTFIX=all20MHz_half
PATH_TO_REPOSITORY=""
for var in "$@"
do
  echo "Generating dataset for $var slice length (raw ratio $RAWRATIO)"
  python TPrime_dataset.py --raw_path $PATH_TO_REPOSITORY/dstl/data/DSTL_DATASET_1_1 --postfix $POSTFIX --raw_data_ratio $RAWRATIO --slicelen $var
done
