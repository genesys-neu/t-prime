#!/bin/bash
RAWRATIO=0.5
POSTFIX=all20MHz_half
for var in "$@"
do
  echo "Generating dataset for $var slice length (raw ratio $RAWRATIO)"
  python DSTL_dataset.py --raw_path /home/belgiovinem/Research/DSTL/DSTL_DATASET_1_1 --postfix $POSTFIX --raw_data_ratio $RAWRATIO --slicelen $var
done
