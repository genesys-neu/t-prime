#!/bin/bash

datasets=("RM_573C_1" "RM_573C_2" "RM_142_1" "RM_142_2" "RM_572C_1" "RM_572C_2")
pathtorepository=""
for dataset in "${datasets[@]}"; do
    echo "Running command without $dataset dataset:"
    command="python3 TPrime_finetune.py --model_path=$pathtorepository/t-prime/TPrime_transformer/model_cp/model_lg.pt --RMSNorm --ds_path=$pathtorepository/t-prime/data --use_gpu --transformer_version v1 --transformer lg --ota_dataset=no$dataset --test_mode=inference --retrain --back_class --datasets"
    for ds in "${datasets[@]}"; do
        if [ "$ds" != "$dataset" ]; then
            command+=" $ds"
        fi
    done
    
    
    echo "Executing command: $command"
    (eval "$command") &
    wait
    
    echo "Running command with $dataset dataset only:"
    model_name="model_lg_no${dataset}_inf_RMSn_bckg_ft.pt"
    command_test="python3 TPrime_finetune.py --model_path=$pathtorepository/t-prime/TPrime_transformer/model_cp/$model_name --RMSNorm --ds_path=$pathtorepository/t-prime/data --use_gpu --transformer_version v1 --transformer lg --ota_dataset=no$dataset --test_mode=inference --back_class --datasets ${dataset} --test"
    echo "Executing command: $command_test"
    (eval "$command_test") &
    wait
done
