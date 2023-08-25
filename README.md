# Main Pipeline
To run real time classification use:
```
python3 dstl_run.py [--frequency] [--timeout] [--model_path] [--model_size] [--RMSNorm] [--tensorRT] [--protocols]
```
All arguments are optional:
- `--frequency` (or -fq) specifies the center frequency the device will be tuned to; the default value is 2.457e9 (WiFi channel 10). 
- `--timeout` (or -t) specifies the time (in seconds) to run before gracefully shutting down; the default is 60 seconds. 
- `--model_path` is self explanatory; the default path is the current directory. 
- `--model_size` specifies if you are using the large "lg" or small transformer "sm". Note that other models may require additional changes. 
- `--RMSNorm` is a flag that specifies to use the RMSNorm block. 
- `--tensorRT` is a flag that specifies to use TensorRT optimization. 
- `--protocols` is a list of the classes to be considered; the default list is ['802_11ax', '802_11b', '802_11n', '802_11g', 'noise'].
An example of a complete command:
```
python dstl_run.py -fq 2.427e9 -t 180 --model_path dstl_transformer/model_cp/model_lg_otaglobal_inf_RMSn_bckg_ft.pt --model_size lg --tensorRT --RMSNorm
```

# Transformer models
All DSTL specific transformer models are in the `dstl_transformer/` folder.
## Training procedure
The main script `DSTL_transformer_train.py` can be used to train T-PRIME transformer architectures as follows:
```
usage: DSTL_transformer_train.py [-h] [--snr_db SNR_DB [SNR_DB ...]] [--useRay] [--num-workers NUM_WORKERS] [--use-gpu] [--address ADDRESS] [--test] [--wchannel WCHANNEL] [--cp_path CP_PATH]
                                 [--cls_token] [--dataset_ratio DATASET_RATIO] [--Layers LAYERS] [--Epochs EPOCHS] [--Learning_rate LEARNING_RATE] [--Batch_size BATCH_SIZE]
                                 [--Slice_length SLICE_LENGTH] [--Sequence_length SEQUENCE_LENGTH] [--Positional_encoder POSITIONAL_ENCODER]
```

The Large (LG) and Small (SM) implementations of T-PRIME can be reproduced and trained with the following commands:
- Large, LG (M=24, S=64)
```
python3 DSTL_transformer_train.py --wchannel=random --snr_db=range --use-gpu --cp_path=./model_cp --Layers=2 --Epochs=5 --Learning_rate=0.0002 --Batch_size=122 --Slice_length=64 --Sequence_length=24 --Positional_encoder=False 
```
- Small, SM (M=64, S=128)
```
python3 DSTL_transformer_train.py --wchannel=random --snr_db=range --use-gpu --cp_path=./model_cp --Layers=2 --Epochs=5 --Learning_rate=0.0002 --Batch_size=122 --Slice_length=128 --Sequence_length=64 --Positional_encoder=False
```
### Arguments description
```
  -h, --help            show this help message and exit
  --snr_db SNR_DB [SNR_DB ...]
                        SNR levels to be considered during training. It's possible to define multiple noise levels to be chosen at random during input slices generation. (default: [30])
  --useRay              Run with Ray's Trainer function (default: False)
  --num-workers NUM_WORKERS, -n NUM_WORKERS
                        Sets number of workers for training. (default: 2)
  --use-gpu             Enables GPU training (default: False)
  --address ADDRESS     the address to use for Ray (default: None)
  --test                Testing the model (default: False)
  --wchannel WCHANNEL   Wireless channel to be applied, it can beTGn, TGax, Rayleigh, relative or random. (default: None)
  --cp_path CP_PATH     Path to the checkpoint to save/load the model. (default: ./model_cp)
  --cls_token           Use the Transformer v2 (default: False)
  --dataset_ratio DATASET_RATIO
                        Portion of the dataset used for training and validation. (default: 1.0)
  --Layers LAYERS
  --Epochs EPOCHS
  --Learning_rate LEARNING_RATE
  --Batch_size BATCH_SIZE
  --Slice_length SLICE_LENGTH
                        Slice length in which a sequence is divided. (default: 128)
  --Sequence_length SEQUENCE_LENGTH
                        Sequence length to input to the transformer. (default: 64)
  --Positional_encoder POSITIONAL_ENCODER
```

# Other models (CNN, ResNet, AMCNet, MCFormer)
We offer several implementations, both adapted from available Github code or based on paper descriptions. The code to train and test all models (except transformers) is in `cnn_baseline/`.

To reproduce the architectures in the paper, train them using the following commands:
- CNN 1D 
  ```
  python3 DSTL_torch_train.py --model= --channel=random --snr_db=range --cp_path=./results_slice512 --postfix=all20MHz --raw_path=/home/<user>/DSTL/DSTL_DATASET_1_1 --slicelen=512
  ```
- ResNet
  ```
  python3 DSTL_torch_train.py --model=ResNet --out_mode=real_invdim --channel=random --snr_db=range --cp_path=./results_ResNet --postfix=all20MHz --raw_path=/home/<user>/DSTL/DSTL_DATASET_1_1 --slicelen=1024
  ```
- AMCNet
  ```
  python3 DSTL_torch_train.py --model=AMCNet --channel=random --snr_db=range --cp_path=./results_AMCNet --postfix=all20MHz --raw_path=/home/<user>/DSTL/DSTL_DATASET_1_1 --slicelen=128
  ```
- MCFormer
  ```
   python3 DSTL_torch_train.py --model=MCformer --channel=random --snr_db=range --cp_path=./results_MCformer_largekernel --postfix=all20MHz_half --raw_data_ratio=0.5 --raw_path=/home/<user>/DSTL/DSTL_DATASET_1_1 --slicelen=128 --debug
  ```


## Training procedure
The main script `DSTL_torch_train.py` can be used in order to select what protocols to train on, what channel models (specific channels or random) to be applied, levels of noise, baseband signal ratio to be used to generate the dataset etc. Note that this code differs from the transformer one in the way slices are generated with a sequence lengt of 1 (i.e. M = 1). The script is used as follows:
```
DSTL_torch_train.py [-h] [--noise NOISE] [--snr_db SNR_DB [SNR_DB ...]] [--test] [--cp_path CP_PATH]
                           [--protocols {802_11ax,802_11b,802_11b_upsampled,802_11n,802_11g} [{802_11ax,802_11b,802_11b_upsampled,802_11n,802_11g} ...]]
                           [--channel {TGn,TGax,Rayleigh,relative,random,None,None}] --raw_path RAW_PATH [--slicelen SLICELEN] [--overlap_ratio OVERLAP_RATIO] [--postfix POSTFIX]
                           [--raw_data_ratio RAW_DATA_RATIO] [--normalize] [--model {baseline_cnn1d,AMCNet,ResNet,LSTM,MCformer}] [--out_mode {real,complex,real_invdim,real_ampphase}] [--debug]
                           [--useRay] [--num-workers NUM_WORKERS] [--use-gpu] [--address ADDRESS]
```
### Arguments description
```
optional arguments:
  -h, --help            show this help message and exit
  --noise NOISE         Specify if noise needs to be applied or not during training (default: True)
  --snr_db SNR_DB [SNR_DB ...]
                        SNR levels to be considered during training. It's possible to define multiple noise levels to be chosen at random during input slices generation. (default: [30])
  --test                Testing the model (default: False)
  --cp_path CP_PATH     Path to the checkpoint to save/load the model. (default: ./)
  --protocols {802_11ax,802_11b,802_11b_upsampled,802_11n,802_11g} [{802_11ax,802_11b,802_11b_upsampled,802_11n,802_11g} ...]
                        Specify the protocols/classes to be included in the training (default: ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g'])
  --channel {TGn,TGax,Rayleigh,relative,random,None,None}
                        Specify the channel models to apply during data generation. (default: None)
  --raw_path RAW_PATH   Path where raw signals are stored. (default: None)
  --slicelen SLICELEN   Signal slice size (default: 128)
  --overlap_ratio OVERLAP_RATIO
                        Overlap ratio for slices generation (default: 0.5)
  --postfix POSTFIX     Postfix to append to dataset file. (default: )
  --raw_data_ratio RAW_DATA_RATIO
                        Specify the ratio of examples per class to consider while training/testing (default: 1.0)
  --normalize           [DEPRECATED] Use a layer norm as a first layer. (default: True)
  --model {baseline_cnn1d,AMCNet,ResNet,LSTM,MCformer}
                        Model to be used for training (default: baseline_cnn1d)
  --out_mode {real,complex,real_invdim,real_ampphase}
                        Specify data generator output format (default: real)
  --debug               It will force run on cpu and disable Wandb. (default: False)
  --useRay              [DEPRECATED] Run with Ray's Trainer function (default: False)
  --num-workers NUM_WORKERS, -n NUM_WORKERS
                        [DEPRECATED] Sets number of workers for training. (default: 2)
  --use-gpu             [DEPRECATED] Enables GPU training (default: True)
  --address ADDRESS     [DEPRECATED] the address to use for Ray (default: None)

```
## Competitor models source code
- CNN1D: variation on [O'Shea _et.al._, 2016](https://arxiv.org/abs/1602.04105)
- AMCNet: adapted from [official code](https://github.com/zjwXDU/AMC-Net/tree/main)
- ResNet: adapted from [available implementation](https://github.com/liuzhejun/ResNet-for-Radio-Recognition/tree/master)
- MCFormer: adapted from [official code](https://github.com/InterDigitalInc/Fireball/blob/8c98a40e6baba489ac9c028aa4fe71b2ae782f79/Playgrounds/MCformer/MCformer.ipynb)


# Pre-processing
All preprocessing specific code is in the /preprocessing folder.


# Other models
The CNN code is in the /cnn_baseline folder.

All huggingface transformer based code is in the /transformer folder. This was only used for the initial visual transformer testing, and is not present or necessary for the final implementation.

# Datasets
-All datasets can be found on the shared OneDrive: https://northeastern-my.sharepoint.com/:f:/r/personal/sioannidis_northeastern_edu/Documents/DSTL-NU%20OneDrive/Datasets?csf=1&web=1&e=RDViVZ
-Unzip the datasets into the data/ folder using this command: unzip DATASET_1_0.zip -d data/
