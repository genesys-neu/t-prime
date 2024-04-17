# T-PRIME Repository
This repository is dedicated to the T-PRIME project, aimed at waveform classification in wireless communications. T-PRIME introduces a novel approach to protocol identification and offers a comprehensive comparison with several state-of-the-art (SoTA) architectures, all of which can be trained and tested using the provided codebase. We also provide the instructions for deploying the trained models on the AIR-T platform. The repository is structured into several sections to guide you through the process. 

## Installation
Before getting started, make sure you have the required dependencies installed. To set up the necessary environment, follow these steps:
1. Clone this repository to your local machine:
```
git clone https://github.com/genesys-neu/t-prime.git
cd t-prime/
```
2. Create a Conda environment and install the required packages:
```
conda env create --name t-prime --file ./conda-envs/TPrime_conda_env_training__nobuilds.yaml
```
3. Activate the newly created Conda environment:
```
conda activate t-prime
```
## Datasets
All [datasets](https://repository.library.northeastern.edu/collections/neu:h989s847q) have been publicly released on Northeastern University's Digital Repository Service.

The following command can be used to unzip the dataset folders into `data/` folder:
```
unzip <dataset filename>.zip -d data/
```
All datasets for provided scripts will be read from `data/` folder.

For more information about provided datasets, refer to dedicated [README](data/README.md) file. 

## Training and testing
To train and test our models, it is essential to distinguish between the three types of data at our disposal: simulated data acquired through MATLAB, data collected over the air (OTA), and data encompassing overlapping protocols. This differentiation is crucial because each data type undergoes distinct processing procedures facilitated by separate scripts.
### Simulated data
#### Training procedure
##### Transformer models
All T-PRIME specific transformer models are in the `TPrime_transformer/` folder. The main script `TPrime_transformer_train.py` can be used to train T-PRIME transformer architectures as follows:
```
usage: TPrime_transformer_train.py [-h] [--snr_db SNR_DB [SNR_DB ...]] [--useRay] [--num-workers NUM_WORKERS] [--use-gpu] [--address ADDRESS] [--test] [--wchannel WCHANNEL] 
                                   [--raw_path RAW_PATH] [--cp_path CP_PATH] [--cls_token] [--dataset_ratio DATASET_RATIO] [--Layers LAYERS] [--Epochs EPOCHS] [--Learning_rate LEARNING_RATE] 
                                   [--Batch_size BATCH_SIZE] [--Slice_length SLICE_LENGTH] [--Sequence_length SEQUENCE_LENGTH] [--Positional_encoder POSITIONAL_ENCODER]

```

The Large (LG) and Small (SM) implementations of T-PRIME can be reproduced and trained with the following commands:
- Large, LG (M=24, S=64)
```
python3 TPrime_transformer_train.py --wchannel=random --snr_db=range --use-gpu --postfix=TransformerLG --raw_path=../data/DATASET1_1 --cp_path=./model_cp --Layers=2 --Epochs=5 --Learning_rate=0.0002 --Batch_size=122 --Slice_length=64 --Sequence_length=24 --Positional_encoder=False 
```
- Small, SM (M=64, S=128)
```
python3 TPrime_transformer_train.py --wchannel=random --snr_db=range --use-gpu --postfix=TransformerSM --raw_path=../data/DATASET1_1 --cp_path=./model_cp --Layers=2 --Epochs=5 --Learning_rate=0.0002 --Batch_size=122 --Slice_length=128 --Sequence_length=64 --Positional_encoder=False
```
###### Arguments description
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
  --raw_path RAW_PATH   Path where raw signals are stored. (default: ../data/DATASET1_1)
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

##### Other models (CNN, ResNet, AMCNet, MCFormer)
We offer several implementations, both adapted from available Github code or based on paper descriptions. The code to train all baseline models (except transformers) is in `baseline_model/`.

To reproduce the architectures in the paper, train them using the following commands:
- CNN 1D 
  ```
  python3 TPrime_torch_train.py --model=baseline_cnn1d --channel=random --snr_db=range --cp_path=./results_slice512 --postfix=all20MHz --raw_path=../data/DATASET1_1 --slicelen=512
  ```
- ResNet
  ```
  python3 TPrime_torch_train.py --model=ResNet --out_mode=real_invdim --channel=random --snr_db=range --cp_path=./results_ResNet --postfix=all20MHz --raw_path=../data/DATASET1_1 --slicelen=1024
  ```
- AMCNet
  ```
  python3 TPrime_torch_train.py --model=AMCNet --channel=random --snr_db=range --cp_path=./results_AMCNet --postfix=all20MHz --raw_path=../data/DATASET1_1 --slicelen=128
  ```
- MCFormer
  ```
   python3 TPrime_torch_train.py --model=MCformer --channel=random --snr_db=range --cp_path=./results_MCformer_largekernel --postfix=all20MHz --raw_path=../data/DATASET1_1 --slicelen=128 --debug
  ```

The main script `TPrime_torch_train.py` can be used in order to select what protocols to train on, what channel models (specific channels or random) to be applied, levels of noise, baseband signal ratio to be used to generate the dataset etc. Note that this code differs from the transformer one in the way slices are generated with a sequence lengt of 1 (i.e. M = 1). The script is used as follows:
```
TPrime_torch_train.py [-h] [--noise NOISE] [--snr_db SNR_DB [SNR_DB ...]] [--test] [--cp_path CP_PATH]
                           [--protocols {802_11ax,802_11b,802_11b_upsampled,802_11n,802_11g} [{802_11ax,802_11b,802_11b_upsampled,802_11n,802_11g} ...]]
                           [--channel {TGn,TGax,Rayleigh,relative,random,None,None}] --raw_path RAW_PATH [--slicelen SLICELEN] [--overlap_ratio OVERLAP_RATIO] [--postfix POSTFIX]
                           [--raw_data_ratio RAW_DATA_RATIO] [--normalize] [--model {baseline_cnn1d,AMCNet,ResNet,LSTM,MCformer}] [--out_mode {real,complex,real_invdim,real_ampphase}] [--debug]
                           [--useRay] [--num-workers NUM_WORKERS] [--use-gpu] [--address ADDRESS]
```
###### Arguments description
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
###### Competitor models source code
- CNN1D: variation on [O'Shea _et.al._, 2016](https://arxiv.org/abs/1602.04105)
- AMCNet: adapted from [official code](https://github.com/zjwXDU/AMC-Net/tree/main)
- ResNet: adapted from [available implementation](https://github.com/liuzhejun/ResNet-for-Radio-Recognition/tree/master)
- MCFormer: adapted from [official code](https://github.com/InterDigitalInc/Fireball/blob/8c98a40e6baba489ac9c028aa4fe71b2ae782f79/Playgrounds/MCformer/MCformer.ipynb)

#### Testing procedure
The script `preprocessing/TPrime_testing_SoTA.py` serves for testing all the different (T-PRIME LG and SM, CNN1D, ResNet, AMCNet, and MCFormer) models with simulated data generated through MATLAB. It is used as follows:
```
TPrime_testing_SoTA.py [-h] [--experiment EXPERIMENT] [--normalize] [--use_gpu] [--test_path]
```
The results will be saved in a file named test_results[experiment_extension].txt. The results for each architecture will be separated by the '%' character to facilitate later processing. For each architecture's results, you will find the testing accuracy for all four explored channel conditions (No channel, TGn, TGax, and Rayleigh) at different noise levels within the range of -30.0 to 30.0 dBs with 5.0 dBs increments. The order of the architectures in the results is the following: T-PRIME LG Trans., T-PRIME SM Trans., CNN1D, ResNet, AMCNet and MCFormer.

An example of how to use this script is the following command:
```
TPrime_testing_SoTA.py --experiment 3 --normalize --use_gpu --test_path ../data/DATASET1_1_TEST
```
###### Arguments description
When selecting the experiment number, 1 and 2 and 4 are only implemented to work with T-PRIME architectures.
```
optional arguments:
  -h, --help            show this help message and exit
  --experiment          Decide which models to test, 1 is for models trained for specific noise and channel conditions, 2 is for models specifically trained for a channel,
                        3 is for single model for all channel and noise conditions (with SoTA models comparison) and 4 is for inference time analysis (default: 3)
  --normalize           Use a layer norm as a first layer for CNN (default: False)
  --use-gpu             [DEPRECATED] Use gpu for inference (default: True)
  --test_path           Path to the dataset that will be used for testing. DATASET1_1_TEST contains the necessary data to test these models (default: '../data/DATASET1_1_TEST')
```

### OTA data
To train and test the models with OTA data the script ```preprocessing/TPrime_finetune.py``` needs to be used. This option can only be used with T-PRIME models. There are two options for training the models with OTA data. These are training the model from scratch with this data, or fine-tuning a preexisting model. To fine-tune a model provide the correct path to the model and include the ```--retrain``` flag.
#### Training procedure
Let's navigate to `preprocessing/` directory:
```
cd preprocessing/
```
The usage is described as follows:
```
usage: TPrime_finetune.py [-h] [--model_path MODEL_PATH] [--ds_path DS_PATH]
                          --datasets DATASETS [DATASETS ...]
                          [--dataset_ratio DATASET_RATIO] [--use_gpu]
                          [--transformer_version {v1,v2}]
                          [--transformer {sm,lg}]
                          [--test_mode {random_sampling,future}]
                          [--retrain] [--ota_dataset OTA_DATASET] --test
                          [--RMSNorm] [--back_class]
```
The Large (LG) and Small (SM) implementations of T-PRIME can now be trained with OTA data with the following commands:
- Large, LG (M=24, S=64)
```
python3 TPrime_finetune.py --model_path=/PATH/TO/REPOSITORY/t-prime/TPrime_transformer/model_cp/tprime_lg.pt --ds_path=../data/DATASET3_0 --datasets RM_573C_1 RM_573C_2 RM_142_1 RM_572C_1 RM_572C_2 --use_gpu --transformer_version v1 --transformer lg --test_mode=future --ota_dataset=ota_training --RMSNorm --back_class 
```
- Small, SM (M=64, S=128)
```
python3 TPrime_finetune.py --model_path=/PATH/TO/REPOSITORY/t-prime/TPrime_transformer/model_cp/tprime_sm.pt --ds_path=../data/DATASET3_0 --datasets RM_573C_1 RM_573C_2 RM_142_1 RM_572C_1 RM_572C_2 --use_gpu --transformer_version v1 --transformer sm --test_mode=future --ota_dataset=ota_training --RMSNorm --back_class
```
Notice these two models would be trained from scratch as the ```--retrain``` option is not included.
###### Optional arguments
```
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the trained model or to where to save the trained model from scratch with model name included (default: ../TPrime_transformer/model_cp)
  --ds_path DS_PATH     Path to the over the air datasets (default: ../data)
  --datasets DATASETS [DATASETS ...]
                        Dataset names to be used for training or test (required)
  --dataset_ratio DATASET_RATIO
                        Portion of the dataset used for training and validation (default: 1.0)
  --use_gpu             [DEPRECATED] Use gpu for fine-tuning and inference (default: false)
  --transformer_version {v1,v2}
                        Architecture of the model that will be finetuned. Options are v1 and v2. These refer to the two Transformer-based architectures available, without or
                        with [CLS] token (default: v1)
  --transformer {sm,lg}
                        Size of transformer to use, options available are small and large. If not defined CNN architecture will be used (default: CNN)
  --test_mode {random_sampling,future}
                        Get test from separate files (future) or a random sampling of dataset indexes (random_sampling) (default: random_sampling)
  --retrain             Load the selected model and fine-tune. If this is false the model will be trained from scratch and the model (default: false)
  --ota_dataset OTA_DATASET
                        Flag to add in results name to identify experiment (default: '')
  --test                If present, just test the provided model on OTA data (default: false)
  --RMSNorm             If present, apply RMS normalization on input signals while training and testing (default: false)
  --back_class          Train/Use model with background or noise class (default: false)
```
#### Testing procedure
When using this script for testing, use the ```--test``` flag and avoid the ```--retrain``` one. The results will be saved in a file named results_finetune[dataset_and_model_flags].pdf. So the resulting commands are now:
- Large, LG (M=24, S=64)
```
python3 TPrime_finetune.py --model_path=/PATH/TO/REPOSITORY/t-prime/TPrime_transformer/model_cp/tprime_lg.pt --ds_path=../data/DATASET3_0 --datasets RM_573C_1 RM_573C_2 RM_142_1 RM_572C_1 RM_572C_2 --use_gpu --transformer_version v1 --transformer lg --test_mode=future --ota_dataset=ota_testing --test --RMSNorm --back_class 
```
- Small, SM (M=64, S=128)
```
python3 TPrime_finetune.py --model_path=/PATH/TO/REPOSITORY/t-prime/TPrime_transformer/model_cp/tprime_sm.pt --ds_path=../data/DATASET3_0 --datasets RM_573C_1 RM_573C_2 RM_142_1 RM_572C_1 RM_572C_2 --use_gpu --transformer_version v1 --transformer sm --test_mode=future --ota_dataset=ota_testing --test --RMSNorm --back_class
```
### Overlapping data
For the overlapping case, the script necessary to train and test is ```TPrime_transformer/TPrime_overlapped.py```. This option can only be used with T-PRIME transformer models. There are two options for training the models with overlapping data. These are training the model from scratch with this data, or fine-tuning a preexisting model. To fine-tune a model provide the correct path to the model and include the ```--retrain``` flag.
#### Training procedure
The usage is described as follows:
```
usage: TPrime_overlapped.py [-h] [--model_path MODEL_PATH] [--ds_path DS_PATH]
                            --datasets DATASETS [DATASETS ...]
                            [--dataset_ratio DATASET_RATIO] [--use_gpu]
                            [--transformer {sm,lg}]
                            [--test_mode {random_sampling,future}]
                            [--retrain] [--ota_dataset OTA_DATASET] [--test]
                            [--RMSNorm] [--back_class]
```
The Large (LG) and Small (SM) implementations of T-PRIME can be trained with the overlapping data with the following commands:
- Large, LG (M=24, S=64)
```
python3 TPrime_overlapped.py --model_path=/PATH/TO/REPOSITORY/t-prime/TPrime_transformer/model_cp/tprime_lg_ov.pt --ds_path=../data --datasets DATASET3_2 DATASET3_1 DATASET3_0/RM_573C_1 DATASET3_0/RM_573C_2 DATASET3_0/RM_573C_power DATASET3_0/RM_142_1 DATASET3_0/RM_142_2 DATASET3_0/RM_572C_1 DATASET3_0/RM_572C_2 --use_gpu --transformer lg --test_mode=future --ota_dataset=ota_overlap_training --RMSNorm --back_class
```
- Small, SM (M=64, S=128)
```
python3 TPrime_overlapped.py --model_path=/PATH/TO/REPOSITORY/t-prime/TPrime_transformer/model_cp/tprime_sm_ov.pt --ds_path=../data --datasets DATASET3_2 DATASET3_1 DATASET3_0/RM_573C_1 DATASET3_0/RM_573C_2 DATASET3_0/RM_573C_power DATASET3_0/RM_142_1 DATASET3_0/RM_142_2 DATASET3_0/RM_572C_1 DATASET3_0/RM_572C_2 --use_gpu --transformer sm --test_mode=future --ota_dataset=ota_overlap_training --RMSNorm --back_class
```
###### Optional arguments
```
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the trained model or where to save the trained from scratch version and under which name (default: ./model_cp)
  --ds_path DS_PATH     Path to the over the air datasets
  --datasets DATASETS [DATASETS ...]
                        Dataset names to be used for training or test (required)
  --dataset_ratio DATASET_RATIO
                        Portion of the dataset used for training and validation (default: 1.0)
  --use_gpu             [DEPRECATED] Use gpu for fine-tuning and inference (default: false)
  --transformer {sm,lg}
                        Size of transformer to use, options available are small and large. If not defined lg architecture will be used (default: lg)
  --test_mode {random_sampling,future}
                        Get test from separate files (future) or a random sampling of dataset indexes (random_sampling). (default: random_sampling)
  --retrain             Load the selected model and fine-tune. If this is false the model will be trained from scratch and the model (default: false)
  --ota_dataset OTA_DATASET
                        Flag to add in results name to identify experiment. (default: '')
  --test                If present, just test the provided model on OTA data (default: false)
  --RMSNorm             If present, apply RMS normalization on input signals while training and testing (default: false)
  --back_class          Train/Use model with background or noise class (default: false)
```
#### Testing procedure
When using this script for testing, use the ```--test``` flag and avoid the ```--retrain``` one. The results will be outputed in the terminal. Since it is the overlapping case several metrics can be evaluated. These are: AUC of the classifier, exact accuracy (prediction and ground truth is exactly the same), soft accuracy (detecting at least one protocol correctly) and other metrics per class. The resulting commands are:
- Large, LG (M=24, S=64)
```
python3 TPrime_overlapped.py --model_path=/PATH/TO/REPOSITORY/t-prime/TPrime_transformer/model_cp/tprime_lg_ov.pt --ds_path=../data --datasets DATASET3_2 DATASET3_1 DATASET3_0/RM_573C_1 DATASET3_0/RM_573C_2 DATASET3_0/RM_573C_power DATASET3_0/RM_142_1 DATASET3_0/RM_142_2 DATASET3_0/RM_572C_1 DATASET3_0/RM_572C_2 --use_gpu --transformer lg --test_mode=future --ota_dataset=ota_overlap_testing --test --RMSNorm --back_class
```
- Small, SM (M=64, S=128)
```
python3 TPrime_overlapped.py --model_path=/PATH/TO/REPOSITORY/t-prime/TPrime_transformer/model_cp/tprime_sm_ov.pt --ds_path=../data --datasets DATASET3_2 DATASET3_1 DATASET3_0/RM_573C_1 DATASET3_0/RM_573C_2 DATASET3_0/RM_573C_power DATASET3_0/RM_142_1 DATASET3_0/RM_142_2 DATASET3_0/RM_572C_1 DATASET3_0/RM_572C_2 --use_gpu --transformer sm --test_mode=future --ota_dataset=ota_overlap_testing --test --RMSNorm --back_class
```

## Execution Pipeline
With trained models in hand, this section will guide you through the deployment process on the AIR-T platform for real-time inference. The initial step involves creating the necessary dependency environment to execute the scripts. To achieve this, you can utilize Conda to create the environment and install the required packages. Use the following command to create the Conda environment and install the required packages:
```
conda env create -f ./conda-envs/TPrime_conda_env_airT.yaml
conda activate airstack-py36
```

Additional instructions and environments are available from [DeepWave documentation](https://github.com/deepwavedigital/airstack-examples/tree/master/conda/environments).
The next step is running real time classification. To do so use the following script:
```
python3 Tprime_airT_run.py [--frequency] [--timeout] [--model_path] [--model_size] [--RMSNorm] [--tensorRT] [--protocols]
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
python Tprime_airT_run.py -fq 2.427e9 -t 180 --model_path TPrime_transformer/model_cp/model_lg_otaglobal_inf_RMSn_bckg_ft.pt --model_size lg --tensorRT --RMSNorm
```
### GUI
While the main program is running, you can also launch a real-time graphical user interface (GUI) using the Streamlit library. This GUI displays the model's predictions as they are made. To set this up, you'll need to install the ```streamlit``` Python package (follow the [installation guide](https://docs.streamlit.io/library/get-started/installation)). Once installed, simply run the following command to open the display in a web browser. 
```
streamlit run GUI.py
```


## OTA Data Capture and Processing Script

This Python script captures radio frequency (RF) signals using the AIR-T software-defined radio (SDR) device and processes them for further analysis. The captured data can be used for various purposes such as wireless communication research, signal analysis, and machine learning.

### Requirements

- Python 3.x
- SoapySDR library
- NumPy
- SciPy
- Matplotlib

### Usage

The script `OTA_capture.py` provides functionalities to capture RF signals and save them to files for later analysis. Below are the available options:

- **-nf, --nfiles**: Number of files to capture, each file is approximately 10ms long.
- **-fq, --frequency**: Center frequency for capturing signals. Default is set to 2.457 GHz.
- **-p, --plot**: Flag to plot a spectrogram of the last file captured.
- **-s, --standard**: Specify the 802.11 standard. Results will be saved to a specific directory based on the standard.
- **-d, --directory**: Directory to save captures. Default is `/home/deepwave/Research/DSTL/OTA_dataset`.

Example usage:

```
python capture_and_process.py --nfiles 10 --frequency 2.4e9 --plot
```
This command will capture 10 files, each approximately 10ms long, at a center frequency of 2.4 GHz and plot the spectrogram of the last captured file.

### File Structure

The captured files are saved in binary format (.bin) and named with a timestamp prefix. The files are saved in the specified directory, with subdirectories created based on the specified 802.11 standard.

### Signal Processing

The script performs the following signal processing steps:

1. **Receive Signal**: Initializes the AIR-T receiver using SoapyAIRT, sets up the stream, and starts streaming samples.
2. **Process Signal**: Converts received samples to complex numbers, applies a low-pass filter, and resamples the signal.
3. **Save Signal**: Saves the processed signal to binary files in the specified directory.
4. **Plot Spectrogram**: If the `--plot` option is enabled, plots the spectrogram of the captured signal using Matplotlib.

### Notes

- Ensure that the AIR-T device is properly connected and configured before running the script.
- Adjust the script parameters according to your specific requirements, such as the number of files to capture and the center frequency.

