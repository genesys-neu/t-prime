#!/bin/bash
# Baseline T-PRIME (synthetic + AWGN) vs CMAT T-PRIME (synthetic + measured HCRO channels + AWGN).
# Identical hyperparameters/splits; the only difference is --use_channel_aug.
# Env overrides: RAW (synthetic dataset dir), TAG (run-name prefix), EPOCHS (max), RATIO (fraction of files used),
# PATIENCE/MIN_DELTA (early stopping; 0 = off), BASE_WORKERS/CMAT_WORKERS (DataLoader processes),
# SNR_MIN/SNR_MAX (training AWGN range, dB).
cd /home/sat3737/t-prime-ext/TPrime_transformer
PY=/home/sat3737/venvs/tprime/bin/python
# local copy of /mnt/ssd/x410_recordings/CMAT_ChannelExtraction/0937_OFDM/estimated_chans (exFAT listdir is ~1000x slower)
CHANS=/home/sat3737/t-prime-ext/data/HCRO_CHANNELS/0937_OFDM
RAW=${RAW:-../data/HCRO_SYNTH}
TAG=${TAG:-}
EPOCHS=${EPOCHS:-120}
RATIO=${RATIO:-1.0}
PATIENCE=${PATIENCE:-0}
MIN_DELTA=${MIN_DELTA:-0}
BASE_WORKERS=${BASE_WORKERS:-8}
CMAT_WORKERS=${CMAT_WORKERS:-8}
SNR_MIN=${SNR_MIN:-5}
SNR_MAX=${SNR_MAX:-30}
COMMON="--wchannel=None --use-gpu --Layers=6 --Epochs=$EPOCHS --Learning_rate=0.0002 --Batch_size=122 --Slice_length=64 \
 --Sequence_length=16 --Positional_encoder=True --use_sota --sota_type=awgn --awgn_snr_min=$SNR_MIN --awgn_snr_max=$SNR_MAX --raw_path=$RAW --dataset_ratio=$RATIO \
 --channel_path=$CHANS --patience=$PATIENCE --min_delta=$MIN_DELTA --protocols 5GNR LoRa Nbiot OFDM ssm Zigbee"
$PY -u TPrime_transformer_train.py $COMMON --postfix=${TAG}baseline --loader_workers=$BASE_WORKERS --cp_path=../checkpoints/hcro/${TAG}baseline > ../logs/${TAG}baseline.log 2>&1 &
$PY -u TPrime_transformer_train.py $COMMON --postfix=${TAG}cmat --use_channel_aug --loader_workers=$CMAT_WORKERS --cp_path=../checkpoints/hcro/${TAG}cmat > ../logs/${TAG}cmat.log 2>&1 &
wait
