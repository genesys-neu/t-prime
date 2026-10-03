#!/bin/bash
# Wait for the SNR 5-50 dB runs, evaluate them, and execute the SNR-range comparison notebook.
cd /home/sat3737/t-prime-ext/cmat_hcro
PY=/home/sat3737/venvs/tprime/bin
TRAIN='^/home/sat3737/venvs/tprime/bin/python -u TPrime_transformer_train.py .*--postfix='
while pgrep -f "${TRAIN}snr50_baseline " >/dev/null || pgrep -f "${TRAIN}snr50_cmat " >/dev/null; do sleep 30; done
echo "training finished $(date)"
# validation splits are identical across runs (same seed/files); reuse the ssmfix_ split pickles
$PY/python eval_tprime.py --models snr50_baseline snr50_cmat --syn_dir ../data/HCRO_SYNTH_v2 --val_tag ssmfix_ \
    --out ../results/hcro_snr50 || { echo "EVAL FAILED"; exit 1; }
$PY/python build_notebook.py --variant snr50 || { echo "BUILD FAILED"; exit 1; }
$PY/jupyter nbconvert --to notebook --execute --inplace compare_tprime_cmat_snr50.ipynb || { echo "NOTEBOOK FAILED"; exit 1; }
echo "PIPELINE DONE $(date)"
