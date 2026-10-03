#!/bin/bash
# Wait for the shift-fixed CMAT run, then evaluate it against the SSM-fixed baseline and execute the notebook.
cd /home/sat3737/t-prime-ext/cmat_hcro
PY=/home/sat3737/venvs/tprime/bin
TRAIN='^/home/sat3737/venvs/tprime/bin/python -u TPrime_transformer_train.py .*--postfix='
while pgrep -f "${TRAIN}ssmfix_shiftfix_cmat " >/dev/null; do sleep 30; done
echo "training finished $(date)"
$PY/python eval_tprime.py --models ssmfix_baseline ssmfix_shiftfix_cmat --syn_dir ../data/HCRO_SYNTH_v2 --val_tag ssmfix_ \
    --out ../results/hcro_final || { echo "EVAL FAILED"; exit 1; }
$PY/jupyter nbconvert --to notebook --execute --inplace compare_tprime_cmat.ipynb || { echo "NOTEBOOK FAILED"; exit 1; }
echo "PIPELINE DONE $(date)"
