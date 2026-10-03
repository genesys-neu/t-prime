#!/bin/bash
# Re-evaluate every model on the current real OOD set and re-execute both comparison notebooks.
cd /home/sat3737/t-prime-ext/cmat_hcro
PY=/home/sat3737/venvs/tprime/bin
set -e
# pre-fix models (original SSM data; validation on the original synthetic set)
$PY/python eval_tprime.py --models baseline cmat_ep77 --syn_dir ../data/HCRO_SYNTH --val_tag '' --out ../results/hcro_interim_ep77
# SSM rate fix, CMAT still trained with the shift bug
$PY/python eval_tprime.py --models ssmfix_baseline ssmfix_cmat --syn_dir ../data/HCRO_SYNTH_v2 --val_tag ssmfix_ --out ../results/hcro_ssmfix
# final models: both fixes, AWGN 5-30 dB
$PY/python eval_tprime.py --models ssmfix_baseline ssmfix_shiftfix_cmat --syn_dir ../data/HCRO_SYNTH_v2 --val_tag ssmfix_ --out ../results/hcro_final
# both fixes, AWGN 5-50 dB
$PY/python eval_tprime.py --models snr50_baseline snr50_cmat --syn_dir ../data/HCRO_SYNTH_v2 --val_tag ssmfix_ --out ../results/hcro_snr50
for v in final snr50; do $PY/python build_notebook.py --variant $v; done
$PY/jupyter nbconvert --to notebook --execute --inplace compare_tprime_cmat.ipynb
$PY/jupyter nbconvert --to notebook --execute --inplace compare_tprime_cmat_snr50.ipynb
echo "PIPELINE DONE $(date)"
