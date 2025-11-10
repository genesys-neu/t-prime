# DroneDetect End-to-End Training and Evaluation Workflow

This guide summarizes the full pipeline the DroneDetect team is using inside this repo: preprocessing large OTA captures, training the base transformer, visualizing metrics, fine-tuning on OTA-focused data, and comparing pre/post finetune performance.

All commands below assume you are in the repo root (e.g., `~/Projects/t-prime`) with the `t-prime` Conda environment activated; replace these with your actual project path and Conda environment name as needed.

## 1. Preprocess OTA IQ Captures

Use `preprocess_large_dat_for_trprime.py` to chunk raw IQ files into TPrime datasets. Adjust the `--src`, `--mode`, `--condition`, and quotas to match the capture you are curating.

```bash
python preprocessing/preprocess_large_dat_for_trprime.py \
  --src /mnt/dm-3/DroneDetect/DroneDetectV2_byModel \
  --mode HO \
  --condition CLEAN HO \
  --samples-per-file 8192 \
  --num-examples-per-class 16000 \
  --out-root data \
  --seed 4389
```

This creates folders such as `data/CLEAN_HO_4000_{train,eval}` with subdirectories per protocol/model (e.g., `AIR`, `INS`, `MIN`, `MP1`, `MP2`, `PHA`).

## 2. Train the Base Transformer

Train on the clean dataset using `TPrime_transformer_train.py`. The example below matches the configuration used for the baseline checkpoint `modelNone_30_lg.pt`.

```bash
python TPrime_transformer/TPrime_transformer_train.py \
  --snr_db 30 \
  --wchannel None \
  --use-gpu \
  --postfix CLEAN_HO_4000 \
  --raw_path ./data/CLEAN_HO_4000_train \  # Path is relative to the repo root
  --cp_path ./model_cp/CLEAN_HO_4000 \
  --Layers 2 \
  --Epochs 15 \
  --Learning_rate 0.0002 \
  --Batch_size 122 \
  --Slice_length 128 \
  --Sequence_length 64 \
  --Positional_encoder False
```

Outputs land in `TPrime_transformer/model_cp/CLEAN_HO_4000/` alongside confusion-matrix pickles and Ray logs if enabled.

## 3. Visualize Baseline Metrics

Open `TPrime_transformer/visualize_results.ipynb` in Jupyter. Use the widgets/cells to:

1. Point to the checkpoint directory created above (`model_cp/CLEAN_HO_4000`).
2. Load `modelNone_30_lg.pt` (or whichever best checkpoint you selected).
3. Run the provided plotting cells to inspect learning curves, confusion matrices, and spectrum visualizations.

Export any figures you want to keep as part of the baseline report before fine-tuning.

## 4. Fine-Tune on OTA-Focused Data

Fine-tune the pretrained model on OTA captures (e.g., BOTH_HO_4000). The finetune script now keeps the original checkpoint untouched and appends `_finetuned` to all new checkpoints.

```bash
python preprocessing/TPrime_finetune.py \
  --model_path /path/to/your/project/TPrime_transformer/model_cp/CLEAN_HO_4000/modelNone_30_lg.pt \  # <-- Update this path to your environment
  --raw_path /path/to/your/data/BOTH_HO_4000_train \  # <-- Replace with your actual dataset path
  --protocols AIR INS MIN MP1 MP2 PHA \
  --transformer lg \
  --transformer_version v1 \
  --use_gpu --gpu_device 1 \
  --retrain
```

Artifacts produced:

- `modelNone_30_lg_finetuned_last.pt` and `modelNone_30_lg_finetuned_best.pt` — checkpoints saved each epoch/best val accuracy.
- Legacy-style `modelNone_30_lg_finetuned.pt` (uses the previous checkpoint naming convention for compatibility with older scripts).
- Training/validation confusion matrix PDFs under `preprocessing/training/` with labels derived from the actual protocol list.

## 5. Evaluate Pre/Post Finetune

Use `TPrime_transformer/transformer_eval.ipynb` to compare checkpoints against the OTA evaluation split (e.g., `data/BOTH_HO_4000_eval`). Suggested workflow:

1. Duplicate the notebook kernel tabs: one for the baseline model (`modelNone_30_lg.pt`) and one for the finetuned model (`modelNone_30_lg_finetuned_best.pt`).
2. In each notebook:
   - Set `MODEL_PATH` (or the equivalent cell) to the checkpoint you want to test.
   - Point the dataset path to `../data/BOTH_HO_4000_eval` and reuse the protocol list `['AIR','INS','MIN','MP1','MP2','PHA']`.
   - Run the evaluation cells to obtain accuracy/recall/precision and confusion matrices.
3. Compare the before/after metrics to quantify OTA transfer gains.

Optional: log both sets of metrics into your experiment tracker, or overlay the confusion matrices to highlight which classes benefited most from fine-tuning.

## Notes & Tips

- The transformer consumes sequences of 64 slices × 128 complex samples (~8,192 IQ samples per item). At 60 MS/s, each training example spans ~136 µs; the 16,000-sequence per-class OTA dataset represents ≈2.18 seconds of capture per class.
- `preprocess_large_dat_for_trprime.py` enforces quotas per class; rerun it whenever new OTA captures arrive to keep training/eval splits balanced.
- `TPrime_finetune.py` writes `_last` and `_best` checkpoints every epoch; monitor validation accuracy to determine when to stop or adjust learning rates.



