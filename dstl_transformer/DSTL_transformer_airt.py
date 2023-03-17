import numpy as np
import torch
import os
proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(os.pardir, os.pardir)))
import sys
sys.path.append(proj_root_dir)
from dstl.preprocessing.DSTL_dataset import DSTLDataset
from model_transformer import TransformerModel

def timing_inference_GPU(dummy_input, model):
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn


Nclass = 4
num_channels = 2
snr_dBs = [30]
transformer_layers = 2
pos_encoder = False
seq_len = 24

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for slice_len in [64, 128]:
    d_model = 2 * slice_len
    if slice_len == 128:
        seq_len = 64
    model = TransformerModel(classes=Nclass, d_model=d_model, seq_len=seq_len, nlayers=transformer_layers, use_pos=pos_encoder)
    model.to(device)
    slice_in = np.random.random((1,seq_len,d_model))
    slice_t = torch.Tensor(slice_in)
    slice_t = slice_t.to(model.device.type)
    out = model(slice_t.float())
    mean_ms, std_ms = timing_inference_GPU(slice_t, model)
    print('Slice len', slice_len)
    print("Mean (ms):", mean_ms, "Std ", std_ms)




