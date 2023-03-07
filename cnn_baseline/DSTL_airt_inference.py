import numpy as np
import torch
import sys
sys.path.append('/home/deepwave/Research/DSTL')
from dstl.preprocessing.DSTL_dataset import DSTLDataset
from model_cnn1d import Baseline_CNN1D
import os

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
num_feats = 1
slice_len = 128
snr_dBs = [30]

PATH = '/home/deepwave/Research/DSTL/dstl/cnn_baseline/results/no_noise/'
model_file_name = 'model.best.pt'
model_path = os.path.join(PATH, model_file_name)

protocols = ['802_11ax', '802_11b', '802_11n', '802_11g']
ds_test = DSTLDataset(protocols, ds_path='/home/deepwave/Research/DSTL/DSTL_DATASET_1_0', ds_type='test', snr_dbs=snr_dBs, slice_len=slice_len, slice_overlap_ratio=0.5, override_gen_map=False)

model = Baseline_CNN1D(classes=Nclass, numChannels=num_channels, slice_len=slice_len)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(model.device.type) # reload the model on the appropriate device
model.eval()    # set the evaluation mode
rand_sixs = np.random.choice(len(ds_test))

slice_in, true_label = ds_test[rand_sixs]
slice_in = slice_in[np.newaxis,:,:]
slice_t = torch.Tensor(slice_in)
slice_t = slice_t.to(model.device.type)
out = model(slice_t.float())
mean_ms, std_ms = timing_inference_GPU(slice_t, model)
print("Mean (ms):", mean_ms, "Std ", std_ms)


