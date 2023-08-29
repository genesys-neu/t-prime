import numpy as np
import torch
import os
proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(os.pardir, os.pardir)))
import sys
sys.path.append(proj_root_dir)
from dstl.preprocessing.TPrime_dataset import TPrimeDataset
from model_cnn1d import Baseline_CNN1D

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

"""
homedir=os.path.expanduser('~')
PATH = os.path.join(homedir, 'Research/DSTL/dstl/baseline_models/results/Rayleigh_ds11_tiny15/')
model_file_name = 'model.best.pt'
model_path = os.path.join(PATH, model_file_name)

protocols = ['802_11ax', '802_11b', '802_11n', '802_11g']
ds_test = TPrimeDataset(protocols, ds_path=os.path.join(homedir,'Research/DSTL/DSTL_DATASET_1_0'), ds_type='test', snr_dbs=snr_dBs, slice_len=slice_len, slice_overlap_ratio=0.5, override_gen_map=False)

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
"""


device = 'cuda' if torch.cuda.is_available() else 'cpu'
for slice_len in [128, 256, 512]:
    model = Baseline_CNN1D(classes=Nclass, numChannels=num_channels, slice_len=slice_len)
    model.eval()
    model.to(device)
    slice_in = np.random.random((1,2,slice_len))
    slice_t = torch.Tensor(slice_in)
    slice_t = slice_t.to(model.device.type)
    out = model(slice_t.float())
    mean_ms, std_ms = timing_inference_GPU(slice_t, model)
    print('Slice len', slice_len)
    print("Mean (ms):", mean_ms, "Std ", std_ms)

homedir=os.path.expanduser('~')
PATH = os.path.join(homedir, 'Research/DSTL/dstl/baseline_models/results_slice512/sweep_TGn_all20MHz_half/slice512_snr30')
model_file_name = 'model.best.pt'
model_path = os.path.join(PATH, model_file_name)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(model.device.type) # reload the model on the appropriate device

ONNX_FILE_NAME = 'dstl_baseline_CNN1D.onnx'  # File name to save network
INPUT_NODE_NAME = 'input_buffer'  # User defined name of input node
OUTPUT_NODE_NAME = 'output_buffer'  # User defined name of output node
ONNX_VERSION = 10  # the ONNX version to export the model to

torch.onnx.export(model, slice_t, ONNX_FILE_NAME, export_params=True,
                  opset_version=ONNX_VERSION, do_constant_folding=True,
                  input_names=[INPUT_NODE_NAME], output_names=[OUTPUT_NODE_NAME],
                  dynamic_axes={INPUT_NODE_NAME: {0: 'batch_size'},
                                OUTPUT_NODE_NAME: {0: 'batch_size'}})



