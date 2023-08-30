import numpy as np
import torch
import os
proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(os.pardir, os.pardir)))
import sys
sys.path.append(proj_root_dir)
from preprocessing.TPrime_dataset import TPrimeDataset
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
    model.eval()
    model.to(device)
    slice_in = np.random.random((1,seq_len,d_model))
    slice_t = torch.Tensor(slice_in)
    slice_t = slice_t.to(model.device.type)
    out = model(slice_t.float())
    mean_ms, std_ms = timing_inference_GPU(slice_t, model)
    print('Shape', slice_in.shape)
    print('Slice len', slice_len)
    print("Mean (ms):", mean_ms, "Std ", std_ms)

    homedir=os.path.expanduser('~')
    PATH = os.path.join(homedir, '/home/deepwave/Research/DSTL/t-prime/TPrime_transformer/model_cp')
    model_file_name = 'model.best_sm.pt' if slice_len == 64 else 'model.best_lg.pt'
    model_path = os.path.join(PATH, model_file_name)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    SIZE = 'SMALL' if slice_len == 64 else 'LARGE'
    ONNX_FILE_NAME = 'TPrime_baseline_transformer_'+str(SIZE)+'.onnx'  # File name to save network
    INPUT_NODE_NAME = 'input_buffer'  # User defined name of input node
    OUTPUT_NODE_NAME = 'output_buffer'  # User defined name of output node
    ONNX_VERSION = 10  # the ONNX version to export the model to

    torch.onnx.export(model, slice_t, ONNX_FILE_NAME, export_params=True,
                      opset_version=ONNX_VERSION, do_constant_folding=True,
                      input_names=[INPUT_NODE_NAME], output_names=[OUTPUT_NODE_NAME],
                      dynamic_axes={INPUT_NODE_NAME: {0: 'batch_size'},
                                    OUTPUT_NODE_NAME: {0: 'batch_size'}})
    # TODO: check the typing warning


