import scipy.io as sio
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from glob import glob
import os
proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
import sys
sys.path.append(proj_root_dir)
import argparse
from TPrime_transformer.model_transformer import TransformerModel
from cnn_baseline.model_cnn1d import Baseline_CNN1D
from tqdm import tqdm
# CONFIG
TEST_DATA_PATH = '../data/DSTL_DATASET_1_1_TEST'
TRANS_PATH = '../TPrime_transformer/model_cp'
CNN_PATH = '../baseline_models/results_slice512'
MODELS = ["Trans. (64 x 128) [6.8M params]", "Trans. (24 x 64) [1.6M params]", "CNN (1 x 512) [4.1M params]"]
PROTOCOLS = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
#CHANNELS = ['None', 'TGn', 'TGax', 'Rayleigh']
SNR = [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
CHANNELS = ['None']
MODE = 'TensorRT' # choices=['pytorch', 'TensorRT']
if MODE == 'TensorRT':
    TEST_DATA_PATH = '../data/DSTL_DATASET_1_1_TEST'
    TRANS_PATH = '../TPrime_transformer/model_cp'
    CNN_PATH = '../baseline_models/results_slice512'
    import tensorrt as trt
    from preprocessing.inference.onnx2plan import onnx2plan
    from preprocessing.inference.plan_bench import plan_bench
    import preprocessing.inference.trt_utils as trt_utils

INPUT_NODE_NAME = 'input_buffer'  # (for TensorRT) User defined name of input node
OUTPUT_NODE_NAME = 'output_buffer'  # User defined name of output node
ONNX_VERSION = 10  # the ONNX version to export the model to
np.random.seed(4389) # for reproducibility


def apply_AWGN(snr_dbs, sig):
    # apply AWGN noise based on the levels specified when instantiating the dataset object
    # if more than one is specified, a random level is picked among the one specified
    # first, let's compute signal power (in Watts) as rms(sig)**2
    rms = np.sqrt(np.mean(np.abs(sig) ** 2))
    sig_W = rms ** 2  # power of signal in Watts
    # convert signal power in dBW
    sig_dbW = 10 * np.log10(sig_W / 1)
    # now compute the relative noise power based on the specified SNR level
    noise_dbW = sig_dbW - float(snr_dbs)
    # noise variance = noise power
    noise_var = 10 ** (noise_dbW / 10)
    # obtain noise standard deviation
    noise_std = np.sqrt(noise_var)
    # now generate the noise samples.
    # NOTE: since we are generating ** complex ** noise samples, the nominal variance for a
    # normal complex distribution is equal to 1/2 instead of 1.
    # https://en.wikipedia.org/wiki/Complex_normal_distribution
    # var = std**2 = 1/2 ---> std = sqrt(1/2) = 1/sqrt(2)
    complex_std = noise_std * 1 / np.sqrt(2)
    noise_samples = np.random.normal(0, complex_std, size=sig.shape) + 1j * np.random.normal(0, complex_std,
                                                                                                size=sig.shape)
    # now apply noise to the signal
    noisy_sig = sig + noise_samples
    return noisy_sig

# Function to change the shape of obs
# the input is obs with shape (channel, slice)
def chan2sequence(obs):
    seq = np.empty((obs.size))
    seq[0::2] = obs[0]
    seq[1::2] = obs[1]
    return seq

def validate(model, class_map, input_shape, seq_len, sli_len, channel, cnn=False, mode='pytorch', plan_file_name='', input_dtype=np.float32, max_samples_p_protocol=99999):
    assert((mode == 'pytorch') or (mode == 'TensorRT'))
    correct = np.zeros(len(SNR))
    total_samples = 0
    prev_time = time.time()
    if mode == 'TensorRT':
        # Setup the pyCUDA context
        trt_utils.make_cuda_context()

    for p in PROTOCOLS:
        print('Protocol ',p)
        path = os.path.join(TEST_DATA_PATH, p) if channel == 'None' else os.path.join(TEST_DATA_PATH, p, channel)
        mat_list = sorted(glob(os.path.join(path, '*.mat'))) if channel == 'None' else sorted(glob(os.path.join(path, '*.npy')))
        for signal_path in tqdm(mat_list[:max_samples_p_protocol], desc=f"TEST signal dataset, protocol {p}..."):
            sig = sio.loadmat(signal_path) if channel == 'None' else np.load(signal_path)
            if channel == 'None':
                sig = sig['waveform']
            len_sig = sig.shape[0]
            for i, dBs in enumerate(SNR):
                noisy_sig = apply_AWGN(dBs, sig)
                len_sig = noisy_sig.shape[0]
                # stack real and imag parts
                obs = np.stack((noisy_sig.real, noisy_sig.imag))
                obs = np.squeeze(obs, axis=2)
                # zip the I and Q values
                if not cnn: # Transformer architecture
                    obs = chan2sequence(obs)
                    # generate idxs for split
                    idxs = list(range(seq_len*sli_len, len_sig, seq_len*sli_len))
                    # split stream in sequences
                    obs = np.split(obs, idxs)[:-1]
                    # split each sequence in slices
                    for j, seq in enumerate(obs):
                        obs[j] = np.split(seq, seq_len)
                else: # CNN
                    # generate idxs for split
                    idxs = list(range(sli_len, len_sig, sli_len))
                    obs = np.split(obs, idxs, axis=1)[:-1]

                if mode == 'pytorch':
                    # create batch of sequences
                    X = np.asarray(obs)
                    # predict
                    X = torch.from_numpy(X)
                    y = np.empty(len(idxs))
                    y.fill(class_map[p])
                    y = torch.from_numpy(y)
                    X = X.to(model.device.type)
                    y = y.to(model.device.type)
                    pred = model(X.float())
                    # add correct ones
                    correct[i] += (pred.argmax(1) == y).type(torch.float).sum().item()

                elif mode == 'TensorRT':
                    # Use pyCUDA to create a shared memory buffer that will receive samples from the
                    # AIR-T to be fed into the neural network.
                    batch_size, seq_len, cplx_samples = input_shape
                    buff_len = seq_len * cplx_samples * batch_size
                    sample_buffer = trt_utils.MappedBuffer(buff_len, input_dtype)

                    # Set up the inference engine. Note that the output buffers are created for
                    # us when we create the inference object.
                    dnn = trt_utils.TrtInferFromPlan(plan_file_name, batch_size,
                                                     sample_buffer, verbose=False)

                    # Populate input buffer with test data
                    for bix in range(0,len(obs),batch_size):
                        X = np.asarray(obs[bix:bix+batch_size])
                        dnn.input_buff.host[:] = X.flatten().astype(input_dtype)
                        y = np.empty(batch_size)
                        y.fill(class_map[p])
                        dnn.feed_forward()
                        pred = dnn.output_buff.host.reshape((batch_size,4))
                        # add correct ones
                        correct[i] += (pred.argmax(1) == y).sum().item()
                if i == 0:
                    total_samples += len(idxs)

        print("\n --- %s seconds for protocol ---" % (time.time() - prev_time))
        prev_time = time.time()
    return correct/total_samples*100

if __name__ == "__main__":

    y_trans_lg, y_trans_sm, y_cnn = [], [], []
    models = ['trans_lg', 'trans_sm', 'cnn']
    #models = ['cnn']
    class_map = dict(zip(PROTOCOLS, range(len(PROTOCOLS))))
    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for channel in CHANNELS:
        for m in models:

            if m == 'cnn':
                cnn = Baseline_CNN1D(classes=len(PROTOCOLS), numChannels=2, slice_len=512)
                cnn.load_state_dict(torch.load(f"{CNN_PATH}/model.cnn.{channel}.pt", map_location=torch.device('cpu'))['model_state_dict'])
                cnn.eval()
                seq_len = 1
                sli_len = 512
                isCNN = True
                ONNX_FILE_NAME = os.path.join(CNN_PATH, f"model.cnn.{channel}.onnx")
                slice_in = np.random.random((batch_size, 2, sli_len))
                model = cnn

            elif m == 'trans_lg':
                model_lg = TransformerModel(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2, use_pos=False)
                model_lg.load_state_dict(torch.load(f"{TRANS_PATH}/model{channel}_lg.pt", map_location=torch.device('cpu'))['model_state_dict'])
                model_lg.eval()
                seq_len = 64
                sli_len = 128*2
                isCNN = False
                ONNX_FILE_NAME = os.path.join(TRANS_PATH, f"model{channel}_lg.onnx")
                slice_in = np.random.random((batch_size, seq_len, sli_len))
                model = model_lg

            elif m == 'trans_sm':
                model_sm = TransformerModel(classes=len(PROTOCOLS), d_model=64*2, seq_len=24, nlayers=2, use_pos=False)
                model_sm.load_state_dict(torch.load(f"{TRANS_PATH}/model{channel}_sm.pt", map_location=torch.device('cpu'))['model_state_dict'])
                model_sm.eval()
                seq_len = 24
                sli_len = 64*2
                isCNN = False
                ONNX_FILE_NAME = os.path.join(TRANS_PATH, f"model{channel}_sm.onnx")
                slice_in = np.random.random((batch_size, seq_len, sli_len))
                model = model_sm

            model.to(device)

            if MODE == 'TensorRT':
                slice_t = torch.Tensor(slice_in)
                slice_t = slice_t.to(model.device.type)
                # Let's produce the ONNX schema of the current model
                torch.onnx.export(model, slice_t, ONNX_FILE_NAME, export_params=True,
                                  opset_version=ONNX_VERSION, do_constant_folding=True,
                                  input_names=[INPUT_NODE_NAME], output_names=[OUTPUT_NODE_NAME],
                                  dynamic_axes={INPUT_NODE_NAME: {0: 'batch_size'},
                                                OUTPUT_NODE_NAME: {0: 'batch_size'}})
                # Then create the relative plan file using TensorRT
                MAX_WORKSPACE_SIZE = 1073741824  # 1 GB for example
                onnx2plan(onnx_file_name=ONNX_FILE_NAME, nchan=slice_in.shape[1],
                          input_len=slice_in.shape[2],  logger=trt.Logger(trt.Logger.WARNING),
                          MAX_BATCH_SIZE=slice_in.shape[0], MAX_WORKSPACE_SIZE=MAX_WORKSPACE_SIZE,
                          BENCHMARK=True)
                print('Running Inference Benchmark')
                plan_file = ONNX_FILE_NAME.replace('.onnx', '.plan')
                plan_bench(plan_file_name=plan_file, cplx_samples=slice_in.shape[2], num_chans=slice_in.shape[1],
                           batch_size=slice_in.shape[0], num_batches=512, input_dtype=np.float32)
            else:
                plan_file = ''


            y = validate(model,
                         class_map,
                         input_shape=slice_in.shape,
                         seq_len=seq_len,
                         sli_len=sli_len,
                         channel=channel,
                         cnn=isCNN,
                         mode=MODE,
                         plan_file_name=plan_file,
                         max_samples_p_protocol=50)

            if m == 'cnn':
                y_cnn.append(y)
                print(f'Accuracy values for channel {channel} and cnn architecture are: ', y_cnn[-1])


            elif m == 'trans_lg':
                y_trans_lg.append(y)
                print(f'Accuracy values for channel {channel} and large architecture are: ', y_trans_lg[-1])

            elif m == 'trans_sm':
                y_trans_sm.append(y)
                print(f'Accuracy values for channel {channel} and small architecture are: ', y_trans_sm[-1])


    
    fig, ax = plt.subplots(2, 2, figsize = (12, 6))

    for i in range(2):
        for j in range(2):
            ax[i][j].plot(SNR, y_trans_lg[i*2+j], color='#000000', linestyle='solid', marker='o', label=MODELS[0])
            ax[i][j].plot(SNR, y_trans_sm[i*2+j], color='#56B4E9', linestyle='dashed', marker='v', label=MODELS[1])
            ax[i][j].plot(SNR, y_cnn[i*2+j], color='#D55E00', linestyle='dashdot', marker='^', label=MODELS[2])
            ax[i][j].set_title(CHANNELS[i*2+j])
            ax[i][j].set_xlabel('SNR (dBs)')
            ax[i][j].set_ylabel('Accuracy (%)')
            ax[i][j].set_ylim(15,105)
            ax[i][j].grid()     
    plt.suptitle('Results comparison between different architectures')
    fig.legend(MODELS, bbox_to_anchor=(0.87, 0.02), ncols=3, labelspacing=1)
    plt.tight_layout() 
    if args.experiment == '1':
        img_name = 'Testing_noiseandchannel.png'
    elif args.experiment == '2':
        img_name = 'Testing_modelperchannel.png'
    else: # experiment 3
        img_name = 'Testing_onemodel.png'
    plt.savefig(img_name) 
