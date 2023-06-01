import scipy.io as sio
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from glob import glob
import os
proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(os.pardir, os.pardir)))
import sys
sys.path.append(proj_root_dir)
sys.path.insert(0, '../')
import argparse
from dstl_transformer.model_transformer import TransformerModel
from cnn_baseline.model_cnn1d import Baseline_CNN1D

# CONFIG
TEST_DATA_PATH = '/home/miquelsirera/Desktop/dstl/data/DSTL_DATASET_1_1_TEST'
TRANS_PATH = '/home/miquelsirera/Desktop/dstl/dstl_transformer/model_cp'
CNN_PATH = '/home/miquelsirera/Desktop/dstl/cnn_baseline/results_slice512'
MODELS = ["Trans. (64 x 128) [6.8M params]", "Trans. (24 x 64) [1.6M params]", "CNN (1 x 512) [4.1M params]"]
PROTOCOLS = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
CHANNELS = ['None', 'TGn', 'TGax', 'Rayleigh']
SNR = [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
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

def validate(model, class_map, seq_len, sli_len, channel, cnn=False):
    correct = np.zeros(len(SNR))
    total_samples = 0
    prev_time = time.time()
    for p in PROTOCOLS:
        path = os.path.join(TEST_DATA_PATH, p) if channel == 'None' else os.path.join(TEST_DATA_PATH, p, channel)
        mat_list = sorted(glob(os.path.join(path, '*.mat'))) if channel == 'None' else sorted(glob(os.path.join(path, '*.npy')))
        for signal_path in mat_list:
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
                    idxs = list(range(seq_len*sli_len*2, len_sig, seq_len*sli_len*2)) # *2 because I and Q are already zipped
                    # split stream in sequences
                    obs = np.split(obs, idxs)[:-1]
                    # split each sequence in slices
                    for j, seq in enumerate(obs):
                        obs[j] = np.split(seq, seq_len)
                else: # CNN
                    # generate idxs for split
                    idxs = list(range(sli_len, len_sig, sli_len))
                    obs = np.split(obs, idxs, axis=1)[:-1]
                # create batch of sequences
                X = np.asarray(obs)
                # predict
                X = torch.from_numpy(X)
                X = X.to(device)
                y = np.empty(len(idxs))
                y.fill(class_map[p])
                y = torch.from_numpy(y)
                y = y.to(device)
                pred = model(X.float())
                # add correct ones
                correct[i] += (pred.argmax(1) == y).type(torch.float).sum().item()
                if i == 0:
                    total_samples += len(idxs)
        print("--- %s seconds for protocol ---" % (time.time() - prev_time))
        prev_time = time.time()
    return correct/total_samples*100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default='3', choices=['1', '2', '3'], help="Decide which models to test, 1 is for specific model per \
                        noise and channel, 2 is for specific model per channel and 3 is for single model for all channel and noise conditions")
    parser.add_argument("--normalize", action='store_true', default=False, help="Use a layer norm as a first layer.")
    parser.add_argument("--use_gpu", action='store_true', default=False, help="Use gpu for inference")
    args, _ = parser.parse_known_args()
    class_map = dict(zip(PROTOCOLS, range(len(PROTOCOLS))))
    norm_flag = '_norm_' if args.normalize else ''
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    if args.experiment == '1': # Evaluate the models trained for a specific noise and channel condition - we took 10.0 dBs as fixed noise during training
        for channel in CHANNELS:
            # Load the three models for each channel evaluation
            model_lg = TransformerModel(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2, use_pos=False)
            model_lg.load_state_dict(torch.load(f"{TRANS_PATH}/model{channel}_10_lg.pt", map_location=device)['model_state_dict'])
            model_lg.eval()
            model_sm = TransformerModel(classes=len(PROTOCOLS), d_model=64*2, seq_len=24, nlayers=2, use_pos=False)
            model_sm.load_state_dict(torch.load(f"{TRANS_PATH}/model{channel}_10_sm.pt", map_location=device)['model_state_dict'])
            model_sm.eval()
            cnn = Baseline_CNN1D(classes=len(PROTOCOLS), numChannels=2, slice_len=512, normalize=args.normalize)
            cnn.load_state_dict(torch.load(f"{CNN_PATH}/model.cnn.{channel}.10.pt", map_location=device)['model_state_dict'])
            cnn.eval()
            if args.use_gpu:
                model_lg.cuda()
                model_sm.cuda()
                cnn.cuda()
            y_trans_lg, y_trans_sm, y_cnn = [], [], []
            for test_channel in CHANNELS:
                y_trans_lg.append(validate(model_lg, class_map, seq_len=64, sli_len=128, channel=test_channel))
                print(f'Accuracy values for channel {test_channel} and large architecture trained for {channel} and 10 dBs are: ', y_trans_lg[-1])
                y_trans_sm.append(validate(model_sm, class_map, seq_len=24, sli_len=64, channel=test_channel))
                print(f'Accuracy values for channel {test_channel} and small architecture trained for {channel} and 10 dBs are: ', y_trans_sm[-1])
                y_cnn.append(validate(cnn, class_map, seq_len=1, sli_len=512, channel=test_channel, cnn=True))
                print(f'Accuracy values for channel {test_channel} and cnn architecture trained for {channel} and 10 dBs are: ', y_cnn[-1])
        
            with open(f'test_results_10dBs_{channel}{norm_flag}.txt', 'w') as f:
                f.write(str(y_trans_lg) + '%' + str(y_trans_sm) + '%' + str(y_cnn))

    elif args.experiment == '2': # Evaluate the models trained for a specific channel condition and variable noise conditions 
        for channel in CHANNELS:
            # Load the three models for each channel evaluation
            model_lg = TransformerModel(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2, use_pos=False)
            model_lg.load_state_dict(torch.load(f"{TRANS_PATH}/model{channel}_lg.pt", map_location=device)['model_state_dict'])
            model_lg.eval()
            model_sm = TransformerModel(classes=len(PROTOCOLS), d_model=64*2, seq_len=24, nlayers=2, use_pos=False)
            model_sm.load_state_dict(torch.load(f"{TRANS_PATH}/model{channel}_sm.pt", map_location=device)['model_state_dict'])
            model_sm.eval()
            cnn = Baseline_CNN1D(classes=len(PROTOCOLS), numChannels=2, slice_len=512, normalize=args.normalize)
            cnn.load_state_dict(torch.load(f"{CNN_PATH}/model.cnn.{channel}.pt", map_location=device)['model_state_dict'])
            cnn.eval()
            if args.use_gpu:
                model_lg.cuda()
                model_sm.cuda()
                cnn.cuda()
            y_trans_lg, y_trans_sm, y_cnn = [], [], []
            for test_channel in CHANNELS:
                y_trans_lg.append(validate(model_lg, class_map, seq_len=64, sli_len=128, channel=test_channel))
                print(f'Accuracy values for channel {test_channel} and large architecture trained for {channel} are: ', y_trans_lg[-1])
                y_trans_sm.append(validate(model_sm, class_map, seq_len=24, sli_len=64, channel=test_channel))
                print(f'Accuracy values for channel {test_channel} and small architecture trained for {channel} are: ', y_trans_sm[-1])
                y_cnn.append(validate(cnn, class_map, seq_len=1, sli_len=512, channel=test_channel, cnn=True))
                print(f'Accuracy values for channel {test_channel} and cnn architecture trained for {channel} are: ', y_cnn[-1])
        
            with open(f'test_results_uniformdist_{channel}{norm_flag}.txt', 'w') as f:
                f.write(str(y_trans_lg) + '%' + str(y_trans_sm) + '%' + str(y_cnn))
    
    else: # Experiment 3: # Evaluate the models trained for general noise and channel conditions
        # Load the three models only one time
        model_lg = TransformerModel(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2, use_pos=False)
        model_lg.load_state_dict(torch.load(f"{TRANS_PATH}/modelrandom_lg.pt", map_location=device)['model_state_dict'])
        model_lg.eval()
        model_sm = TransformerModel(classes=len(PROTOCOLS), d_model=64*2, seq_len=24, nlayers=2, use_pos=False)
        model_sm.load_state_dict(torch.load(f"{TRANS_PATH}/modelrandom_sm.pt", map_location=device)['model_state_dict'])
        model_sm.eval()
        cnn = Baseline_CNN1D(classes=len(PROTOCOLS), numChannels=2, slice_len=512, normalize=args.normalize)
        cnn.load_state_dict(torch.load(f"{CNN_PATH}/model.cnn.random.pt", map_location=device)['model_state_dict'])
        cnn.eval()
        if args.use_gpu:
            model_lg.cuda()
            model_sm.cuda()
            cnn.cuda()
        y_trans_lg, y_trans_sm, y_cnn = [], [], []
        for channel in CHANNELS:
            y_trans_lg.append(validate(model_lg, class_map, seq_len=64, sli_len=128, channel=channel))
            print(f'Accuracy values for channel {channel} and large architecture are: ', y_trans_lg[-1])
            y_trans_sm.append(validate(model_sm, class_map, seq_len=24, sli_len=64, channel=channel))
            print(f'Accuracy values for channel {channel} and small architecture are: ', y_trans_sm[-1])
            y_cnn.append(validate(cnn, class_map, seq_len=1, sli_len=512, channel=channel, cnn=True))
            print(f'Accuracy values for channel {channel} and cnn architecture are: ', y_cnn[-1])
        
        with open(f'test_results_uniformdist_onemodel{norm_flag}.txt', 'w') as f:
            f.write(str(y_trans_lg) + '%' + str(y_trans_sm) + '%' + str(y_cnn))
    
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
