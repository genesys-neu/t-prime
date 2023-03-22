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
from dstl_transformer.model_transformer import TransformerModel

# CONFIG
TEST_DATA_PATH = '/home/miquelsirera/Desktop/dstl/data/DSTL_DATASET_1_1_TEST'
MODELS_PATH = '/home/miquelsirera/Desktop/dstl/dstl_transformer/model_cp'
MODELS = ["Trans. (64 x 128) [6.8M params]", "Trans. (24 x 64) [1.6M params]"]
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

def validate(model, class_map, seq_len, sli_len, channel):
    correct = np.zeros(len(SNR))
    total_samples = 0
    start_time = time.time()
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
                X = []
                # create batch of sequences
                idxs = list(range(0, len_sig-(sli_len*seq_len), (seq_len-1) + sli_len))
                for idx in idxs:
                    obs = noisy_sig[idx:idx+sli_len*(seq_len-1) + sli_len, 0]
                    obs = np.stack((obs.real, obs.imag))
                    obs = chan2sequence(obs)
                    slice_ixs = list(range(0, obs.size-sli_len*2+1, sli_len*2))
                    obs = [obs[j:j+sli_len*2] for j in slice_ixs]
                    obs = np.asarray(obs)
                    if len(X) == 0:
                        X = [obs]
                    else:
                        X = np.concatenate((X, [obs]), axis=0)
                # predict
                X = torch.from_numpy(X)
                y = np.empty(len(idxs))
                y.fill(class_map[p])
                y = torch.from_numpy(y)
                pred = model(X.float())
                # add correct ones
                correct[i] += (pred.argmax(1) == y).type(torch.float).sum().item()
                if i == 0:
                    total_samples += len(idxs)
            print("--- %s seconds ---" % (time.time() - prev_time))
            prev_time = time.time()
        print("--- %s seconds for protocol ---" % (time.time() - start_time))
    return correct/total_samples*100


if __name__ == "__main__":
    y_trans_lg, y_trans_sm = [], []
    class_map = dict(zip(PROTOCOLS, range(len(PROTOCOLS))))
    for channel in CHANNELS:
        # Load the two models
        model_lg = TransformerModel(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2, use_pos=False)
        model_lg.load_state_dict(torch.load(f"{MODELS_PATH}/model{channel}_lg.pt", map_location=torch.device('cpu'))['model_state_dict'])
        model_lg.eval()
        model_sm = TransformerModel(classes=len(PROTOCOLS), d_model=64*2, seq_len=24, nlayers=2, use_pos=False)
        model_sm.load_state_dict(torch.load(f"{MODELS_PATH}/model{channel}_sm.pt", map_location=torch.device('cpu'))['model_state_dict'])
        model_sm.eval()

        y_trans_lg.append(validate(model_lg, class_map, seq_len=64, sli_len=128, channel=channel))
        print(f'Accuracy values for channel {channel} and large architecture are: ', y_trans_lg[-1])
        y_trans_sm.append(validate(model_sm, class_map, seq_len=24, sli_len=64, channel=channel))
        print(f'Accuracy values for channel {channel} and small architecture are: ', y_trans_sm[-1])
    
    fig, ax = plt.subplots(2, 2, figsize = (12, 6))

    for i in range(2):
        for j in range(2):
            ax[i][j].plot(SNR, y_trans_lg[i*2+j], color='#000000', linestyle='solid', marker='o', label=MODELS[0])
            ax[i][j].plot(SNR, y_trans_sm[i*2+j], color='#56B4E9', linestyle='dashed', marker='v', label=MODELS[1])
            #ax[i][j].plot(x_cnn[i*2+j], y_cnn[i*2+j], color='#D55E00', linestyle='dashdot', marker='^', label=models[2])
            ax[i][j].set_title(CHANNELS[i*2+j])
            ax[i][j].set_xlabel('SNR (dBs)')
            ax[i][j].set_ylabel('Accuracy (%)')
            ax[i][j].set_ylim(15,105)
            ax[i][j].grid()     
    plt.suptitle('Results comparison between different architectures')
    fig.legend(MODELS, bbox_to_anchor=(0.87, 0.02), ncols=3, labelspacing=1)
    plt.tight_layout() 
    plt.savefig('TESTING.png')