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
from TPrime_transformer.model_transformer import TransformerModel
from baseline_models.model_cnn1d import Baseline_CNN1D
from baseline_models.model_AMCNet import AMC_Net
from baseline_models.model_ResNet import ResNet
from baseline_models.model_MCFormer import MCformer
from baseline_models.model_LSTM import LSTM_ap


supported_outmode = ['real', 'complex', 'real_invdim', 'real_ampphase'] # has to be same as in TPrime_torch_train

# CONFIG
TRANS_PATH = '../TPrime_transformer/model_cp'
CNN_PATH = '../baseline_models/results_slice512'
MODELS = ["Trans. (64 x 128) [6.8M params]", "Trans. (24 x 64) [1.6M params]", "CNN (1 x 512) [4.1M params]"]

RESNET_PATH = '../baseline_models/results_ResNet_norm'
AMCNET_PATH = '../baseline_models/results_AMCNet'
MCFORMER_PATH = '../baseline_models/results_MCformer_largekernel'
#LSTM_PATH = '../baseline_models/results_LSTM/'  # this is just random output
MODELS += ["ResNet [13] (1 x 1024) [162K params]", "AMCNet [16] (1 x 128) [462K params]", "MCFormer [8] (1 x 128) [78K params]"]

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

def validate(model, class_map, seq_len, sli_len, channel, cnn=False, out_mode='real'):

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
                    assert (out_mode in supported_outmode)
                    if out_mode == 'real_invdim':
                        obs = np.swapaxes(obs, 1, 0)  # shape = [N, 2]
                        # generate idxs for split
                        idxs = list(range(sli_len, len_sig, sli_len))
                        obs = np.split(obs, idxs, axis=0)[:-1]
                    else:
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
        print("--------", p, "--------")
        print("--- %s seconds for protocol ---" % (time.time() - prev_time))
        prev_time = time.time()
    return correct/total_samples*100

def generate_dummy_input(channel, seq_len, sli_len):
    p = '802_11ax'
    dBs = 10
    path = os.path.join(TEST_DATA_PATH, p) if channel == 'None' else os.path.join(TEST_DATA_PATH, p, channel)
    mat_list = sorted(glob(os.path.join(path, '*.mat'))) if channel == 'None' else sorted(glob(os.path.join(path, '*.npy')))
    signal_path = mat_list[0]
    sig = sio.loadmat(signal_path) if channel == 'None' else np.load(signal_path)
    if channel == 'None':
        sig = sig['waveform']
    len_sig = sig.shape[0]
    noisy_sig = apply_AWGN(dBs, sig)
    len_sig = noisy_sig.shape[0]
    # stack real and imag parts
    obs = np.stack((noisy_sig.real, noisy_sig.imag))
    obs = np.squeeze(obs, axis=2)
    # zip the I and Q values
    if seq_len != 1: # Transformer architecture
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
    # keep just one
    X = X[0,:,:]
    # create torch tensor
    X = torch.from_numpy(X)
    X = torch.unsqueeze(X, 0)
    print(X.shape)
    return X

def timing_inference_GPU(device, channel, seq_len, sli_len, model):
    # PREPARE INPUT
    dummy_input = generate_dummy_input(channel, seq_len, sli_len)
    dummy_input = dummy_input.to(device)
    model = model.double()
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(100):
        _ = model(dummy_input.double())
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

def calculate_avg_time(means, sds):
    # Calculate weights as inverse squared standard deviations
    weights = 1 / sds ** 2

    # Calculate weighted average (mean)
    weighted_average = np.average(means, weights=weights)

    # Calculate uncertainty-weighted root mean square deviation (standard deviation)
    weighted_std = np.sqrt(np.average((means - weighted_average) ** 2, weights=weights))

    return weighted_average, weighted_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default='3', choices=['1', '2', '3', '4'], help="Decide which models to test, 1 is for models trained for \
                         specific noise and channel conditions, 2 is for models specifically trained for a channel, 3 is for single model for all channel and noise conditions (with SoTA models comparison) and 4 is for inference time analysis")
    parser.add_argument("--normalize", action='store_true', default=False, help="Use a layer norm as a first layer for CNN")
    parser.add_argument("--use_gpu", action='store_true', default=False, help="Use gpu for inference")
    parser.add_argument("--test_path", default='../data/DATASET1_1_TEST', help="Path to the dataset that will be used for testing. DATASET1_1_TEST contains the necessary data to test these models.")
    args, _ = parser.parse_known_args()
    TEST_DATA_PATH = args.test_path
    class_map = dict(zip(PROTOCOLS, range(len(PROTOCOLS))))
    norm_flag = '.norm' if args.normalize else ''
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
            cnn.load_state_dict(torch.load(f"{CNN_PATH}/model.cnn.{channel}{norm_flag}.10.pt", map_location=device)['model_state_dict'])
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
            cnn.load_state_dict(torch.load(f"{CNN_PATH}/model.cnn.{channel}{norm_flag}.range.pt", map_location=device)['model_state_dict'])
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
    
    elif args.experiment == '3': # Evaluate the models trained for general noise and channel conditions
        # Load the three models only one time
        models_loaded = [False] * 6
        print("The models that have not been loaded will appear as empty lists in the results.")
        try:
            model_lg = TransformerModel(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2, use_pos=False)
            model_lg.load_state_dict(torch.load(f"{TRANS_PATH}/modelrandom_lg.pt", map_location=device)['model_state_dict'])
            model_lg.eval()
            if args.use_gpu:
                model_lg.cuda()
            models_loaded[0] = True
        except: 
            print(f"LG model not found, the model name should be modelrandom_lg.pt and be placed at {TRANS_PATH}.")
        try:
            model_sm = TransformerModel(classes=len(PROTOCOLS), d_model=64*2, seq_len=24, nlayers=2, use_pos=False)
            model_sm.load_state_dict(torch.load(f"{TRANS_PATH}/modelrandom_sm.pt", map_location=device)['model_state_dict'])
            model_sm.eval()
            if args.use_gpu:
                model_sm.cuda()
            models_loaded[1] = True
        except: 
            print(f"SM model not found, the model name should be modelrandom_sm.pt and be placed at {TRANS_PATH}.")
        # CNN Baseline
        cnn_slicelen = 512
        try:
            cnn = Baseline_CNN1D(classes=len(PROTOCOLS), numChannels=2, slice_len=cnn_slicelen, normalize=args.normalize)
            cnn.load_state_dict(torch.load(f"{CNN_PATH}/model.cnn.random{norm_flag}.range.pt", map_location=device)['model_state_dict'])
            cnn.eval()
            if args.use_gpu:
                cnn.cuda()
            models_loaded[2] = True
        except: 
            print(f"CNN model not found, the model name should be model.cnn.random{norm_flag}.range.pt and be placed at {CNN_PATH}.")
        # ResNet
        resnet_slicelen = 1024
        try:
            resnet = ResNet(num_classes=len(PROTOCOLS), num_samples=resnet_slicelen, iq_dim=2, kernel_size=3, pool_size=2)
            resnet.load_state_dict(torch.load(f"{RESNET_PATH}/model.ResNet.random.range.pt", map_location=device)['model_state_dict'])
            resnet.eval()
            if args.use_gpu:
                resnet.cuda()
            models_loaded[3] = True
        except: 
            print(f"ResNet model not found, the model name should be model.ResNet.random.range.pt and be placed at {RESNET_PATH}.")
            
        # AMCNet
        amcnet_slicelen = 128
        try:
            amcnet = AMC_Net(num_classes=len(PROTOCOLS))
            amcnet.load_state_dict(torch.load(f"{AMCNET_PATH}/model.AMCNet.random.range.pt", map_location=device)['model_state_dict'])
            amcnet.eval()
            if args.use_gpu:
                amcnet.cuda()
            models_loaded[4] = True
        except: 
            print(f"AMCNet model not found, the model name should be model.AMCNet.random.range.pt and be placed at {AMCNET_PATH}.")

        # MCformer
        mcformer_slicelen = 128
        try:
            mcformer = MCformer()
            mcformer.load_state_dict(torch.load(f"{MCFORMER_PATH}/model.MCformer.random.range.pt", map_location=device)['model_state_dict'])
            mcformer.eval()
            if args.use_gpu:
                mcformer.cuda()
            models_loaded[5] = True
        except: 
            print(f"MCformer model not found, the model name should be model.MCformer.random.range.pt and be placed at {MCFORMER_PATH}.")

        y_trans_lg, y_trans_sm, y_cnn = [], [], []
        y_resnet, y_amcnet, y_mcformer = [], [], []
        for channel in CHANNELS:
            if models_loaded[0]:
                y_trans_lg.append(validate(model_lg, class_map, seq_len=64, sli_len=128, channel=channel))
                print(f'Accuracy values for channel {channel} and large architecture are: ', y_trans_lg[-1])
            if models_loaded[1]:
                y_trans_sm.append(validate(model_sm, class_map, seq_len=24, sli_len=64, channel=channel))
                print(f'Accuracy values for channel {channel} and small architecture are: ', y_trans_sm[-1])
            if models_loaded[2]:
                y_cnn.append(validate(cnn, class_map, seq_len=1, sli_len=cnn_slicelen, channel=channel, cnn=True))
                print(f'Accuracy values for channel {channel} and cnn architecture are: ', y_cnn[-1])
            if models_loaded[3]:
                y_resnet.append(validate(resnet, class_map, seq_len=1, sli_len=resnet_slicelen, channel=channel, cnn=True, out_mode='real_invdim'))
                print(f'Accuracy values for channel {channel} and ResNet architecture are: ', y_resnet[-1])
            if models_loaded[4]:
                y_amcnet.append(validate(amcnet, class_map, seq_len=1, sli_len=amcnet_slicelen, channel=channel, cnn=True))
                print(f'Accuracy values for channel {channel} and AMCNet architecture are: ', y_amcnet[-1])
            if models_loaded[5]:
                y_mcformer.append(validate(mcformer, class_map, seq_len=1, sli_len=mcformer_slicelen, channel=channel, cnn=True))
                print(f'Accuracy values for channel {channel} and MCformer architecture are: ', y_mcformer[-1])

        
        with open(f'test_results_uniformdist_onemodel{norm_flag}.txt', 'w') as f:
            f.write(str(y_trans_lg) + '%' + str(y_trans_sm) + '%' + str(y_cnn) + '%' + str(y_resnet) + '%' + str(y_amcnet) + '%' + str(y_mcformer))
    
    else: # Experiment 4: # Inference time analysis for each of the model architectures
        print('Using protocol 802.11ax and 10 dBs as a sample input.')
        # Load the three models only one time
        models_loaded = [False] * 3
        print("The models that have not been loaded will not appear in the results.")
        try:
            model_lg = TransformerModel(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2, use_pos=False)
            model_lg.load_state_dict(torch.load(f"{TRANS_PATH}/modelrandom_lg.pt", map_location=device)['model_state_dict'])
            model_lg.eval()
            if args.use_gpu:
                model_lg.cuda()
            models_loaded[0] = True
        except: 
            print(f"LG model not found, the model name should be modelrandom_lg.pt and be placed at {TRANS_PATH}.")
        try:
            model_sm = TransformerModel(classes=len(PROTOCOLS), d_model=64*2, seq_len=24, nlayers=2, use_pos=False)
            model_sm.load_state_dict(torch.load(f"{TRANS_PATH}/modelrandom_sm.pt", map_location=device)['model_state_dict'])
            model_sm.eval()
            if args.use_gpu:
                model_sm.cuda()
            models_loaded[1] = True
        except: 
            print(f"SM model not found, the model name should be modelrandom_sm.pt and be placed at {TRANS_PATH}.")
        # CNN Baseline
        cnn_slicelen = 512
        try:
            cnn = Baseline_CNN1D(classes=len(PROTOCOLS), numChannels=2, slice_len=cnn_slicelen, normalize=args.normalize)
            cnn.load_state_dict(torch.load(f"{CNN_PATH}/model.cnn.random{norm_flag}.range.pt", map_location=device)['model_state_dict'])
            cnn.eval()
            if args.use_gpu:
                cnn.cuda()
            models_loaded[2] = True
        except: 
            print(f"CNN model not found, the model name should be model.cnn.random{norm_flag}.range.pt and be placed at {CNN_PATH}.")
        y_trans_lg_time, y_trans_sm_time, y_cnn_time = [], [], []
        y_trans_lg_sd, y_trans_sm_sd, y_cnn_sd = [], [], []
        for channel in CHANNELS:
            if models_loaded[0]:
                lg_ch_time, lg_ch_sd = timing_inference_GPU(device, channel=channel, seq_len=64, sli_len=128, model=model_lg)
                y_trans_lg_time.append(lg_ch_time)
                y_trans_lg_sd.append(lg_ch_sd)
                print(f'Inference time mean and sd for channel {channel} and large architecture are: ', lg_ch_time, ' +- ', lg_ch_sd)
            if models_loaded[1]:
                sm_ch_time, sm_ch_sd = timing_inference_GPU(device, channel=channel, seq_len=24, sli_len=64, model=model_sm)
                y_trans_sm_time.append(sm_ch_time)
                y_trans_sm_sd.append(sm_ch_sd)
                print(f'Inference time mean and sd for channel {channel} and small architecture are: ', sm_ch_time, ' +- ', sm_ch_sd)
            if models_loaded[2]:
                cnn_ch_time, cnn_ch_sd = timing_inference_GPU(device, channel=channel, seq_len=1, sli_len=512, model=cnn)
                y_cnn_time.append(cnn_ch_time)
                y_cnn_sd.append(cnn_ch_sd)
                print(f'Inference time mean and sd for channel {channel} and cnn architecture are: ', cnn_ch_time, ' +- ', cnn_ch_sd)
        print('---------------------------------------------')
        # Calculate total mean and sd
        if models_loaded[0]:
            mean, sd = calculate_avg_time(np.array(y_trans_lg_time), np.array(y_trans_lg_sd))
            print(f'Average inference time mean and sd for large architecture are: ', mean, ' +- ', sd)
        if models_loaded[1]:
            mean, sd = calculate_avg_time(np.array(y_trans_sm_time), np.array(y_trans_sm_sd))
            print(f'Average inference time mean and sd for small architecture are: ', mean, ' +- ', sd)
        if models_loaded[2]:
            mean, sd = calculate_avg_time(np.array(y_cnn_time), np.array(y_cnn_sd))
            print(f'Average inference time mean and sd for cnn architecture are: ', mean, ' +- ', sd)