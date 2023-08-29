import sys
import scipy.io as sio
import shutil
import matlab
from matutils import matutils
import os
from glob import glob
import numpy as np

DATA_PATH = '../data/DSTL_DATASET_1_1_TEST'
TEST_DATA_PATH = '../data/DSTL_DATASET_1_1_TEST'
PROTOCOLS = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
CHANNELS = ['TGn', 'TGax', 'Rayleigh']
SNR = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0] # in dBs
#TEST_RATIO = 0.05

def check_test_directory():
    if not os.path.isdir(TEST_DATA_PATH):
        #os.mkdir(TEST_DATA_PATH)
        raise Exception("No TPrime TEST DATASET found on ", TEST_DATA_PATH)


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

def apply_wchan(mat_sig, mateng, chan_models, channel):

    ch = chan_models[channel]
    proc_sig = mateng.eng.step(ch, mat_sig, nargout=1)
    return np.array(proc_sig)


if __name__ == "__main__":
    # Initialize channel models
    mateng = matutils.MatlabEngine()
    tgn = mateng.eng.wlanTGnChannel('SampleRate', float(20e6), 'DelayProfile', 'Model-B', 'LargeScaleFadingEffect', 'Pathloss', 'PathGainsOutputPort', True)
    tgax = mateng.eng.wlanTGaxChannel('SampleRate', float(20e6), 'ChannelBandwidth', 'CBW20', 'DelayProfile', 'Model-B', 'LargeScaleFadingEffect', 'Pathloss', 'PathGainsOutputPort', True)
    rayleigh = mateng.eng.comm.RayleighChannel('SampleRate', float(20e6), 'PathDelays', float(1.5e-9), 'AveragePathGains', float(-3), 'PathGainsOutputPort', True)
    channels = {'TGn': tgn, 'TGax': tgax, 'Rayleigh': rayleigh}
    assert(not ('802_11b' in PROTOCOLS))
    check_test_directory()
    # retrieve the list of signals (.mat) from every folder/protocol specified
    for i, p in enumerate(PROTOCOLS):
        path = os.path.join(DATA_PATH, p)
        if os.path.isdir(path):
            mat_list = sorted(glob(os.path.join(path, '*.mat')))
            #n_signals = int(len(mat_list) * TEST_RATIO) If there is a test portion to extract 
            #mat_list = mat_list[len(mat_list) - n_signals:] # Pick last n_signals
            for file_name in mat_list:
                mat_dict = sio.loadmat(file_name)
                # move .mat and .yaml to the test directory
                #shutil.move(file_name, os.path.join(TEST_DATA_PATH, p))
                #shutil.move(file_name[:-4] + '.yaml', os.path.join(TEST_DATA_PATH, p))
                name = file_name.split("/")[-1][:-4]
                for channel in CHANNELS:
                    #for noise in SNR:
                    chan_sig = apply_wchan(mateng.py2mat_array(mat_dict['waveform']), mateng, channels, channel) if channel != 'None' else mat_dict['waveform']
                    #noisy_sig = apply_AWGN(noise, chan_sig)
                    name_ch_noise = name + '_' + channel + '.npy'
                    if not os.path.isdir(os.path.join(TEST_DATA_PATH, p, channel)):
                        os.mkdir(os.path.join(TEST_DATA_PATH, p, channel))
                    np.save(os.path.join(TEST_DATA_PATH, p, channel, name_ch_noise), chan_sig)
                

        else:
            sys.exit('[TPrimeDataset] folder ' + path + ' not found. Aborting...')
