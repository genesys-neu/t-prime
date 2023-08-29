import numpy as np
import torch
from dstl.preprocessing.TPrime_dataset import TPrimeDataset
from model_cnn1d import Baseline_CNN1D
import os


Nclass = 4
num_channels = 2
num_feats = 1
slice_len = 128
snr_dBs = [30]

PATH = './results/SNR30/'
model_file_name = 'model.best.pt'
model_path = os.path.join(PATH, model_file_name)


protocols = ['802_11ax', '802_11b', '802_11n', '802_11g']
ds_test = TPrimeDataset(protocols, ds_type='test', snr_dbs=snr_dBs, slice_len=slice_len, slice_overlap_ratio=0.5, override_gen_map=True)

model = Baseline_CNN1D(classes=Nclass, numChannels=num_channels, slice_len=slice_len)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(model.device.type) # reload the model on the appropriate device
model.eval()    # set the evaluation mode


from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt

def find_label(class_map, true_label):

    for k, v in class_map.items():
        if v == true_label:
            label_txt = k
            break
    return label_txt

plot_dir = os.path.join(PATH, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

for _ in range(30):
    # obtain a random sample from the test dataset. This will be needed in order to populate the cache initially
    rand_sixs = np.random.choice(len(ds_test))
    _, true_label = ds_test[rand_sixs]
    # now let's retrieve the whole signal from the signal cache
    signal_path = ds_test.last_file_loaded
    # let's obtain the whole signal and run inference on each slice
    sig = ds_test.signal_cache.cache[signal_path]
    # apply AWGN
    noisy_sig = ds_test.apply_AWGN(sig)
    len_sig = noisy_sig.shape[0]
    window_ixs = list(range(0, len_sig-ds_test.slice_len, ds_test.overlap))

    fs = 20e6 if true_label != 1 else 11e6
    f, t, Sxx = signal.spectrogram(np.squeeze(noisy_sig), fs)
    plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0)) # cmap='afmhot'
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    #plt.show()

    plt.savefig(os.path.join(plot_dir, str(rand_sixs)+'ex_' + find_label(ds_test.ds_info['class_map'], true_label) + '_spectrogram.png'))

    plt.clf()

    # Create two subplots and unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(noisy_sig.real)
    ax1.plot(noisy_sig.imag)
    ax1.set_title('Signal type: '+ find_label(ds_test.ds_info['class_map'], true_label))
    ax2.plot(noisy_sig.real)
    ax2.plot(noisy_sig.imag)



    color_class = ['red', 'green', 'blue', 'yellow']

    for w in window_ixs:
        complex_slice = noisy_sig[w : w + ds_test.slice_len, 0]
        slice = np.stack((complex_slice.real, complex_slice.imag))
        slice = slice[np.newaxis, :, :]
        slice_t = torch.Tensor(slice)
        slice_t = slice_t.to(model.device.type)
        out = model(slice_t.float())
        _, predicted = torch.max(out, 1)
        ax2.axvspan(w, w + ds_test.slice_len, color=color_class[predicted], alpha=0.25)

    #plt.show()
    plt.savefig(os.path.join(plot_dir, str(rand_sixs)+'ex_' + find_label(ds_test.ds_info['class_map'], true_label) + '_wavetag.png'))
    plt.clf()
