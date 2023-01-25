import numpy.typing as npt
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import matplotlib
import matplotlib.axes as axes
import matplotlib.axis as axis
import matplotlib.figure as figure
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import argparse
from scipy.io import loadmat
import glob
import os
import random

from matplotlib.colors import LinearSegmentedColormap

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

def plot_spectrogram(
    ax: axes.Axes,
    x: npt.ArrayLike,
    sample_rate: int,
    center_freq: int = 0,
    **kwargs):
    """
    Plot a spectrogram with additional styling (such as axes labels and colorbar).

    :param ax: Axis onto which the spectrogram will be plotted.
    :param x: Signal whose spectrogram is to be plotted.
    :param sample_rate: Sample rate in samples per unit time.
    :param center_freq: Center frequency.
    :param colorbar: Either boolean indicating whether to add a colorbar or
                     a dictionary of keyword arguments to pass to 'colorbar'.
    :param si_freqs: Whether to display frequency axis using SI prefixes.
    :param si_time: Whether to display time axis using SI prefixes.
    :param kwargs: Additional keyword arguments to be forwarded to `specgram`.
    :return: Unaltered return value of 'matplotlib.axes.Axes.specgram'.
    """

    # Default arguments repeated explicitly for ease of future configuration.
    kwargs.setdefault("NFFT", 64)
    kwargs.setdefault("window", mlab.window_hanning)  # of size nfft
    kwargs.setdefault("noverlap", 32)

    # Plot spectrogram.
    spectrum: np.ndarray
    freqs: np.ndarray
    time_axis: np.ndarray
    im: "matplotlib.image.AxesImage"
    (spectrum, freqs, time_axis, im) = ax.specgram(x, Fs=sample_rate, Fc=center_freq, cmap=parula_map, **kwargs)

    # Add colorbar if requested.

    ax.set_xticks([], [])
    ax.set_yticks([], [])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = '',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--IQ_folder_path', default='', type=str, help='')

    args = parser.parse_args()
    
    spectrogram_folder = 'images'
    save_path = os.path.join(os.path.abspath(args.IQ_folder_path),spectrogram_folder)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    #print(save_path)
    
    IQ_file_list = glob.glob(os.path.abspath(args.IQ_folder_path)+'/*')

    # sort the files
    IQ_file_list = sorted(IQ_file_list, key = os.path.getmtime)
    #IQ_file_list = IQ_file_list[-20:]

    print('********** Creating Spectrograms ***********') 
    for this_IQ_path in tqdm(IQ_file_list):
        #print(this_IQ_path)

        if this_IQ_path.endswith('.mat'):
            filename = this_IQ_path.split('/')[-1].split('.mat')[0]
            #print(filename)
            
            with open(this_IQ_path,'rb') as handle:
                #content = np.fromfile(handle, dtype=np.complex64)
                mat_content = loadmat(handle, squeeze_me=True)
                #print(mat_content['waveform'].shape)
                #print(mat_content['waveform'].dtype)
                #print(mat_content['waveform'])
            # discard the fist element
            # x = content[1:]

            x = mat_content['waveform']
            #x = np.fft.fft(x, 64)
            #x = np.fft.fftshift(x)
            #x = np.fft.ifft(x, 64)
            fig, ax = plt.subplots(figsize=(10, 10))
            plt.rcParams['figure.dpi'] = 100
            plt.rcParams['savefig.dpi'] = 100
            
            plot_spectrogram(ax, x, sample_rate=20000000, center_freq=0)

            final_save_path = save_path + '/' + filename + '.jpg'
            #print(save_path)
            fig.savefig(final_save_path)

            plt.close(fig)

