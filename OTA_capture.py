#!/usr/bin/env python3

# Import Packages
import numpy as np
import os
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
from scipy.signal import resample_poly, firwin, bilinear, lfilter
import matplotlib.pyplot as plt
import time
import argparse

########################################################################################
# Settings
########################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-nf", "--nfiles", help="number of files to capture", type=int)
parser.add_argument('-fq', '--frequency', help='center frequency', type=float)
parser.add_argument('-p', '--plot', help='to plot spectrogram of last file captured')
args = parser.parse_args()

# Determine how much data to record
nfiles = 1  # Number of files to record, each is approximately 20ms long
if args.nfiles:
    nfiles = args.nfiles

N = 16384 * 38  # Number of complex samples per file - approximately 20ms

rx_chan = 0  # RX1 = 0, RX2 = 1
fs = 31.25e6  # Radio sample Rate
freq = 2.457e9  # LO tuning frequency in Hz
if args.frequency:
    freq = args.frequency

use_agc = True  # Use or don't use the AGC
timeout_us = int(5e6)
rx_bits = 16            # The AIR-T's ADC is 16 bits


# Recording Settings
rec_dir = '/home/deepwave/Research/DSTL/OTA_dataset'  # Location of drive for recording


########################################################################################
# Receive Signal
########################################################################################

#  Initialize the AIR-T receiver using SoapyAIRT
sdr = SoapySDR.Device(dict(driver="SoapyAIRT")) # Create AIR-T instance
sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)          # Set sample rate
sdr.setGainMode(SOAPY_SDR_RX, 0, use_agc)       # Set the gain mode
sdr.setFrequency(SOAPY_SDR_RX, 0, freq)         # Tune the LO

# Create data buffer and start streaming samples to it
rx_buff = np.empty(2 * N, np.int16)  # Create memory buffer for data stream
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [rx_chan])  # Setup data stream
sdr.activateStream(rx_stream)  # this turns the radio on

file_cntr = 0
while file_cntr < nfiles:

    # Read the samples from the data buffer
    sr = sdr.readStream(rx_stream, [rx_buff], N, timeoutUs=timeout_us)
    rc = sr.ret  # number of samples read or the error code
    assert rc == N, 'Error {}'.format(sr.ret)


    ############################################################################################
    # Process Signal
    ############################################################################################
    # Convert interleaved shorts (received signal) to numpy.complex64 normalized between [-1, 1]
    s0 = rx_buff.astype(float) / np.power(2.0, rx_bits-1)
    # print(s0.size)
    s = (s0[::2] + 1j*s0[1::2])
    # print(s.size)

    # Low-Pass Filter
    taps = firwin(numtaps=101, cutoff=10e6, fs=fs)
    lpf_samples = np.convolve(s, taps, 'valid')

    # rational resample
    # Resample to 20e6
    resampled_samples = resample_poly(lpf_samples, 16, 25)
    # 16*31.25=500,20*25=500(need LCM because input needs to be an int).
    # So we go up by factor of 16, then down by factor of 25 to reach final samp_rate of 20e6
    # print(resampled_samples.size)
    N_plot = resampled_samples.size

    ############################################################################################
    # Save Signal
    ############################################################################################

    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_prefix = 'OTA' + timestr  # File prefix for each file
    file_name = os.path.join(rec_dir, '{}_{}.bin'.format(file_prefix, file_cntr))

    resampled_samples.tofile(file_name)
    file_cntr += 1

# Stop streaming
sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)

##############################################################################
# To Plot
##############################################################################

if args.plot:
    plt.title('Spectrogram')
    plt.specgram(resampled_samples, Fs=20e6)
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    plt.show()
