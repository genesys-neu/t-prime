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
import threading
from queue import Queue


N = 12900 # number of complex samples needed
q = Queue(2)
q2 = Queue(2)
# our decisions will also be delayed by 206 ms once the buffer is full
freq = 2.457e9  # LO tuning frequency in Hz
exitFlag = 0
fs = 31.25e6  # Radio sample Rate



# producer task
def receiver():
    rx_chan = 0  # RX1 = 0, RX2 = 1

    use_agc = True  # Use or don't use the AGC
    timeout_us = int(5e6)

    #  Initialize the AIR-T receiver using SoapyAIRT
    sdr = SoapySDR.Device(dict(driver="SoapyAIRT"))  # Create AIR-T instance
    sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)  # Set sample rate
    sdr.setGainMode(SOAPY_SDR_RX, 0, use_agc)  # Set the gain mode
    sdr.setFrequency(SOAPY_SDR_RX, 0, freq)  # Tune the LO

    # Create data buffer and start streaming samples to it
    rx_buff = np.empty(2 * N, np.int16)  # Create memory buffer for data stream
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [rx_chan])  # Setup data stream
    sdr.activateStream(rx_stream)  # this turns the radio on
    file_cntr = 0
    restart_cntr = 0

    while not exitFlag:
        sr = sdr.readStream(rx_stream, [rx_buff], N, timeoutUs=timeout_us)
        file_cntr = file_cntr + 1
        rc = sr.ret  # number of samples read or the error code
        if rc != N:
            print('Error {} after {} attempts at reading the buffer'.format(sr.ret, file_cntr))
            t1 = time.perf_counter()
            sdr.deactivateStream(rx_stream)  # turn off the stream
            sdr.activateStream(rx_stream)  # turn on the stream again
            t2 = time.perf_counter()
            print('restarting the stream took {} ms'.format(1000*(t2 - t1)))
            restart_cntr = restart_cntr + 1
        if not q.full():
            q.put(rx_buff)
            # print('Putting ' + str(rx_buff) + ' : ' + str(q.qsize()) + ' items in queue')

    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)
    print('Restarted {} times'.format(restart_cntr))


def signalprocessing():
    rx_bits = 16  # The AIR-T's ADC is 16 bits
    taps = firwin(numtaps=101, cutoff=10e6, fs=fs)

    while not exitFlag:
        if not q.empty():
            t1 = time.perf_counter()
            s_final = np.empty(16384)
            item = q.get()
            # print(str(q.qsize()) + ' items in queue')

            ############################################################################################
            # Process Signal
            ############################################################################################
            # Convert interleaved shorts (received signal) to numpy.complex64 normalized between [-1, 1]
            s0 = item.astype(float) / np.power(2.0, rx_bits - 1)
            # print('s0 {}'.format(s0.size))
            s = (s0[::2] + 1j * s0[1::2])
            # print('s {}'.format(s.size))
            t2 = time.perf_counter()
            print('reading queue and converting to complex float took {} ms'.format(1000*(t2-t1)))

            # Low-Pass Filter
            lpf_samples = np.convolve(s, taps, 'valid')
            t3 = time.perf_counter()
            print('lpf took {} ms'.format(1000*(t3-t2)))

            # rational resample
            # Resample to 20e6
            resampled_samples = resample_poly(lpf_samples, 16, 25)
            # 16*31.25=500,20*25=500(need LCM because input needs to be an int).
            # So we go up by factor of 16, then down by factor of 25 to reach final samp_rate of 20e6
            # print('resampled_samples {}, # {}'.format(resampled_samples, resampled_samples.size))
            t4 = time.perf_counter()
            print('resampling took {} ms'.format(1000*(t4-t3)))

            # convert to ML input
            s_final[::2] = resampled_samples.real
            s_final[1::2] = resampled_samples.imag
            # print('final s {}, # {}'.format(s_final, s_final.size))

            if not q2.full():
                q2.put(s_final)
                #print(str(q2.qsize()) + ' items in queue 2')
            t5 = time.perf_counter()
            print('final format converstion took {} ms'.format(1000*(t5-t4)))
            # print("signal processing took {} ms".format(1000*(t5-t1)))
        else:
            print('q is empty!')


def machinelearning():
    while not exitFlag:
        if not q2.empty():
            input = q2.get()
            #print(str(q2.qsize()) + ' items in queue 2')


if __name__ == '__main__':
    # TODO add argparsing here

    rec = threading.Thread(target=receiver)
    rec.start()

    sp = threading.Thread(target=signalprocessing)
    sp.start()

    ml = threading.Thread(target=machinelearning)
    ml.start()

    # gracefully end program
    time.sleep(60)
    exitFlag = 1

    rec.join()
    sp.join()
    ml.join()

