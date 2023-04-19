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
import Queue
import logging


logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-9s) %(message)s',)
buffer_size = 12900*1e6
q = Queue.Queue(buffer_size)


class ReceiverThread(threading.Thread):
    def __init__(self, name=None, target=None, freq=2.457e9):
        super(ProducerThread,self).__init__()
        self.target = target
        self.name = name
        self.freq = freq

    def run(self):
        N = 12900 # number of complex samples per file
        rx_chan = 0  # RX1 = 0, RX2 = 1
        fs = 31.25e6  # Radio sample Rate

        use_agc = True  # Use or don't use the AGC
        timeout_us = int(5e6)
        rx_bits = 16  # The AIR-T's ADC is 16 bits

        #  Initialize the AIR-T receiver using SoapyAIRT
        sdr = SoapySDR.Device(dict(driver="SoapyAIRT"))  # Create AIR-T instance
        sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)  # Set sample rate
        sdr.setGainMode(SOAPY_SDR_RX, 0, use_agc)  # Set the gain mode
        sdr.setFrequency(SOAPY_SDR_RX, 0, freq)  # Tune the LO

        # Create data buffer and start streaming samples to it
        rx_buff = np.empty(2 * N, np.int16)  # Create memory buffer for data stream
        rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [rx_chan])  # Setup data stream
        sdr.activateStream(rx_stream)  # this turns the radio on

        while True:
            sr = sdr.readStream(rx_stream, [rx_buff], N, timeoutUs=timeout_us)
            rc = sr.ret  # number of samples read or the error code
            if rc != N:
                logging.debug('Error {} after {} attempts at reading the buffer'.format(sr.ret, file_cntr))
                t1 = time.perf_counter()
                sdr.deactivateStream(rx_stream)  # turn off the stream
                sdr.activateStream(rx_stream)  # turn on the stream again
                t2 = time.perf_counter()
                logging.debug('restarting the stream took {} s'.format(t2 - t1))
            if not q.full():
                q.put(rx_buff)
                logging.debug('Putting ' + str(item) + ' : ' + str(q.qsize()) + ' items in queue')
        return


class MLThread(threading.Thread):
    def __init__(self, name=None, target=None):
        super(ConsumerThread,self).__init__()
        self.target = target
        self.name = name

    def run(self):
        while True:
            if not q.empty():
                item = q.get()
                logging.debug('Getting ' + str(item) + ' : ' + str(q.qsize()) + ' items in queue')
        return


if __name__ == '__main__':
    r = ReceiverThread(name='receiver')
    m = MLThread(name='ml')

    r.start()
    m.start()