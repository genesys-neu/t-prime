#!/usr/bin/env python3

# Import Packages
import numpy as np
import torch
import os
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
from scipy.signal import resample_poly, firwin, bilinear, lfilter
import matplotlib.pyplot as plt
import time
import argparse
import threading
from dstl_transformer.model_transformer import TransformerModel
from queue import Queue


N = 12900 # number of complex samples needed
q = Queue(2)
q2 = Queue(2)
# our decisions will also be delayed by 206 ms once the buffer is full
freq = 2.457e9  # LO tuning frequency in Hz
exitFlag = 0
fs = 31.25e6  # Radio sample Rate
t_out = 60



# producer task
def receiver():
    rx_chan = 0  # RX1 = 0, RX2 = 1

    use_agc = True  # Use or don't use the AGC
    timeout_us = int(5e6)
    time_avg = 0

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
        t1 = time.perf_counter()
        sr = sdr.readStream(rx_stream, [rx_buff], N, timeoutUs=timeout_us)
        file_cntr = file_cntr + 1
        rc = sr.ret  # number of samples read or the error code
        if rc != N:
            # print('Error {} after {} attempts at reading the buffer'.format(sr.ret, file_cntr))
            sdr.deactivateStream(rx_stream)  # turn off the stream
            sdr.activateStream(rx_stream)  # turn on the stream again
            # print('restarting the stream took {} ms'.format(1000*(t2 - t1)))
            restart_cntr = restart_cntr + 1
        if not q.full():
            q.put(rx_buff)
            # print('Putting ' + str(rx_buff) + ' : ' + str(q.qsize()) + ' items in queue')
        t2 = time.perf_counter()
        time_avg = time_avg + (t2-t1)
        # print('Reciver took {} ms'.format(1000*(t2-t1)))

    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)
    
    time.sleep(1)
    print('Restarted {} times'.format(restart_cntr))
    print('Reciever takes {} ms on average to complete {} cycles'.format(1000*time_avg/file_cntr,file_cntr))


def signalprocessing():
    rx_bits = 16  # The AIR-T's ADC is 16 bits
    taps = firwin(numtaps=101, cutoff=10e6, fs=fs)
    time_avg = 0
    cntr = 0

    while not exitFlag:
        if not q.empty():
            t1 = time.perf_counter()
            s_final = np.empty(16384)
            if MODEL_SIZE == 'sm':
                s_final = np.empty(3072)

            item = q.get()
            # print(str(q.qsize()) + ' items in queue')

            ############################################################################################
            # Process Signal
            ############################################################################################
            # Convert interleaved shorts (received signal) to numpy.complex64 normalized between [-1, 1]
            s0 = item.astype(float) / np.power(2.0, rx_bits - 1)
            # print('s0 {}'.format(s0.size))
            s = s0[::2] + 1j * s0[1::2]
            # print('s {}'.format(s.size))
            if MODEL_SIZE == 'sm':
                s = s[:2500:]

            t2 = time.perf_counter()
            # print('reading queue and converting to complex float took {} ms'.format(1000*(t2-t1)))

            # Low-Pass Filter
            lpf_samples = np.convolve(s, taps, 'valid')
            t3 = time.perf_counter()
            # print('lpf took {} ms'.format(1000*(t3-t2)))

            # rational resample
            # Resample to 20e6
            resampled_samples = resample_poly(lpf_samples, 16, 25)
            # 16*31.25=500,20*25=500(need LCM because input needs to be an int).
            # So we go up by factor of 16, then down by factor of 25 to reach final samp_rate of 20e6
            # print('resampled_samples {}, # {}'.format(resampled_samples, resampled_samples.size))
            t4 = time.perf_counter()
            # print('resampling took {} ms'.format(1000*(t4-t3)))

            # convert to ML input
            s_final[::2] = resampled_samples.real
            s_final[1::2] = resampled_samples.imag
            # print('final s {}, # {}'.format(s_final, s_final.size))

            if not q2.full():
                q2.put(s_final)
                #print(str(q2.qsize()) + ' items in queue 2')
            t5 = time.perf_counter()
            # print('final format converstion took {} ms'.format(1000*(t5-t4)))
            # print("signal processing took {} ms".format(1000*(t5-t1)))
            time_avg = time_avg + (t5-t1)
            cntr = cntr + 1
        # else:
            # print('q is empty!')
    
    time.sleep(1)
    print('Signal processor takes {} ms on average to complete {} cycles'.format(1000*time_avg/cntr,cntr))

def machinelearning():
    # Model configuration and loading
    PROTOCOLS = ['802_11ax', '802_11b', '802_11n', '802_11g']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('Device is {}'.format(device))

    if MODEL_SIZE == 'sm':
        seq_len = 24
        model = TransformerModel(classes=len(PROTOCOLS), d_model=64*2, seq_len=seq_len, nlayers=2, use_pos=False)
    else: # lg architecture
        seq_len = 64
        model = TransformerModel(classes=len(PROTOCOLS), d_model=128*2, seq_len=seq_len, nlayers=2, use_pos=False)
    try:
        model.load_state_dict(torch.load(MODEL_PATH,map_location=device)['model_state_dict'])
    except:
        raise Exception("The model you provided does not correspond with the selected architecture. Please revise the path and try again.")
    # model.eval()
    model = model.float()
    model.to(device)
    model.eval()

    preds = [] # list to keep track of model predictions
    pred_cntr = 0
    time_avg = 0
    while not exitFlag:
        if not q2.empty():
            t1 = time.perf_counter()
            input = q2.get()
            # print('ML input recieved')
            # split sequence into words
            input = np.split(input, seq_len)
            # print('words are now {}'.format(input))
            input = np.array(input)
            input = torch.from_numpy(input)
            # create empty batch dimension
            input = torch.unsqueeze(input, 0) 
            input = input.to(device)
            # predict class
            pred = model(input.float()).argmax(1)
            print(PROTOCOLS[pred])
            preds.append(pred) # This will need to be sent to GUI 
            #print(str(q2.qsize()) + ' items in queue 2')
            t2 = time.perf_counter()
            pred_cntr = pred_cntr + 1
            time_avg = time_avg + (t2-t1)
    
    time.sleep(1)
    print("ML predictions takes {} ms on average to complete {} cycles".format(1000*time_avg/pred_cntr,pred_cntr)) 


# TODO add GUI interface - will this require another threadsafe queue?
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fq', '--frequency', help='center frequency, default is 2.457e9', type=float)
    parser.add_argument('-t', '--timeout', help='amount of time (in seconds) to run before graceful exit, '
                                                'default is 60s', type=int)
    parser.add_argument("--model_path", default='./', help='Path to the checkpoint to load the model for inference.')
    parser.add_argument("--model_size", default="lg", choices=["sm", "lg"], help="Define the use of the large or the small transformer.")
    args = parser.parse_args()

    if args.frequency:
        freq = args.frequency

    if args.timeout:
        t_out = args.timeout

    MODEL_PATH = args.model_path
    MODEL_SIZE = args.model_size
    
    # if MODEL_SIZE == 'sm':
        # N = 2500

    rec = threading.Thread(target=receiver)
    rec.start()

    sp = threading.Thread(target=signalprocessing)
    sp.start()

    ml = threading.Thread(target=machinelearning)
    ml.start()

    # gracefully end program
    time.sleep(t_out)
    exitFlag = 1
    time.sleep(1)

    rec.join()
    sp.join()
    ml.join()
