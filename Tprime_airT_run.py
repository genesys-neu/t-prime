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
from TPrime_transformer.model_transformer import TransformerModel
from queue import Queue
from preprocessing.model_rmsnorm import RMSNorm
import tensorrt as trt
from preprocessing.inference.onnx2plan import onnx2plan
from preprocessing.inference.plan_bench import plan_bench
import preprocessing.inference.trt_utils as trt_utils


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('Device is {}'.format(device))
    # RMS layer
    if RMSNORM:
        RMSNorm_layer = RMSNorm(model='Transformer')
    else:
        RMSNorm_layer = None

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
        with torch.no_grad():
            if not q2.empty():
                t1 = time.perf_counter()
                input = q2.get()
                # print('ML input recieved')
                # split sequence into words
                input = np.split(input, seq_len)
                # print('words are now {}'.format(input))
                input = np.array(input)
                input = torch.from_numpy(input)
                # create empty batch dimension
                input = torch.unsqueeze(input, 0) 
                input = input.to(device)
                # predict class
                if RMSNorm_layer is not None:
                    input = RMSNorm_layer(input)
                pred = model(input.float()).argmax(1)
                print(PROTOCOLS[pred])
                # Write it in output file to pass it to the GUI
                file_flag = 'a'
                # Every 500 predictions flush output content 
                if pred_cntr%500 == 0:
                    file_flag = 'w'
                with open('output.txt', file_flag) as file:
                    file.write(f'{pred.item()} {time.time()}\n') 
                preds.append(pred)
                #print(str(q2.qsize()) + ' items in queue 2')
                t2 = time.perf_counter()
                pred_cntr = pred_cntr + 1
                time_avg = time_avg + (t2-t1)
    
    time.sleep(1)
    print("ML predictions takes {} ms on average to complete {} cycles".format(1000*time_avg/pred_cntr,pred_cntr))


def machinelearning_tensorRT():
    # Model configuration and loading

    batch_size = 1
    INPUT_NODE_NAME = 'input_buffer'  # (for TensorRT) User defined name of input node
    OUTPUT_NODE_NAME = 'output_buffer'  # User defined name of output node
    ONNX_VERSION = 10  # the ONNX version to export the model to
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Nclasses = len(PROTOCOLS)
    # Setup the pyCUDA context
    trt_utils.make_cuda_context()
    # print('Device is {}'.format(device))
    # RMS layer
    if RMSNORM:
        RMSNorm_layer = RMSNorm(model='Transformer')
    else:
        RMSNorm_layer = None

    fname = os.path.basename(MODEL_PATH)
    if fname[-3:] == '.pt':
        fname = fname[:-3]
    ONNX_FILE_NAME = os.path.join(os.path.dirname(MODEL_PATH), str(fname) + ".onnx")

    if MODEL_SIZE == 'sm':
        seq_len = 24
        sli_len = 64 * 2
        model = TransformerModel(classes=Nclasses, d_model=64 * 2, seq_len=seq_len, nlayers=2, use_pos=False)
    else:  # lg architecture

        seq_len = 64
        sli_len = 128 * 2
        model = TransformerModel(classes=Nclasses, d_model=128 * 2, seq_len=seq_len, nlayers=2, use_pos=False)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
    except:
        raise Exception(
            "The model you provided does not correspond with the selected architecture. Please revise the path and try again.")
    # model.eval()

    slice_in = np.random.random((batch_size, seq_len, sli_len))
    model = model.float()
    model.to(device)
    model.eval()

    plan_file = generate_model_plan(INPUT_NODE_NAME, ONNX_FILE_NAME, ONNX_VERSION, OUTPUT_NODE_NAME, model, slice_in, benchmark=False)
    input_dtype = np.float32
    # Use pyCUDA to create a shared memory buffer that will receive samples from the
    # AIR-T to be fed into the neural network.
    batch_size, seq_len, cplx_samples = slice_in.shape
    buff_len = seq_len * cplx_samples * batch_size
    sample_buffer = trt_utils.MappedBuffer(buff_len, input_dtype)

    # Set up the inference engine. Note that the output buffers are created for
    # us when we create the inference object.
    dnn = trt_utils.TrtInferFromPlan(plan_file, batch_size,
                                     sample_buffer, verbose=False)

    preds = []  # list to keep track of model predictions
    pred_cntr = 0
    time_avg = 0
    while not exitFlag:
        with torch.no_grad():
            if not q2.empty():
                t1 = time.perf_counter()
                input = q2.get()
                # print('ML input recieved')
                # split sequence into words
                input = np.split(input, seq_len)
                # print('words are now {}'.format(input))
                input = np.array(input)
                input = torch.from_numpy(input)
                #  create empty batch dimension
                input = torch.unsqueeze(input, 0)
                input = input.to(device)
                # predict class
                if RMSNorm_layer is not None:
                    input = RMSNorm_layer(input)    # NOTE: this should also be included in the .plan
                X = input.cpu().numpy()
                dnn.input_buff.host[:] = X.flatten().astype(input_dtype)
                dnn.feed_forward()
                trt_out = dnn.output_buff.host.reshape((batch_size, Nclasses))
                pred = trt_out.argmax(1)

                #pred = model(input.float()).argmax(1)
                print(PROTOCOLS[np.squeeze(pred)])

                # Write it in output file to pass it to the GUI
                file_flag = 'a'
                # Every 500 predictions flush output content
                if pred_cntr % 500 == 0:
                    file_flag = 'w'
                with open('output.txt', file_flag) as file:
                    file.write(f'{pred.item()} {time.time()}\n')
                preds.append(pred)
                # print(str(q2.qsize()) + ' items in queue 2')
                t2 = time.perf_counter()
                pred_cntr = pred_cntr + 1
                time_avg = time_avg + (t2 - t1)

    time.sleep(1)
    print("ML predictions takes {} ms on average to complete {} cycles".format(1000 * time_avg / pred_cntr, pred_cntr))


def generate_model_plan(INPUT_NODE_NAME, ONNX_FILE_NAME, ONNX_VERSION, OUTPUT_NODE_NAME, model, slice_in, benchmark=False):
    slice_t = torch.Tensor(slice_in)
    slice_t = slice_t.to(model.device.type)
    # Let's produce the ONNX schema of the current model
    torch.onnx.export(model, slice_t, ONNX_FILE_NAME, export_params=True,
                      opset_version=ONNX_VERSION, do_constant_folding=True,
                      input_names=[INPUT_NODE_NAME], output_names=[OUTPUT_NODE_NAME],
                      dynamic_axes={INPUT_NODE_NAME: {0: 'batch_size'},
                                    OUTPUT_NODE_NAME: {0: 'batch_size'}})
    # Then create the relative plan file using TensorRT
    MAX_WORKSPACE_SIZE = 1073741824  # 1 GB for example
    onnx2plan(onnx_file_name=ONNX_FILE_NAME, nchan=slice_in.shape[1],
              input_len=slice_in.shape[2], logger=trt.Logger(trt.Logger.WARNING),
              MAX_BATCH_SIZE=slice_in.shape[0], MAX_WORKSPACE_SIZE=MAX_WORKSPACE_SIZE,
              BENCHMARK=True)

    plan_file = ONNX_FILE_NAME.replace('.onnx', '.plan')
    if benchmark:
        print('Running Inference Benchmark')
        plan_bench(plan_file_name=plan_file, cplx_samples=slice_in.shape[2], num_chans=slice_in.shape[1],
                   batch_size=slice_in.shape[0], num_batches=512, input_dtype=np.float32)
    return plan_file


# TODO add GUI interface - will this require another threadsafe queue?
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fq', '--frequency', help='center frequency, default is 2.457e9', type=float)
    parser.add_argument('-t', '--timeout', help='amount of time (in seconds) to run before graceful exit, '
                                                'default is 60s', type=int)
    parser.add_argument("--model_path", default='./', help='Path to the checkpoint to load the model for inference.')
    parser.add_argument("--model_size", default="lg", choices=["sm", "lg"], help="Define the use of the large or the small transformer.")
    parser.add_argument("--RMSNorm", default=False, action='store_true', help="If present, we apply RMS normalization on input signals while training and testing")
    parser.add_argument("--tensorRT", action="store_true", default=False, help='Use TensorRT model' )
    parser.add_argument("--protocols", default=['802_11ax', '802_11b', '802_11n', '802_11g', 'noise'], help="Specify the list of classes")
    args = parser.parse_args()

    if args.frequency:
        freq = args.frequency

    if args.timeout:
        t_out = args.timeout

    MODEL_PATH = args.model_path
    MODEL_SIZE = args.model_size
    RMSNORM = args.RMSNorm
    if args.tensorRT:
        MODE = 'TensorRT'  # choices=['pytorch', 'TensorRT']
    else:
        MODE = 'pytorch'

    PROTOCOLS = args.protocols
    
    # if MODEL_SIZE == 'sm': #this causes more frequent buffer overflows in the receiver thread
    #     N = 2500

    rec = threading.Thread(target=receiver)
    rec.start()

    sp = threading.Thread(target=signalprocessing)
    sp.start()

    if MODE == 'pytorch':
        ml = threading.Thread(target=machinelearning)
    elif MODE == 'TensorRT':
        ml = threading.Thread(target=machinelearning_tensorRT)
    ml.start()

    # gracefully end program
    time.sleep(t_out)
    exitFlag = 1
    time.sleep(1)

    rec.join()
    sp.join()
    ml.join()
