#!/usr/bin/env python3
#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility script to benchmark the data rate that a neural network will support.
"""

import numpy as np
import time
import preprocessing.inference.trt_utils as trt_utils
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--plan', help='Indicate the source ONNX file to convert')
#parser.add_argument('--model', choices=['CNN1D', 'TRANSF_SM', 'TRANSF_LG'], help='specify the model type')
parser.add_argument('--nchan', type=int, help='Number of channels, where input shape for 1 sample is (1, NCHAN, SLICELEN)')
parser.add_argument('--slicelen', type=int, help='Length of input signal, where input shape for 1 sample is (1, NCHAN, SLICELEN)')
args, _ = parser.parse_known_args()

# Default inference settings.
PLAN_FILE_NAME = args.plan  # Plan file
CPLX_SAMPLES_PER_INFER = args.slicelen  # This should be half input_len from the neural network
BATCH_SIZE = 1  # Must be less than or equal to max_batch_size when creating plan file
NUM_BATCHES = 512  # Number of batches to run. Set to float('Inf') to run continuously
INPUT_DTYPE = np.float32
NCHAN = args.nchan


def plan_bench(plan_file_name=PLAN_FILE_NAME, cplx_samples=CPLX_SAMPLES_PER_INFER, num_chans=NCHAN,
         batch_size=BATCH_SIZE, num_batches=NUM_BATCHES, input_dtype=INPUT_DTYPE):
    # Setup the pyCUDA context
    trt_utils.make_cuda_context()

    # Use pyCUDA to create a shared memory buffer that will receive samples from the
    # AIR-T to be fed into the neural network.
    buff_len = num_chans * cplx_samples * batch_size # [MAURO] our shape is (1,2,512) for CNN1D
    sample_buffer = trt_utils.MappedBuffer(buff_len, input_dtype)

    # Set up the inference engine. Note that the output buffers are created for
    # us when we create the inference object.
    dnn = trt_utils.TrtInferFromPlan(plan_file_name, batch_size,
                                     sample_buffer)

    # Populate input buffer with test data
    dnn.input_buff.host[:] = np.random.randn(buff_len).astype(input_dtype)

    for _ in range(25):    # warmup GPU
        dnn.feed_forward()
    # Time the DNN Execution
    start_time = time.monotonic()
    for _ in range(num_batches):
        dnn.feed_forward()
    elapsed_time = time.monotonic() - start_time
    total_cplx_samples = cplx_samples * batch_size * num_batches

    throughput_msps = total_cplx_samples / elapsed_time / 1e6
    rate_gbps = throughput_msps * 2 * sample_buffer.host.itemsize * 8 / 1e3
    print('Result:')
    print('  Samples Processed : {:,}'.format(total_cplx_samples))
    print('  Tot. Processing Time   : {:0.3f} msec'.format(elapsed_time / 1e-3))
    print('  Avg. batch infer time  : {:0.6f} msec'.format((elapsed_time / num_batches) / 1e-3))
    print('  Throughput        : {:0.3f} MSPS'.format(throughput_msps))
    print('  Data Rate         : {:0.3f} Gbit / sec'.format(rate_gbps))


if __name__ == '__main__':
    plan_bench()


