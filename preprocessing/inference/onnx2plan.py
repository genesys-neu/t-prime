#!/usr/bin/env python3
#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import tensorrt as trt
import os
#import numpy as np
#from plan_bench import plan_bench

def onnx2plan(onnx_file_name, nchan, input_len, logger, MAX_BATCH_SIZE, MAX_WORKSPACE_SIZE, framework='pytorch', FP16_MODE=True, INPUT_NODE_NAME='input_buffer', BENCHMARK=False):
    assert((framework == 'pytorch') or (framework == 'tensorflow'))
    INPUT_PORT_NAME = '' if framework == 'pytorch' else ':0' # ':0' for tensorflow, '' for pytorch
    input_shape = (MAX_BATCH_SIZE, nchan, input_len)
    # File and path checking
    plan_file = onnx_file_name.replace('.onnx', '.plan')
    assert os.path.isfile(onnx_file_name), 'ONNX file not found: {}'.format(onnx_file_name)
    if os.path.isfile(plan_file):
        os.remove(plan_file)

    # Setup TensorRT builder and create network
    builder = trt.Builder(logger)
    batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags=batch_flag)

    # Parse the ONNX file
    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(onnx_file_name)

    # Define DNN parameters for inference
    builder.max_batch_size = MAX_BATCH_SIZE
    config = builder.create_builder_config()
    config.max_workspace_size = MAX_WORKSPACE_SIZE
    if FP16_MODE:
        config.set_flag(trt.BuilderFlag.FP16)

    # Optimize the network
    optimized_input_dims = (MAX_BATCH_SIZE, nchan, input_len)
    profile = builder.create_optimization_profile()
    input_name = INPUT_NODE_NAME + INPUT_PORT_NAME
    # Set the min, optimal, and max dimensions for the input layer.
    profile.set_shape(input_name, input_shape, optimized_input_dims,
                      optimized_input_dims)
    config.add_optimization_profile(profile)
    engine = builder.build_engine(network, config)

    # Write output plan file
    assert engine is not None, 'Unable to create TensorRT engine. Check settings'
    with open(plan_file, 'wb') as file:
        file.write(engine.serialize())

    # Print information to user
    if os.path.isfile(plan_file):
        print('\nONNX File Name  : {}'.format(onnx_file_name))
        print('ONNX File Size  : {}'.format(os.path.getsize(onnx_file_name)))
        print('PLAN File Name : {}'.format(plan_file))
        print('PLAN File Size : {}\n'.format(os.path.getsize(plan_file)))
        print('Network Parameters inference on AIR-T:')
        print('CPLX_SAMPLES_PER_INFER = {}'.format(input_len))
        print('BATCH_SIZE <= {}'.format(MAX_BATCH_SIZE))
        # TODO modify this function to match new parameters
        #if BENCHMARK:
        #    print('Running Inference Benchmark')
        #    plan_bench(plan_file_name=plan_file, cplx_samples=input_len, num_chans=nchan,
        #               batch_size=MAX_BATCH_SIZE, num_batches=512, input_dtype=np.float32)
    else:
        print('Result    : FAILED - plan file not created')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--onnx', help='Indicate the source ONNX file to convert')
    # parser.add_argument('--model', choices=['CNN1D', 'TRANSF_SM', 'TRANSF_LG'], help='specify the model type')
    parser.add_argument('--nchan', type=int,
                        help='Number of channels, where input shape for 1 sample is (1, NCHAN, SLICELEN)')
    parser.add_argument('--slicelen', type=int,
                        help='Length of input signal, where input shape for 1 sample is (1, NCHAN, SLICELEN)')
    args, _ = parser.parse_known_args()

    """
    This script converts an ONNX file to a optimized plan file using NVIDIA's TensorRT. It
    must be executed on the platform that will be used for inference, i.e., the AIR-T.
    """

    # Top-level inference settings.
    ONNX_FILE_NAME = args.onnx
    NCHAN = args.nchan
    INPUT_LEN = args.slicelen
    INPUT_PORT_NAME = ''  # ':0' for tensorflow, '' for pytorch
    """
    # CNN1D configuration
    ONNX_FILE_NAME = '/home/deepwave/Research/DSTL/dstl/baseline_models/dstl_baseline_CNN1D.onnx'  # Name of input CNN onnx file
    NCHAN = 2
    INPUT_LEN = 512  # Length of the input buffer (# of elements)

    # transformer configuration (SMALL)
    ONNX_FILE_NAME = '/home/deepwave/Research/DSTL/dstl/dstl_transformer/dstl_baseline_transformer_SMALL.onnx'  # Name of input Transfomer onnx file
    NCHAN = 24
    INPUT_LEN = 128  # Length of the input buffer (# of elements)
    # transformer configuration (LARGE)
    ONNX_FILE_NAME = '/home/deepwave/Research/DSTL/dstl/dstl_transformer/dstl_baseline_transformer_SMALL.onnx'  # Name of input Transfomer onnx file
    NCHAN = 64
    INPUT_LEN = 256  # Length of the input buffer (# of elements)
    """
    INPUT_NODE_NAME = 'input_buffer'  # Input node name defined in dnn
    MAX_WORKSPACE_SIZE = 1073741824  # 1 GB for example
    MAX_BATCH_SIZE = 1  # Maximum batch size for which plan file will be optimized
    FP16_MODE = True  # Use float16 if possible (all layers may not support this)
    LOGGER = trt.Logger(trt.Logger.VERBOSE)
    BENCHMARK = False  # Run plan file benchmarking at end
    onnx2plan(ONNX_FILE_NAME, NCHAN, INPUT_LEN, LOGGER, MAX_BATCH_SIZE, MAX_WORKSPACE_SIZE, 'pytorch')


