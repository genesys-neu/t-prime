#!/usr/bin/env python3
#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
This script converts an ONNX file to a optimized plan file using NVIDIA's TensorRT. It
must be executed on the platform that will be used for inference, i.e., the AIR-T.
"""

import tensorrt as trt
import os
from plan_benchplan_bench import plan_bench

# Top-level inference settings.
ONNX_FILE_NAME = '/home/deepwave/Research/DSTL/dstl/cnn_baseline/dstl_baseline_CNN1D.onnx'  # Name of input onnx file
INPUT_NODE_NAME = 'input_buffer'  # Input node name defined in dnn
INPUT_PORT_NAME = ''  # ':0' for tensorflow, '' for pytorch
INPUT_LEN = 512  # Length of the input buffer (# of elements)
MAX_BATCH_SIZE = 1  # Maximum batch size for which plan file will be optimized
MAX_WORKSPACE_SIZE = 1073741824  # 1 GB for example
FP16_MODE = True  # Use float16 if possible (all layers may not support this)
LOGGER = trt.Logger(trt.Logger.VERBOSE)
BENCHMARK = False  # Run plan file benchmarking at end
input_shape = (1, 2, INPUT_LEN)

def main():
    # File and path checking
    plan_file = ONNX_FILE_NAME.replace('.onnx', '.plan')
    assert os.path.isfile(ONNX_FILE_NAME), 'ONNX file not found: {}'.format(ONNX_FILE_NAME)
    if os.path.isfile(plan_file):
        os.remove(plan_file)

    # Setup TensorRT builder and create network
    builder = trt.Builder(LOGGER)
    batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags=batch_flag)

    # Parse the ONNX file
    parser = trt.OnnxParser(network, LOGGER)
    parser.parse_from_file(ONNX_FILE_NAME)

    # Define DNN parameters for inference
    builder.max_batch_size = MAX_BATCH_SIZE
    config = builder.create_builder_config()
    config.max_workspace_size = MAX_WORKSPACE_SIZE
    if FP16_MODE:
        config.set_flag(trt.BuilderFlag.FP16)

    # Optimize the network
    optimized_input_dims = (MAX_BATCH_SIZE, INPUT_LEN)
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
        print('\nONNX File Name  : {}'.format(ONNX_FILE_NAME))
        print('ONNX File Size  : {}'.format(os.path.getsize(ONNX_FILE_NAME)))
        print('PLAN File Name : {}'.format(plan_file))
        print('PLAN File Size : {}\n'.format(os.path.getsize(plan_file)))
        print('Network Parameters inference on AIR-T:')
        print('CPLX_SAMPLES_PER_INFER = {}'.format(INPUT_LEN))
        print('BATCH_SIZE <= {}'.format(MAX_BATCH_SIZE))
        if BENCHMARK:
            print('Running Inference Benchmark')
            plan_bench(plan_file_name=plan_file, cplx_samples=INPUT_LEN,
                       batch_size=MAX_BATCH_SIZE)
    else:
        print('Result    : FAILED - plan file not created')


if __name__ == '__main__':
    main()