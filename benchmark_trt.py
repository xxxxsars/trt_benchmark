# -*- coding: utf-8 -*-
'''
Created on 2021/4/15

Author Andy Huang
'''
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit
import time
from PIL import Image

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet import decode_predictions

# x = tf.random.uniform([1, 3, 224, 224])

# input_size = 224

INPUT_SHAPE = (3, 224, 224)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
MAX_SIZE = 1 << 30  # 1GB
MAX_BATCH_SIZE = 1
DTYPE = trt.float32


def build_engine(model_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network,
                                                                                                               TRT_LOGGER) as parser:
        builder.max_workspace_size = MAX_SIZE
        builder.max_batch_size = MAX_BATCH_SIZE
        with open(model_path, "rb") as f:
            parser.parse(f.read())
        engine = builder.build_cuda_engine(network)
    return engine


def alloc_buf(engine):
    # h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    # h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)

    dtype = trt.nptype(DTYPE)
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=dtype)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=dtype)

    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    # np.copyto(h_input, (np.random.random((1, 3, input_size, input_size)).astype(np.float32)).reshape(-1))

    return h_input, h_output, d_input, d_output, stream


def load_input(img_path, host_buffer):
    with Image.open(img_path) as img:
        c, h, w = INPUT_SHAPE
        dtype = trt.nptype(DTYPE)
        img_array = np.asarray(img.resize((w, h), Image.BILINEAR)).transpose([2, 0, 1]).astype(dtype).ravel()
        # preprocess for mobilenet
        img_array = img_array / 127.5 - 1.0

    np.copyto(host_buffer, np.reshape(img_array, -1))


def inference(context, h_input, h_output, d_input, d_output, stream):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    return h_output


def run_trt_model(model_name):
    model_path = f"/tmp/trt_example/Model/{model_name}.onnx"
    img_path = "elephhant2.jpeg"

    engine = build_engine(model_path)
    context = engine.create_execution_context()

    # in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
    h_input, h_output, d_input, d_output, stream = alloc_buf(engine)

    load_input(img_path, h_input)

    t1 = time.time()
    # load_input("elephant.png",h_input)
    res = inference(context, h_input, h_output, d_input, d_output, stream)

    out = '{} Time : {} {} - Predicted: {} \n'.format(model_name, time.time() - t1, img_path,
                                                      decode_predictions(res.reshape(1, -1), top=3)[0][0][1])
    with open("trt_eport.txt", 'a+') as fin:
        fin.write(out)


if __name__ == "__main__":

    run_trt_model("vgg16")
    run_trt_model("vgg19")
    run_trt_model("resnet50")
    run_trt_model("resnet101")
    run_trt_model("densenet")
    run_trt_model("mobilenet")


