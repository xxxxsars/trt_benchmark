# -*- coding: utf-8 -*-
'''
Created on 2021/4/15

Author Andy Huang
'''
import tensorflow as tf
import onnx
from tensorflow.keras.models import load_model
import onnxmltools
from benchmark import get_model
import multiprocessing


def save_h5(model_name):
    keras_path = '/tmp/trt_example/Model/tmp.h5'
    model = get_model(model_name)[0]
    model.save(keras_path)


def save_onnx(model_name):
    keras_path = '/tmp/trt_example/Model/tmp.h5'
    onnx_path = f'/tmp/trt_example/Model/{model_name}.onnx'
    keras_model = load_model(keras_path)
    onnx_model = onnxmltools.convert_keras(keras_model)
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_path)

def gen_onnx(model_name):

    p = multiprocessing.Process(target=save_h5,args=(model_name,))
    p.start()
    p.join()

    p = multiprocessing.Process(target=save_onnx, args=(model_name,))
    p.start()
    p.join()


if __name__ == "__main__":
    gen_onnx("vgg16")
    gen_onnx("vgg19")
    gen_onnx("resnet50")
    gen_onnx("resnet101")
    gen_onnx("densenet")
    gen_onnx("mobilenet")




