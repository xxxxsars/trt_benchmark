# -*- coding: utf-8 -*-
'''
Created on 2021/4/15

Author Andy Huang
'''
from tensorflow.keras.applications import vgg16,vgg19,resnet50,mobilenet,densenet,resnet,ResNet101
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np




def get_model(model_name):

    if model_name == "vgg16":
        return vgg16.VGG16(weights='imagenet'),vgg16.decode_predictions,vgg16.preprocess_input
    elif model_name =="vgg19":
        return  vgg19.VGG19(weights='imagenet'), vgg19.decode_predictions,vgg19.preprocess_input
    elif model_name =="resnet50":
        return resnet50.ResNet50(weights='imagenet'),resnet50.decode_predictions,resnet50.preprocess_input
    elif model_name =="resnet101":
        return ResNet101(weights='imagenet'),resnet.decode_predictions,resnet.preprocess_input
    elif model_name == "mobilenet":
        return mobilenet.MobileNet(weights='imagenet'),mobilenet.decode_predictions,mobilenet.preprocess_input
    elif model_name == "densenet":
        return densenet.DenseNet121(weights='imagenet'),densenet.decode_predictions,densenet.preprocess_input

def run_model(model_name):
    model_data = get_model(model_name)
    img_path = 'elephhant2.jpeg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = model_data[2](x)

    t1 = time.time()
    preds = model_data[0].predict(x)


    out = ('{} Time : {} {} - Predicted: {} \n'.format( model_name,time.time() -t1,img_path, model_data[1](preds, top=3)[0][0][1]))
    with open("report.txt",'a+') as fin:
        fin.write(out)
    K.clear_session()


if __name__ == "__main__":
    # config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    # sess = tf.compat.v1.Session(config=config)

    import multiprocessing

    p = multiprocessing.Process(target=run_model, args=("vgg16",))
    p.start()
    p.join()

    p = multiprocessing.Process(target=run_model, args=("vgg19",))
    p.start()
    p.join()

    p = multiprocessing.Process(target=run_model, args=("resnet50",))
    p.start()
    p.join()


    p = multiprocessing.Process(target=run_model, args=("resnet101",))
    p.start()
    p.join()

    p = multiprocessing.Process(target=run_model, args=("densenet",))
    p.start()
    p.join()

    p = multiprocessing.Process(target=run_model, args=("mobilenet",))
    p.start()
    p.join()







