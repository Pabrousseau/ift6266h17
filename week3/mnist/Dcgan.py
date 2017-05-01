# coding: utf-8

import lasagne
import numpy as np
import theano
import theano.tensor as T


def buildGenerator(input_var=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm, TransposedConv2DLayer, Conv2DLayer
    from lasagne.nonlinearities import sigmoid, LeakyRectify
    
    lrelu = LeakyRectify(0.2)

    # input
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    
    #layer = batch_norm(Conv2DLayer(layer, 12, 5, nonlinearity=lrelu))
    
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 128*6*6, nonlinearity=lrelu))
    layer = ReshapeLayer(layer, ([0], 128, 6, 6))
    
    # two fractional-stride convolutions
    layer = batch_norm(TransposedConv2DLayer(layer, 64, 3, stride=2, nonlinearity=lrelu))
    layer = batch_norm(TransposedConv2DLayer(layer, 1, 4, stride=2, nonlinearity=lrelu))
    
    print ("Generator output:", layer.output_shape)
    print('num params', lasagne.layers.count_params(layer))
    return layer


def buildDiscriminator(input_var=None):
    from lasagne.layers import InputLayer, Conv2DLayer, ReshapeLayer, DenseLayer, batch_norm
    #from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer  # override CuDnn
    from lasagne.nonlinearities import LeakyRectify, sigmoid
    
    lrelu = LeakyRectify(0.2)
    
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    
    # two convolutions
    layer = batch_norm(Conv2DLayer(layer, 32, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad=2, nonlinearity=lrelu))
    
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    
    # output layer
    layer = DenseLayer(layer, 1, nonlinearity=sigmoid)
    
    print ("Discriminator output:", layer.output_shape)
    print('num params', lasagne.layers.count_params(layer))
    return layer

