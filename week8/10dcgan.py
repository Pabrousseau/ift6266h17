# coding: utf-8

import sys
import os
import time
import glob
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import theano
import theano.tensor as T

import lasagne

import Dcgan


def im2ar(x):
    y = [[[0]*len(x)]*len(x)]*3
    y[0] = x[:,:,0]
    y[1] = x[:,:,1]
    y[2] = x[:,:,2]
    return y


# Load data
def load_dataset():
    # Path
    mscoco = "/home/pab/Documents/Lasagne/inpainting"
    split = "train2014"
    data_path = os.path.join(mscoco, split)
    imgs = glob.glob(data_path + "/*.jpg")

    X_train = []
    y_train = []

    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        img_array = np.divide(np.array(img, dtype='float32'), 255)

        if len(img_array.shape) == 3:
            temp = np.copy(img_array)
            input = np.copy(img_array)
            input[16:48, 16:48, :] = 0
            target = img_array[16:48, 16:48, :]
        else:
            input[:, :, 0] = np.copy(img_array)
            input[:, :, 1] = np.copy(img_array)
            input[:, :, 2] = np.copy(img_array)
            target = input[16:48, 16:48, :]
            input[16:48, 16:48, :] = 0

        X_train.append(im2ar(input))
        y_train.append(im2ar(target))

    split = "val2014"
    data_path = os.path.join(mscoco, split)
    imgs = glob.glob(data_path + "/*.jpg")

    X_val = []
    y_val = []

    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        img_array = np.divide(np.array(img, dtype='float32'), 255)

        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[16:48, 16:48, :] = 0
            target = img_array[16:48, 16:48, :]
        else:
            input[:, :, 0] = np.copy(img_array)
            input[:, :, 1] = np.copy(img_array)
            input[:, :, 2] = np.copy(img_array)
            target = input[16:48, 16:48, :]
            input[16:48, 16:48, :] = 0

        X_val.append(im2ar(input))
        y_val.append(im2ar(target))

    # We reserve the last 10000 training examples for testing.
    X_val, X_test = X_val[:-10000], X_val[-10000:]
    y_val, y_test = y_val[:-10000], y_val[-10000:]

    return (np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test))


# Batch iterator
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# Main
def main():
    
    # Hyper Params
    num_epochs = 101
    batchsize = 128
    initial_eta = 0.0005

    # Load the dataset
    print("Loading data...")
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    _, _, _, _, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    noiseVar = T.matrix('noise')
    inputVar = T.tensor4('inputs')

    # Build Network
    print("Building model and compiling functions...")
    generator = Dcgan.buildGenerator(noiseVar)
    discriminator = Dcgan.buildDiscriminator(inputVar)

    # Update function
    disc_fn = theano.function([inputVar], lasagne.layers.get_output(discriminator))
    gen_fn = theano.function([noiseVar], lasagne.layers.get_output(generator))

    # Model Load if resumable
    with np.load('09dcgan_gen200.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(generator, param_values)
    with np.load('09dcgan_disc200.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(discriminator, param_values)


    print('Start completing')
    n = 1000
    R = []
    for i in range(64):
        Z = np.array([X_test[i]])

        X = gen_fn(lasagne.utils.floatX(np.random.rand(n, 100)))
        M = np.array([Z[0] for _ in range(n)])
        M[:, :, 16:48, 16:48] = deepcopy(X[:, :, 16:48, 16:48])
        X[:, :, 16:48, 16:48] = 0
        D = np.transpose(np.log(1-disc_fn(M)/2))
        S = ((Z - X) ** 2).mean(axis=1).mean(axis=1).mean(axis=1)
        L = np.add(S, 0.01*D[0])
        R.append(M[np.argmin(S)])

    R = np.array(R)
    im = Image.fromarray(
        (R.reshape(8, 8, 3, 64, 64).transpose(1, 3, 0, 4, 2) * 255).astype('uint8').reshape(8 * 64, 8 * 64, 3))
    im.save('10dcgan.jpeg')


if __name__ == '__main__':
    main()

