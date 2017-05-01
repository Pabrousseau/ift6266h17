# coding: utf-8

import sys
import os
import time
import glob
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

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
    num_epochs = 301
    batchsize = 128
    initial_eta = 0.0004
    
    # Load the dataset
    print("Loading data...")
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    X_train, y_train, _, _, _, _ = load_dataset()
    #_, y_train, _, y_val, _, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    noiseVar = T.matrix('noise')
    inputVar = T.tensor4('inputs')

    # Build Network
    print("Building model and compiling functions...")
    generator = Dcgan.buildGenerator(noiseVar)
    discriminator = Dcgan.buildDiscriminator(inputVar)

    # Output expressions
    realOut = lasagne.layers.get_output(discriminator)
    fakeOut = lasagne.layers.get_output(discriminator, lasagne.layers.get_output(generator))
    
    # Loss expressions
    generatorLoss = lasagne.objectives.binary_crossentropy(fakeOut, 1).mean()
    discriminatorLoss = (lasagne.objectives.binary_crossentropy(realOut, 1) + lasagne.objectives.binary_crossentropy(fakeOut, 0)).mean()

    # Update expressions 
    learning_rate = theano.shared(lasagne.utils.floatX(initial_eta))
    generatorParams = lasagne.layers.get_all_params(generator, trainable=True)
    discriminatorParams = lasagne.layers.get_all_params(discriminator, trainable=True)
    updates = lasagne.updates.adam(generatorLoss, generatorParams, learning_rate, beta1=0.5)
    updates.update(lasagne.updates.adam(discriminatorLoss, discriminatorParams, learning_rate, beta1=0.5))

    # Train Function
    train_fn = theano.function([noiseVar, inputVar], [(realOut > .5).mean(), (fakeOut < .5).mean()], updates=updates)
    # Data generating function
    gen_fn = theano.function([noiseVar], lasagne.layers.get_output(generator, deterministic=True))

    # Model Load if resumable
    with np.load('09dcgan_gen300.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(generator, param_values)
    with np.load('09dcgan_disc300.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(discriminator, param_values)

    # Training Loop
    print("Starting training...")
    # We iterate over epochs
    for epoch in range(301, 801):
        # Full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batchsize, shuffle=True):
            inputs, targets = batch
            Z = inputs
            Z[:, :, 16:48, 16:48] = targets
            noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))
            train_err += np.array(train_fn(noise, Z))
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{}".format(train_err / train_batches))

        if epoch%10==0:
            # Gen image
            X = gen_fn(lasagne.utils.floatX(np.random.rand(64, 100)))
            im = Image.fromarray(
                (X.reshape(8, 8, 3, 64, 64).transpose(1, 3, 0, 4, 2) * 255).astype('uint8').reshape(8 * 64, 8 * 64, 3))
            im.save('09dcgan'+str(epoch)+'.jpeg')

        if epoch % 15 == 0:
            # Save model
            np.savez('09dcgan_gen'+str(epoch)+'.npz', *lasagne.layers.get_all_param_values(generator))
            np.savez('09dcgan_disc'+str(epoch)+'.npz', *lasagne.layers.get_all_param_values(discriminator))

if __name__ == '__main__':
    main()

