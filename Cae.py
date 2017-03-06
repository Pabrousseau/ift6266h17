
# coding: utf-8

# In[ ]:

def build(inputVar):
    import lasagne
    
    # Hyper Params
    numFilters = 24
    filterSize = (5,5)
    poolSize = (2,2)
    denseM = 512 
    encode = 256
    nonlin = lasagne.nonlinearities.rectify
    
    # Input 3*64*64 
    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64), input_var=inputVar)
       
    # Conv 24*60*60
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=numFilters, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            W=lasagne.init.GlorotUniform(),
            pad = 'valid')
    
    # Conv 24*56*56
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=numFilters, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            pad = 'valid')

    # MaxPool 48*28*28
    network = lasagne.layers.MaxPool2DLayer(network, 
            pool_size=poolSize)
    
    # Conv 48*24*24
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=2*numFilters, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            pad = 'valid')

    # MaxPool 48*12*12
    network = lasagne.layers.MaxPool2DLayer(network, 
            pool_size=poolSize)
    
    # Reshape 6912 
    network = lasagne.layers.ReshapeLayer(network,
            shape=(([0], -1)))
     
    # Dense 512
    network = lasagne.layers.DenseLayer(network,
            num_units=denseM,
            nonlinearity=nonlin)
    
    # Dense 256
    network = lasagne.layers.DenseLayer(network,
            num_units=encode,
            nonlinearity=nonlin)
    
    # Dense 512
    network = lasagne.layers.DenseLayer(network,
            num_units=denseM,
            nonlinearity=nonlin)
    
    # Dense 6912
    network = lasagne.layers.DenseLayer(network,
            num_units=6912,
            nonlinearity=nonlin)
    
    # Reshape 48*12*12
    network = lasagne.layers.ReshapeLayer(network,
            shape=(([0], 2*numFilters, 12, 12)))

    # Upscale 48*24*24
    network = lasagne.layers.Upscale2DLayer(network,
            scale_factor=poolSize)
    
    # Conv 24*28*28
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=numFilters, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            pad = 'full')
    
    # Conv 3*32*32
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=3, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            pad = 'full')
    
    # Reshape
    network = lasagne.layers.ReshapeLayer(network,
            shape=(-1,3,32,32))
    
    return network

