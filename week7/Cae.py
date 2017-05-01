
# coding: utf-8

# In[ ]:

def build(inputVar):
    import lasagne
    
    # Hyper Params
    nonlin = lasagne.nonlinearities.rectify
    
    # Input 3*64*64 
    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64), input_var=inputVar)
       
    # Conv 64
    network = lasagne.layers.BatchNormLayer(lasagne.layers.Conv2DLayer(network, 
            num_filters=16, 
            filter_size=(9,9),
            nonlinearity=nonlin,
            W=lasagne.init.GlorotUniform(),
            pad = 'valid'))  
    
    print ("Network output:", network.output_shape)
           
    # MaxPool 56
    network = lasagne.layers.MaxPool2DLayer(network, 
            pool_size=(2,2))
    
    print ("Network output:", network.output_shape)
    
    # Conv 28
    network = lasagne.layers.BatchNormLayer(lasagne.layers.Conv2DLayer(network, 
            num_filters=16, 
            filter_size=(3,3),
            nonlinearity=nonlin,
            pad = 'valid'))
    
    print ("Network output:", network.output_shape)
    
    # MaxPool 26
    network = lasagne.layers.MaxPool2DLayer(network, 
            pool_size=(2,2))
    
    print ("Network output:", network.output_shape)
    
    # Dense 13
    network = lasagne.layers.DropoutLayer(lasagne.layers.DenseLayer(network,
           num_units=2048,
           nonlinearity=nonlin),
           p = 0.5)
    
    # Dense
    network = lasagne.layers.DropoutLayer(lasagne.layers.DenseLayer(network,
           num_units=3136,
           nonlinearity=nonlin),
           p = 0.5)
    
    # Reshape 
    network = lasagne.layers.ReshapeLayer(network,
            shape=(([0], 16, 14, 14)))
    
    print ("Network output:", network.output_shape)
 
    # Upscale 16*14*14
    network = lasagne.layers.Upscale2DLayer(network,
            scale_factor=(2,2))
    
    print ("Network output:", network.output_shape)
 
    # Conv 16*28*28
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=3, 
            filter_size=(5,5),
            nonlinearity=nonlin,
            pad = 'full')
    
    print ("Network output:", network.output_shape)

    # Reshape
    network = lasagne.layers.ReshapeLayer(network,
            shape=(-1,3,32,32))
    
    print('network params', lasagne.layers.count_params(network))

    return network

