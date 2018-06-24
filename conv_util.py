import numpy as np
import tensorflow as tf

def conv_layer(
    x,
    input_channels, 
    filter_height, 
    filter_width, 
    num_filters, 
    stride_y, 
    stride_x, 
    padding, 
    name):

    with tf.variable_scope(name):

        weights = tf.get_variable(
            name='weights', 
            shape=[filter_height, filter_width, input_channels, num_filters])

        biases = tf.get_variable(
            name='biases', 
            shape=[num_filters])
        
        conv = tf.nn.conv2d(
            input=x,
            filter=weights,
            strides=[1, stride_y, stride_x, 1], 
            padding=padding)

        bias = tf.nn.bias_add(
            value=conv, 
            bias=biases)

        relu = tf.nn.relu(
            features=bias)        

        # Summaries
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", relu)

        return relu

def fc_layer(
    x,
    num_input,
    num_output, 
    name,
    is_relu):
    
    with tf.variable_scope(name):

        weights = tf.get_variable(
            name='weights', 
            shape=[num_input, num_output])

        biases = tf.get_variable(
            name='biases', 
            shape=[num_output])

        bias = tf.nn.xw_plus_b(
            x=x,
            weights=weights, 
            biases=biases)

        # Summaries
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)

        if is_relu:

            relu = tf.nn.relu(
                features=bias)

            tf.summary.histogram("relu", relu)

            return relu
        else:
            return bias

def max_pool(
    x,
    filter_height,
    filter_width, 
    stride_y, 
    stride_x, 
    padding, 
    name):

    return tf.nn.max_pool(
        value=x, 
        ksize=[1, filter_height, filter_width, 1], 
        strides=[1, stride_y, stride_x, 1], 
        padding=padding, 
        name=name)


def dropout(
    x,
    keep_rate):
    
    return tf.nn.dropout(
        x=x,
        keep_prob=keep_rate)

def build_model(
    x,
    keep_rate,
    input_dim,
    mean):

    # Conv layer #1
    conv1 = conv_layer(
        x=(x-mean),
        input_channels=3, 
        filter_height=3, 
        filter_width=3, 
        num_filters=32, 
        stride_y=1, 
        stride_x=1, 
        padding='SAME', 
        name='conv1')
    
    # Conv layer #2
    conv2 = conv_layer(
        x=conv1,
        input_channels=32, 
        filter_height=3, 
        filter_width=3, 
        num_filters=64, 
        stride_y=1, 
        stride_x=1, 
        padding='SAME', 
        name='conv2')

    # Fc layer #3
    flat = tf.reshape(
        tensor=conv2,
        shape=[-1, input_dim[0]*input_dim[1]*64])

    fc3 = fc_layer(
        x=flat,
        num_input=input_dim[0]*input_dim[1]*64,
        num_output=1024, 
        name='fc3',
        is_relu=True)

#     drop = dropout(
#         x=fc5,
#         keep_rate=keep_rate)

    # Logit layer #6
    fc4 = fc_layer(
        x=fc3,
        num_input=1024,
        num_output=6, 
        name='fc4',
        is_relu=False)

    return fc4