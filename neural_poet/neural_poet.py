'''
Created on 27.11.2017
Authors: Mikko Karjanmaa, 
'''
import tensorflow as tf
import numpy as np
#from word_vectors.vectorizer import Vectorizer
#from word_vectors import tokenizer

def rnn_net(features, batch_size, seq_length, input_width, layer_widths):
    '''builds a densely connected recurrent neural net graph
    The implementation is (loosely) based on:
    http://arxiv.org/abs/1707.06130
    
    Arguments:
      features: input tensor of shape [batch_size, seq_length, input_width]
      seq_length: integer, sequence length
      layer_widths: list of integers:
        index 0: recurrent layer 1 output cells
        index 1: recurrent layer 2 output cells
        index 2: fully connected layer
        
    '''   
    lstm1 = tf.contrib.rnn.BasicLSTMCell(layer_widths[0])
    lstm2 = tf.contrib.rnn.BasicLSTMCell(layer_widths[1])
    state1 = lstm1.zero_state(batch_size, tf.float32)
    state2 = lstm2.zero_state(batch_size, tf.float32)
    
    for i in range(seq_length):
      
        '''layer 1: first recurrent layer''' 
        with tf.variable_scope("lstm1"):      
            recurrent_input1 = features[:, i]
            _, state1 = lstm1(recurrent_input1, state1)
            layer1_hidden = state1[1]
        
        '''layer 2: second recurrent layer'''
        with tf.variable_scope("lstm2"):
            recurrent_input2 = tf.concat([features[:, i], layer1_hidden], axis=1)
            _, state2 = lstm2(recurrent_input2, state2)
            layer2_hidden = state2[1]
    
    '''dense layer'''
    with tf.variable_scope("dense"):          
        dense_input = tf.concat([features[:, seq_length - 1], layer1_hidden, layer2_hidden], axis=1)
        net_output = tf.layers.dense(inputs=dense_input, units=layer_widths[2], activation=tf.nn.softmax)
    
    return net_output

feature_ph = tf.placeholder(dtype=tf.float32, shape=[None, 10, 300], name="feature_placeholder")
dummy = np.random.uniform(low=0.0, high=1.0, size=[1, 10, 300])

net_out = rnn_net(feature_ph, 1, 10, 196, [64, 64, 64])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(net_out, feed_dict={feature_ph: dummy}))