'''
Created on 27.11.2017
Authors: Mikko Karjanmaa, -- your name here! ---
'''
import tensorflow as tf
import numpy as np
import os
from batch_manager.batch_manager import BatchManager

def rnn_net(features, batch_size, seq_length, layer_widths, reuse=False):
    '''builds a densely connected recurrent neural net graph
    The implementation is (loosely) based on:
    http://arxiv.org/abs/1707.06130
    
    Arguments:
      features: input tensor of shape [batch_size, seq_length, ...]
      batch_size, seq_length: python integers
      layer_widths: list of integers:
        index 0: recurrent layer 1 output cells
        index 1: recurrent layer 2 output cells
        index 2: fully connected layer
        
    to enable layer normalization
    lstm1 = tf.contrib.rnn.LayerNormBasicLSTMCell(layer_widths[0])
    lstm2 = tf.contrib.rnn.LayerNormBasicLSTMCell(layer_widths[1])
    '''
    lstm1 = tf.contrib.rnn.BasicLSTMCell(layer_widths[0])
    lstm2 = tf.contrib.rnn.BasicLSTMCell(layer_widths[1]) 
     
    state1 = lstm1.zero_state(batch_size, tf.float32)
    state2 = lstm2.zero_state(batch_size, tf.float32)
    
    for i in range(seq_length):
      
        '''layer 1: first recurrent layer''' 
        with tf.variable_scope("lstm1", reuse=reuse):      
            recurrent_input1 = features[:, i]
            _, state1 = lstm1(recurrent_input1, state1)
            layer1_hidden = state1[1]
        
        '''layer 2: second recurrent layer'''
        with tf.variable_scope("lstm2", reuse=reuse):
            recurrent_input2 = tf.concat([features[:, i], layer1_hidden], axis=1)
            _, state2 = lstm2(recurrent_input2, state2)
            layer2_hidden = state2[1]
    
    '''dense layer'''
    with tf.variable_scope("dense", reuse=reuse):          
        dense_input = tf.concat([features[:, seq_length - 1], layer1_hidden, layer2_hidden], axis=1)
        net_output = tf.layers.dense(inputs=dense_input, units=layer_widths[2], activation=tf.nn.tanh)
    
    return net_output

batch_size = 32
feature_length = 8
feature_ph = tf.placeholder(dtype=tf.float32, shape=[None, feature_length, 300], name="feature_placeholder")
dummy = np.random.uniform(low=0.0, high=1.0, size=[2, feature_length, 300])

net_out = rnn_net(feature_ph, batch_size, feature_length - 1, [64, 64, 300])

'''TODO we should try different distance metrics'''
def loss(model, ground_truth):
    return tf.reduce_sum(tf.norm(model - ground_truth[:,feature_length - 1,:], axis=1)) #euclidean distance
  
training = tf.train.GradientDescentOptimizer(1e-3).minimize(loss(net_out, feature_ph))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

dataset = BatchManager(batch_size = batch_size, sequence_length = feature_length)
dataset.read_csv(os.path.join('dataset','RedditPoetry.csv'), column=9)
print(dataset.next_batch())

while(True):
    current_batch = dataset.next_batch()
    sess.run(training, feed_dict={feature_ph: current_batch})
    print(sess.run(loss(net_out, feature_ph), feed_dict={feature_ph: current_batch}))