'''
Created on 27.11.2017
Authors: Mikko Karjanmaa, -- your name here! ---
'''
import tensorflow as tf
import numpy as np
import os
import random
from batch_manager.batch_manager import BatchManager
from word_vectors.vectorizer import Vectorizer

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
    '''
    lstm1 = tf.contrib.rnn.BasicLSTMCell(layer_widths[0])
    lstm2 = tf.contrib.rnn.BasicLSTMCell(layer_widths[1]) 
    '''
    lstm1 = tf.contrib.rnn.BasicLSTMCell(layer_widths[0], activation=tf.nn.softsign)
    lstm2 = tf.contrib.rnn.BasicLSTMCell(layer_widths[1], activation=tf.nn.softsign)
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
        net_output = tf.layers.dense(inputs=dense_input, units=layer_widths[2], activation=tf.nn.relu)
    
    return net_output

vectorizer = Vectorizer()
model_name='model3'
batch_size = 4096
feature_length = 9
test_batch_size = 100
test_print_size = 20
random_dict_words = 14
training_mode = False
feature_ph = tf.placeholder(dtype=tf.float32, shape=[None, feature_length, 300], name="feature_placeholder")
wordvec_ph = tf.placeholder(dtype=tf.float32, shape=[None, 300], name="word_vector_placeholder")
random_word_ph = tf.placeholder(dtype=tf.float32, shape=[None, random_dict_words, 300], name="random_word_placeholder")
dummy = np.random.uniform(low=0.0, high=1.0, size=[2, feature_length, 300])

'''
We are going to interpret the rnn output as a description of a probability space, and we are going to use
monte carlo sampling to evaluate the probability space's correctness.'''
def prob_transform_nn(probability_space, word_vector, reuse=False):
    with tf.variable_scope("prob_transform_l1", reuse=reuse):  
        layer1 = tf.layers.dense(inputs=tf.concat([probability_space, word_vector], axis= 1), units=128, activation=tf.nn.crelu)
    with tf.variable_scope("prob_transform_l2", reuse=reuse): 
        prob_transform = tf.layers.dense(inputs=tf.concat([layer1, probability_space], axis= 1), units=1, activation=tf.nn.sigmoid)
    return prob_transform

negative_samples = vectorizer.all_vectors()
total_words = batch_size * random_dict_words
word_dictionary = tf.constant(negative_samples, dtype=tf.float32)
random_words = tf.tile(word_dictionary, [total_words//len(negative_samples)+1, 1])

net_out = rnn_net(feature_ph, batch_size, feature_length - 1, [128, 96, 128])
test_out = rnn_net(feature_ph, test_batch_size, feature_length - 1, [128, 96, 128], reuse=True)
dict_out = rnn_net(tf.tile(feature_ph, [len(negative_samples),1,1]), len(negative_samples), feature_length - 1, [128, 96, 128], reuse=True)
def random_vectors(stddev=0.15):
    return tf.random_normal([batch_size, 300], 0.0, stddev=stddev, dtype=tf.float32)
prob_random = prob_transform_nn(net_out, random_vectors(), reuse= False)


def random_probs(model, ground_truth):
    '''calculates (max of) 10 random vector probs per batch and returns mean'''
    prob_rand = []
    for i in range(4):
        prob_rand.append(prob_transform_nn(model, random_vectors(), reuse= True))
    randomized_dict = tf.random_shuffle(random_words)
    for i in range(random_dict_words):
        pick = randomized_dict[batch_size*i: batch_size*(i+1)]
        prob_rand.append(prob_transform_nn(model, pick, reuse= True))
    #return tf.reduce_mean(tf.reduce_max(tf.concat(prob_rand, axis=1),axis=1))
    max_val =  tf.reduce_mean(tf.reduce_max(tf.concat(prob_rand, axis=1),axis=1))
    min_val =  tf.reduce_mean(tf.reduce_min(tf.concat(prob_rand, axis=1),axis=1))
    return tf.reduce_mean(tf.concat(prob_rand, axis=1) ), max_val, min_val

def loss(model, ground_truth):
    '''return tf.reduce_sum(tf.norm(model - ground_truth[:,feature_length - 1,:], axis=1)) #euclidean distance to gt'''
    avg_rand , max_rand, min_rand = random_probs(model, ground_truth)
    shuffled_words = tf.random_shuffle(ground_truth[:,feature_length - 1,:])
    prob_rand = prob_transform_nn(model, shuffled_words, reuse = True)   
    prob_gt = prob_transform_nn(model, ground_truth[:,feature_length - 1,:], reuse = True)
    prob_gt_mean = tf.reduce_mean(prob_gt)
    prob_rand_mean = tf.reduce_mean(prob_rand)
    return prob_rand_mean - prob_gt_mean + avg_rand + max_rand

training = tf.train.AdamOptimizer(6e-5).minimize(loss(net_out, feature_ph))

print('starting tf session..')
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, 'models/'+model_name)

dataset = BatchManager(batch_size = batch_size, sequence_length = feature_length)
dataset.read_csv(os.path.join('dataset','RedditPoetry.csv'), column=9)

iterationcount = 0

'''build random testing dictionary of tuples containing word vector and string'''
print('building test dictionary..')
testwords = ["i", "you", "the", "this", "is"]
for i in range(len(testwords)):
    testwords[i] = (vectorizer.vectorize_tokens([testwords[i]])[0],  testwords[i])
for i in range(test_batch_size - len(testwords)):
    idx = random.randint(0, 10000)
    testwords.append(vectorizer.tuple_at_index(idx))

if (training_mode):
    print('starting training..')
    while(True):    
        for j in range(100):
            current_batch = dataset.next_batch()
            sess.run(training, feed_dict={feature_ph: current_batch})
        print(sess.run(loss(net_out, feature_ph), feed_dict={feature_ph: current_batch}))    
        #print ground truth
        for i in range(feature_length):
            print(vectorizer.find_nearest(current_batch[0,i]))  
             
        ground_truth__prob = sess.run(prob_transform_nn(test_out, wordvec_ph, reuse= True),
                        feed_dict={feature_ph: np.tile(current_batch[None, 0, :, :], (test_batch_size,1,1) ),
                                   wordvec_ph: np.tile(current_batch[None, 0, feature_length - 1, :], (test_batch_size,1))})[0]  
        print(' = {}'.format(ground_truth__prob))
        col_width = max(len(val[1]) for val in testwords)
        test_batch = np.array([val[0] for val in testwords])        
        prob = sess.run(prob_transform_nn(test_out, wordvec_ph, reuse= True),
                        feed_dict={feature_ph: np.tile(current_batch[None, 0, :, :], (test_batch_size,1,1) ),
                                   wordvec_ph: test_batch})    
        sorted_words = []
        for i in range(len(testwords)):
            sorted_words.append((testwords[i][1], prob[i]))
        sorted_words.sort(key=lambda tup: tup[1], reverse = True)
        
        for i in range(test_print_size):
            print('{0:>{1}}: {2}'.format(sorted_words[i][0], col_width, sorted_words[i][1]))
        print('---')
        iterationcount+=1
        if iterationcount % 10 == 9:        
            print('saving {}...'.format(model_name), end='')
            saver.save(sess, 'models/'+model_name)
            print('...done.')
            
print('starting stext generation...')
text_history = np.zeros((feature_length, 300))
words = [elem for elem in vectorizer.all_tuples()]
while(True):
    word_probs = sess.run(prob_transform_nn(dict_out, word_dictionary, reuse=True), feed_dict={feature_ph: [text_history]})[:,0]
    word_probs = word_probs**1.5
    word_probs = word_probs/np.sum(word_probs)
    next_word_idx = np.random.choice(range(len(words)), p=word_probs)
    next_word_vec, next_word = words[next_word_idx]
    print(next_word)
    text_history[:feature_length - 1] = text_history[1:feature_length]
    text_history[feature_length - 1] = next_word_vec