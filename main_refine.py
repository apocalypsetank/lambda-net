import scipy.io as sio
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import operation
from datetime import datetime
import os
import tensorflow as tf
import scipy.io as scipyio
import u_net
from scipy import interpolate
import math

os.environ["CUDA_VISIBLE_DEVICES"]=" 0"

phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
batch_size=8

images_input=tf.placeholder(tf.float32,shape=[batch_size,256,256,1])
ground_truth=tf.placeholder(tf.float32,shape=[batch_size,256,256,1])

output=u_net.inference(images_input,0.8,phase_train_placeholder)


L2_loss=tf.reduce_mean(tf.square(tf.subtract(ground_truth,output)))

global_step=tf.Variable(0,trainable=False)
train_op=tf.train.AdamOptimizer(learning_rate_placeholder).minimize(L2_loss,global_step=global_step,
                                                var_list=tf.trainable_variables())

saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)


sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())



for epoch in range(10000000):
    train_truth,train_input=operation.load_data_painting(800)
    index=np.random.choice(800,400)
    train_input[index]=np.flip(train_input[index,],1)
    train_truth[index]=np.flip(train_truth[index,],1)
    index=np.random.choice(800,400)
    train_input[index]=np.flip(train_input[index,],2)
    train_truth[index]=np.flip(train_truth[index,],2)
    for i in range(100):
        lr=0.0001

        out,train_err,_, step = sess.run([output,L2_loss,train_op, global_step], feed_dict={images_input: train_input[i*batch_size:i*batch_size+batch_size,:,:,:],
                                                                                ground_truth: train_truth[i*batch_size:i*batch_size+batch_size,:,:,:],
                                                                             phase_train_placeholder: True,
                                                                          learning_rate_placeholder: lr})

        print 'epoch:',epoch,i,'training_mse:',train_err

