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
import deeper_u_net
import math
from scipy import interpolate
import math
import discriminator

os.environ["CUDA_VISIBLE_DEVICES"]="0 "

subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

model_dir = os.path.join('models/', subdir)
if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
    os.makedirs(model_dir)

result_dir = os.path.join('result/', subdir)
if not os.path.isdir(result_dir):  # Create the result directory if it doesn't exist
    os.makedirs(result_dir)

phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

batch_size=10
masks=sio.loadmat("mask.mat")['data']



training_data_whole = operation.load_hscnn()
testing_data=sio.loadmat('testing_data/test.mat')['data'] #(10,256,256,24)
testing_data=operation.normalize_0_to_1(testing_data)
measurement_test=np.sum(testing_data*masks,axis=3)[:,:,:,np.newaxis] # 20,256,256,1
test_input=np.concatenate((measurement_test,np.tile(masks[np.newaxis,:,:,:],[batch_size,1,1,1])),3) # 20,256,256,25

images_input=tf.placeholder(tf.float32,shape=[batch_size,256,256,25])
ground_truth=tf.placeholder(tf.float32,shape=[batch_size,256,256,24])



output6,output12,output_g=deeper_u_net.inference(images_input,0.8,phase_train_placeholder)


L2_loss24=tf.reduce_mean(tf.square(tf.subtract(ground_truth,output_g)))
L2_loss12=tf.reduce_mean(tf.square(tf.subtract(ground_truth[:,:,:,::2],output12)))
L2_loss6=tf.reduce_mean(tf.square(tf.subtract(ground_truth[:,:,:,::4],output6)))
L2_loss=0.1*L2_loss6+0.1*L2_loss12+L2_loss24


real_AB = tf.concat([images_input[:,:,:,0:1], ground_truth], 3)
fake_AB = tf.concat([images_input[:,:,:,0:1], output_g], 3)
D_logits = discriminator.inference(real_AB,1.0,phase_train_placeholder,0.0,reuse=False)
D_logits_ = discriminator.inference(fake_AB,1.0,phase_train_placeholder, 0.0,reuse=True)

d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits,labels=tf.ones_like(D_logits)))
d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_,labels=tf.zeros_like(D_logits_)))
d_loss=d_loss_real+d_loss_fake






g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_,labels=tf.ones_like(D_logits_)))+100*L2_loss


t_vars=tf.trainable_variables()
g_vars=[var for var in t_vars if 'generator' in var.name]
d_vars=[var for var in t_vars if 'discriminator' in var.name]



global_step=tf.Variable(0,trainable=False)

d_optim=tf.train.AdamOptimizer(0.00015,beta1=0.5).minimize(d_loss,global_step=global_step,var_list=d_vars)
g_optim=tf.train.AdamOptimizer(0.00015,beta1=0.5).minimize(g_loss,global_step=global_step,var_list=g_vars)

# Create a saver
saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
#
#
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
#
# ckpt = tf.train.get_checkpoint_state('models/20190830-815')
# saver.restore(sess, ckpt.model_checkpoint_path)

last_round_psnr=[0 for i in range(batch_size)]
for epoch in range(1):
    for i in range(100000000):
        if i % 200 == 0:
            recon_test = sess.run(output_g, feed_dict={images_input: test_input,phase_train_placeholder: False})


            for kk in range(batch_size):
                psnr = 0.0
                for w in range(24):
                    single_mse=np.mean((recon_test[kk,:,:,w] - testing_data[kk,:,:,w]) ** 2)
                    psnr+=20 * math.log10(1 / math.sqrt(single_mse))
                with open(os.path.join(result_dir, 'psnr.txt'), 'at') as f:
                    f.write('PSNR for scene' + str(kk + 1) + ': %.5f\n' % (psnr/24.0))
                if psnr/24.0 > last_round_psnr[kk]:
                    sio.savemat(result_dir + '/test_result'+str(kk+1)+'.mat', {'result': recon_test[kk,:,:,:]})
                    last_round_psnr[kk] = psnr/24.0

        training_data_batch = operation.shuffle_crop(training_data_whole, batch_size=batch_size)  # (10, 256, 256, 24)
        training_data_batch =operation.normalize_0_to_1(training_data_batch)
        measurement_train   = np.sum(training_data_batch * masks, axis=3)[:, :, :, np.newaxis]

        train_input = np.concatenate((measurement_train, np.tile(masks[np.newaxis, :, :, :], [batch_size, 1, 1, 1])), 3)

        out,train_err,_, step = sess.run([output_g,L2_loss,g_optim, global_step], feed_dict={images_input: train_input,
                                                                                ground_truth: training_data_batch,
                                                                                phase_train_placeholder: True})

        _ = sess.run( d_optim, feed_dict={images_input: train_input, phase_train_placeholder: True, ground_truth: training_data_batch})


        print 'training_mse:',train_err
        with open(os.path.join(result_dir, 'psnr.txt'), 'at') as f:
            f.write('train_mse: %.5f\n' % (train_err))

        checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % subdir)
        saver.save(sess, checkpoint_path, global_step=i, write_meta_graph=False)

